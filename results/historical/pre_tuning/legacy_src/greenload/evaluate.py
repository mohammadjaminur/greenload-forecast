"""Evaluate forecasters (closed-loop rollout) and the downstream MPC scheduler.

Four arms are compared, all evaluated identically:
  * naive       -- zero-order-hold baseline (no learning).
  * gru_baseline -- GRU trained with pure teacher forcing (the original bug).
  * gru_fixed    -- GRU trained with scheduled sampling (the fix).
  * offline_oracle -- perfect-foresight upper bound (uses true future data,
    not a forecaster at all; the ceiling nothing with imperfect information
    can beat).

Models are trained and evaluated in normalized (z-score) space -- see
`data/dataset.py` -- but every number in this script is converted back to
real-world units (via `SignalScaler.denormalize`) before it's reported or
handed to the MPC solver, which needs physically meaningful prices/water/
carbon/workload values.

Everything this script computes is written once to `results/metrics.json`.
Figures and the README both read from that file -- nothing is hand-typed,
which is the structural fix for the original repo's worst bug (headline
comparison-chart numbers that didn't match the notebook's own computed
output).

Usage:
    python -m greenload.evaluate
    python -m greenload.evaluate --smoke-test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from greenload.config import DEFAULT, SIGNALS, Config
from greenload.data.dataset import SignalSplits, build_all_splits
from greenload.data.generate_synthetic_data import generate_raw_series
from greenload.models.gru_forecaster import GRUForecaster
from greenload.models.naive_baseline import NaiveForecaster
from greenload.scheduler import mpc
from greenload.train import smoke_config

FORECAST_MODES = ("naive", "gru_baseline", "gru_fixed")


def load_model(signal: str, mode: str, cfg: Config, checkpoints_dir: Path, device: str):
    if mode == "naive":
        return NaiveForecaster(horizon=cfg.window_size)
    model = GRUForecaster(
        hidden_size=cfg.hidden_sizes[signal],
        num_layers=cfg.num_layers,
        horizon=cfg.window_size,
        dropout=cfg.dropout,
    ).to(device)
    ckpt_mode = "baseline" if mode == "gru_baseline" else "fixed"
    state = torch.load(
        checkpoints_dir / ckpt_mode / f"{signal}.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    return model


def closed_loop_rollout(model, source: torch.Tensor, device: str) -> torch.Tensor:
    """Fully autoregressive rollout -- the real inference path, `target=None`."""
    with torch.no_grad():
        pred, _ = model(source.to(device), target=None)
    return pred.cpu()


def per_step_mse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return ((pred - target) ** 2).mean(axis=(0, 1))


def nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    rmse = np.sqrt(((pred - target) ** 2).mean())
    return float(rmse / (target.std() + 1e-8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = smoke_config(DEFAULT) if args.smoke_test else DEFAULT
    device = cfg.device
    checkpoints_dir = args.out_dir / "checkpoints"
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_series = generate_raw_series(cfg)  # deterministic given cfg.seed -- same data train.py used
    all_splits: dict[str, SignalSplits] = build_all_splits(raw_series, cfg)

    # ---- forecast accuracy: closed-loop rollout on the FULL test set, real units ----
    forecast_metrics: dict[str, dict[str, dict]] = {mode: {} for mode in FORECAST_MODES}
    predictions: dict[str, dict[str, np.ndarray]] = {mode: {} for mode in FORECAST_MODES}
    actuals: dict[str, np.ndarray] = {}

    for signal in SIGNALS:
        test_split = all_splits[signal].test
        scaler = all_splits[signal].scaler
        num_loc = test_split.num_loc
        num_windows = test_split.num_windows

        actual = scaler.denormalize(test_split.target.numpy())
        actuals[signal] = actual.reshape(num_windows, num_loc, cfg.window_size)

        for mode in FORECAST_MODES:
            model = load_model(signal, mode, cfg, checkpoints_dir, device)
            pred_norm = closed_loop_rollout(model, test_split.source, device).numpy()
            pred = scaler.denormalize(pred_norm)

            forecast_metrics[mode][signal] = {
                "per_step_mse": per_step_mse(pred, actual).tolist(),
                "nrmse": nrmse(pred, actual),
            }
            # Physical signals (price/water/carbon/workload) can never be negative,
            # but a plain linear output layer is unbounded. Clip only the copy
            # handed to the MPC solver -- without this, an unlucky prediction can
            # make the `sum(y) == sum(workload)` equality infeasible (no nonnegative
            # allocation can sum to a negative predicted workload) and crash the
            # scheduling loop. Forecast-accuracy metrics above use the unclipped
            # value, so a garbage/negative prediction is still scored honestly.
            scheduler_input = np.clip(pred, a_min=0.0, a_max=None)
            predictions[mode][signal] = scheduler_input.reshape(
                num_windows, num_loc, cfg.window_size
            )

    # ---- downstream scheduling: receding-horizon MPC with each forecaster ----
    price_test = all_splits["price"].test
    true_price = (
        all_splits["price"]
        .scaler.denormalize(
            price_test.source.numpy().reshape(price_test.num_windows, price_test.num_loc)
        )
        .T
    )
    true_water = (
        all_splits["water"]
        .scaler.denormalize(
            all_splits["water"]
            .test.source.numpy()
            .reshape(price_test.num_windows, price_test.num_loc)
        )
        .T
    )
    true_carbon = (
        all_splits["carbon"]
        .scaler.denormalize(
            all_splits["carbon"]
            .test.source.numpy()
            .reshape(price_test.num_windows, price_test.num_loc)
        )
        .T
    )
    true_workload = (
        all_splits["workload"]
        .scaler.denormalize(
            all_splits["workload"].test.source.numpy().reshape(price_test.num_windows, 1)
        )
        .T
    )

    schedule_metrics: dict[str, dict] = {}
    for mode in FORECAST_MODES:
        allocation = mpc.run_receding_horizon(
            predicted_price=predictions[mode]["price"],
            predicted_water=predictions[mode]["water"],
            predicted_carbon=predictions[mode]["carbon"],
            predicted_workload=predictions[mode]["workload"],
            true_price=true_price,
            true_water=true_water,
            true_carbon=true_carbon,
            true_workload=true_workload,
            l1=cfg.l1_water,
            l2=cfg.l2_carbon,
            max_cap=cfg.max_cap,
        )
        schedule_metrics[mode] = mpc.evaluate_allocation(
            allocation, true_price, true_water, true_carbon, cfg.l1_water, cfg.l2_carbon
        )

    _, oracle_allocation = mpc.offline_oracle(
        true_price, true_water, true_carbon, true_workload, cfg.l1_water, cfg.l2_carbon, cfg.max_cap
    )
    schedule_metrics["offline_oracle"] = mpc.evaluate_allocation(
        oracle_allocation, true_price, true_water, true_carbon, cfg.l1_water, cfg.l2_carbon
    )

    oracle_total = schedule_metrics["offline_oracle"]["total_cost"]
    for method in FORECAST_MODES:
        gap = (schedule_metrics[method]["total_cost"] - oracle_total) / oracle_total * 100
        schedule_metrics[method]["optimality_gap_pct"] = gap
    schedule_metrics["offline_oracle"]["optimality_gap_pct"] = 0.0

    # ---- write single source of truth ----
    metrics = {"forecast": forecast_metrics, "scheduling": schedule_metrics}
    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics["scheduling"], indent=2))

    if not args.smoke_test:
        plot_forecast_error_by_step(forecast_metrics, figures_dir / "forecast_error_by_step.png")
        plot_forecast_vs_actual(actuals, predictions, figures_dir / "forecast_vs_actual.png")
        plot_cost_comparison(schedule_metrics, figures_dir / "cost_comparison.png")

    print(f"Saved metrics to {args.out_dir / 'metrics.json'}, figures to {figures_dir}")


def plot_forecast_error_by_step(forecast_metrics: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, signal in zip(axes.flat, SIGNALS, strict=True):
        for mode in FORECAST_MODES:
            mse = forecast_metrics[mode][signal]["per_step_mse"]
            ax.plot(range(1, len(mse) + 1), mse, label=mode)
        ax.set_title(signal)
        ax.set_xlabel("steps ahead")
        ax.set_ylabel("MSE (real units)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Closed-loop rollout error vs. forecast horizon (compounding error = exposure bias)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_forecast_vs_actual(actuals: dict, predictions: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, signal in zip(axes.flat, SIGNALS, strict=True):
        ax.plot(actuals[signal][0, 0], label="actual", linewidth=2)
        for mode in ("gru_baseline", "gru_fixed"):
            ax.plot(predictions[mode][signal][0, 0], "--", label=mode)
        ax.set_title(signal)
        ax.set_xlabel("hour")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Example 24h closed-loop rollout: location 0, first test window")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cost_comparison(schedule_metrics: dict, out_path: Path) -> None:
    methods = ["naive", "gru_baseline", "gru_fixed", "offline_oracle"]
    labels = [
        "Naive\n(zero-order-hold)",
        "GRU-MPC\n(pre-fix)",
        "GRU-MPC\n(fixed)",
        "Offline\noracle",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, cost_key, title in zip(
        axes,
        ["price_cost", "water_cost", "carbon_cost"],
        ["Price cost (USD)", "Water cost", "Carbon cost"],
        strict=True,
    ):
        values = [schedule_metrics[m][cost_key] for m in methods]
        colors = ["#999999", "#e07b39", "#2c7fb8", "#2ca25f"]
        ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Receding-horizon MPC cost: naive vs. pre-fix GRU vs. fixed GRU vs. oracle")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
