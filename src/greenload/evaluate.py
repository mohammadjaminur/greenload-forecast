"""Evaluate saved run configuration on validation or explicitly authorized test data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from greenload.artifacts import check_deadline, fingerprint, load_data, write_json
from greenload.config import SIGNALS, Config
from greenload.data.dataset import SignalScaler, inputs_at
from greenload.models.baselines import persistence, ridge_predict
from greenload.models.gru_forecaster import make_model
from greenload.scheduler import mpc
from greenload.train import inference


def forecast_scores(pred, raw, origins, start, end):
    times = origins[:, None] + np.arange(1, pred.shape[-1] + 1)
    mask = (times >= start) & (times < end)
    safe_times = np.clip(times, 0, len(raw) - 1)
    actual = raw[safe_times].transpose(0, 2, 1)
    valid = np.broadcast_to(mask[:, None], pred.shape)
    errors = pred - actual
    scale = float(raw[start:end].std()) + 1e-8
    mse = float(np.mean(errors[valid] ** 2))
    by_horizon, by_location = [], []
    for step in range(pred.shape[-1]):
        selected = errors[:, :, step][valid[:, :, step]]
        by_horizon.append(
            {
                "hours_ahead": step + 1,
                "rmse": float(np.sqrt(np.mean(selected**2))) if selected.size else None,
                "mae": float(np.mean(np.abs(selected))) if selected.size else None,
                "count": int(selected.size),
            }
        )
    for loc in range(raw.shape[1]):
        selected = errors[:, loc][mask]
        rmse = float(np.sqrt(np.mean(selected**2)))
        by_location.append(
            {
                "location": loc,
                "rmse": rmse,
                "mae": float(np.mean(np.abs(selected))),
                "nrmse": rmse / (float(raw[start:end, loc].std()) + 1e-8),
            }
        )
    result = {
        "rmse": float(np.sqrt(mse)),
        "nrmse": float(np.sqrt(mse) / scale),
        "mae": float(np.mean(np.abs(errors[valid]))),
        "by_horizon": by_horizon,
        "by_location": by_location,
        "scoring": "all available within-period future targets; masked at the endpoint",
    }
    if raw.shape[1] > 1:
        predicted_ranks = pred.argsort(axis=1).argsort(axis=1)
        actual_ranks = actual.argsort(axis=1).argsort(axis=1)
        result["mean_absolute_rank_error"] = float(
            np.mean(np.abs(predicted_ranks - actual_ranks)[valid])
        )
        result["cheapest_location_accuracy"] = float(
            np.mean((pred.argmin(axis=1) == actual.argmin(axis=1))[mask])
        )
    return result


def gru_predictions(run_dir, raw, origins, cfg, checksum):
    predictions = {}
    for signal in SIGNALS:
        checkpoint = torch.load(
            run_dir / "checkpoints" / f"{signal}.pt",
            map_location=cfg.device,
            weights_only=True,
        )
        if checkpoint["config"] != cfg.to_dict() or checkpoint["dataset_checksum"] != checksum:
            raise ValueError("Checkpoint configuration or dataset mismatch")
        scaler = SignalScaler.from_dict(checkpoint["scaler"])
        source, future = inputs_at(scaler.normalize(raw[signal]), origins, cfg)
        model = make_model(cfg)
        model.load_state_dict(checkpoint["model"])
        normalized = inference(model, source, future, cfg).numpy()
        predictions[signal] = scaler.predictions_to_raw(
            normalized.reshape(len(origins), raw[signal].shape[1], cfg.window_size)
        )
    return predictions, {}


def baseline_predictions(method, raw, origins, cfg):
    predictions, details = {}, {}
    for signal in SIGNALS:
        if method == "ridge":
            predictions[signal], details[signal] = ridge_predict(raw[signal], origins, cfg)
        else:
            period = {"persistence": None, "seasonal_daily": 24, "seasonal_weekly": 168}[method]
            predictions[signal] = persistence(raw[signal], origins, cfg.window_size, period)
    return predictions, details


def oracle_metrics(raw, cfg, split, cache_dir, checksum, deadline=None):
    start, end = cfg.bounds(split)
    key = fingerprint(
        {
            "dataset": checksum,
            "split": [start, end],
            "horizon": cfg.window_size,
            "l1": cfg.l1_water,
            "l2": cfg.l2_carbon,
            "max_cap": cfg.max_cap,
            "upper_bound": cfg.workload_upper_bound,
            "scheduler": "robust-reserve-v1",
        }
    )
    path = cache_dir / (key + ".json")
    if path.exists():
        return json.loads(path.read_text())
    check_deadline(deadline)
    truth = {signal: raw[signal][start:end].T for signal in SIGNALS}
    value, allocation = mpc.offline_oracle(
        truth["price"],
        truth["water"],
        truth["carbon"],
        truth["workload"],
        cfg.l1_water,
        cfg.l2_carbon,
        cfg.max_cap,
        workload_upper_bound=cfg.workload_upper_bound,
        forecast_horizon=cfg.window_size,
    )
    result = mpc.evaluate_allocation(
        allocation,
        truth["price"],
        truth["water"],
        truth["carbon"],
        cfg.l1_water,
        cfg.l2_carbon,
    )
    result["diagnostics"] = mpc.allocation_diagnostics(allocation, truth["workload"], cfg.max_cap)
    if not result["diagnostics"]["valid"]:
        raise mpc.SchedulingError("Oracle allocation failed feasibility checks")
    result["solver"] = mpc.SOLVER
    result["objective"] = value
    write_json(path, result)
    return result


def evaluate_run(
    run_dir: Path,
    split="val",
    method="gru",
    deadline=None,
    cache_dir=None,
    allow_test=False,
):
    if split == "test" and not allow_test:
        raise ValueError("Test evaluation requires a frozen selection or explicit --allow-test")
    cfg = Config.load(run_dir / "config.json")
    torch.set_num_threads(cfg.torch_threads)
    metrics_path = run_dir / f"{split}_{method}_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    started = time.monotonic()
    check_deadline(deadline)
    raw, metadata = load_data(run_dir)
    start, end = cfg.bounds(split)
    origins = np.arange(start - 1, end)
    if method == "gru":
        predictions, details = gru_predictions(run_dir, raw, origins, cfg, metadata["checksum"])
    else:
        predictions, details = baseline_predictions(method, raw, origins, cfg)
    forecast = {
        signal: forecast_scores(predictions[signal], raw[signal], origins, start, end)
        for signal in SIGNALS
    }
    truth = {signal: raw[signal][start:end].T for signal in SIGNALS}
    allocation, diagnostics = mpc.run_receding_horizon(
        *[predictions[signal][1:] for signal in ("price", "water", "carbon", "workload")],
        *[truth[signal] for signal in ("price", "water", "carbon", "workload")],
        cfg.l1_water,
        cfg.l2_carbon,
        cfg.max_cap,
        workload_upper_bound=cfg.workload_upper_bound,
        return_diagnostics=True,
        deadline=deadline,
    )
    scheduling = mpc.evaluate_allocation(
        allocation,
        truth["price"],
        truth["water"],
        truth["carbon"],
        cfg.l1_water,
        cfg.l2_carbon,
    )
    scheduling["diagnostics"] = diagnostics
    oracle = oracle_metrics(
        raw,
        cfg,
        split,
        cache_dir or run_dir.parent / "oracle_cache",
        metadata["checksum"],
        deadline,
    )
    scheduling["optimality_gap_pct"] = (scheduling["total_cost"] / oracle["total_cost"] - 1) * 100
    if scheduling["optimality_gap_pct"] < -1e-3:
        raise mpc.SchedulingError("MPC cost below the matching feasible oracle")
    result = {
        "method": method,
        "split": split,
        "config": cfg.to_dict(),
        "dataset_checksum": metadata["checksum"],
        "forecast": forecast,
        "mean_nrmse": float(np.mean([forecast[s]["nrmse"] for s in SIGNALS])),
        "scheduling": scheduling,
        "offline_oracle": oracle,
        "baseline_details": details,
        "elapsed_seconds": time.monotonic() - started,
    }
    np.savez_compressed(
        run_dir / f"{split}_{method}_predictions.npz",
        **predictions,
        origins=origins,
        allocation=allocation,
    )
    write_json(metrics_path, result)
    print(
        f"{run_dir.name} {split} {method}: cost={scheduling['total_cost']:.2f} "
        f"gap={scheduling['optimality_gap_pct']:.3f}% "
        f"mean_nrmse={result['mean_nrmse']:.4f}",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", "--out-dir", type=Path, default=Path("results/manual"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--method",
        choices=("gru", "persistence", "seasonal_daily", "seasonal_weekly", "ridge"),
        default="gru",
    )
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="settings come from checkpoint")
    args = parser.parse_args()
    evaluate_run(args.run_dir, args.split, args.method, allow_test=args.allow_test)


if __name__ == "__main__":
    main()
