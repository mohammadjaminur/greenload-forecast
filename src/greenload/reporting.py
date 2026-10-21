"""Standalone scientific plots from frozen experiment artifacts."""

from __future__ import annotations

import json

import numpy as np

from greenload.config import SIGNALS


def generate_figures(root, state):
    if not state.get("incumbent") or not state.get("initial"):
        return
    selected_path = root / state["incumbent"] / "test_gru_metrics.json"
    if not selected_path.exists():
        return
    selected = json.loads(selected_path.read_text())
    if "forecast" not in selected:
        return  # lightweight test executors need not produce scientific artifacts
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = root / "figures"
    output.mkdir(exist_ok=True)
    methods = {}
    for label, run_id, method in [
        ("Initial GRU", state["initial"], "gru"),
        ("Selected GRU", state["incumbent"], "gru"),
        ("Persistence", "baseline", "persistence"),
        ("Daily seasonal", "baseline", "seasonal_daily"),
        ("Weekly seasonal", "baseline", "seasonal_weekly"),
        ("Ridge", "baseline", "ridge"),
    ]:
        path = root / run_id / f"test_{method}_metrics.json"
        if path.exists():
            methods[label] = json.loads(path.read_text())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    labels = list(methods)
    values = [methods[label]["scheduling"]["total_cost"] for label in labels]
    ax.bar(labels, values)
    ax.set_ylabel("Realized objective cost (all workload served)")
    ax.set_title("Final test: identical workload, capacities, and terminal policy")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output / "test_mpc_cost.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, signal in zip(axes.flat, SIGNALS, strict=True):
        for label, metric in methods.items():
            steps = metric["forecast"][signal]["by_horizon"]
            ax.plot(
                [step["hours_ahead"] for step in steps],
                [step["rmse"] for step in steps],
                label=label,
            )
        ax.set_title(signal)
        ax.set_xlabel("Hours ahead")
        ax.set_ylabel("RMSE in raw units")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "test_forecast_rmse.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, signal in zip(axes.flat, SIGNALS, strict=True):
        for label, reference in (
            ("Initial GRU", state["initial"]),
            ("Selected GRU", state["incumbent"]),
        ):
            history_path = root / reference / f"history_{signal}.json"
            if history_path.exists():
                history = json.loads(history_path.read_text())
                ax.plot(
                    [h["epoch"] for h in history], [h["val_loss"] for h in history], label=label
                )
        ax.set_title(signal)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Deployment-mode validation MSE")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "training_validation_curves.png", dpi=180)
    plt.close(fig)

    selected_predictions = root / state["incumbent"] / "test_gru_predictions.npz"
    if selected_predictions.exists():
        from greenload.artifacts import load_data

        raw, _ = load_data(root / state["incumbent"])
        with np.load(selected_predictions) as saved:
            origin = int(saved["origins"][0])
            fig, axes = plt.subplots(2, 2, figsize=(11, 7))
            hours = np.arange(1, selected["config"]["window_size"] + 1)
            for ax, signal in zip(axes.flat, SIGNALS, strict=True):
                ax.plot(hours, raw[signal][origin + hours, 0], label="Actual")
                ax.plot(hours, saved[signal][0, 0], label="Selected GRU")
                ax.set_title(f"{signal}: location 0, first test origin")
                ax.set_xlabel("Hours ahead")
                ax.legend()
            fig.tight_layout()
            fig.savefig(output / "example_test_forecast.png", dpi=180)
            plt.close(fig)
