"""Resumable, time-bounded coordinate search with a frozen final holdout."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from greenload.artifacts import (
    BudgetExceeded,
    fingerprint,
    prepare_data,
    provenance,
    write_json,
)
from greenload.config import DEFAULT, Config, smoke_config
from greenload.evaluate import evaluate_run
from greenload.train import train_run

BASELINES = ("persistence", "seasonal_daily", "seasonal_weekly", "ridge")
GROUPS = (
    ("history", ({"history_hours": 24}, {"history_hours": 48}, {"history_hours": 168})),
    ("hidden_size", ({"hidden_size": 32}, {"hidden_size": 64})),
    ("layers_dropout", ({"num_layers": 1, "dropout": 0.0}, {"num_layers": 2, "dropout": 0.1})),
    ("learning_rate", ({"learning_rate": 0.0003}, {"learning_rate": 0.001})),
    ("normalization", ({"normalization": "global"}, {"normalization": "per_location"})),
    ("decoder", ({"decoder": "direct"}, {"decoder": "autoregressive"})),
)


def better(candidate, incumbent):
    """Cost first; forecast error breaks numerical cost ties."""
    a = candidate["scheduling"]["total_cost"]
    b = incumbent["scheduling"]["total_cost"]
    if not candidate["scheduling"]["diagnostics"]["valid"]:
        return False
    return a < b - 1e-6 or (
        abs(a - b) <= 1e-6 and candidate["mean_nrmse"] < incumbent["mean_nrmse"]
    )


def meaningful_gain(before, after, threshold=0.005):
    return (before - after) / before >= threshold


def stopping_reason(no_improvement_rounds, unseen, elapsed, search_seconds):
    if elapsed >= search_seconds:
        return "search_time_budget"
    if no_improvement_rounds >= 3:
        return "validation_plateau"
    if not unseen:
        return "untried_neighbor_exhaustion"
    return None


def worker(run_dir, stage, method, deadline, cache_dir):
    cfg = Config.load(run_dir / "config.json")
    if stage == "train":
        train_run(cfg, run_dir, deadline)
    elif not (run_dir / "dataset.json").exists():
        prepare_data(cfg, run_dir)
    result = evaluate_run(
        run_dir,
        "test" if stage == "test" else "val",
        method,
        deadline,
        cache_dir=cache_dir,
        allow_test=stage == "test",
    )
    write_json(
        run_dir / "status.json",
        {
            "status": "complete",
            "stage": stage,
            "method": method,
            "valid": result["scheduling"]["diagnostics"]["valid"],
        },
    )


def execute(run_dir, stage, method, seconds, cache_dir):
    """A subprocess enforces the wall limit even during an individual solver call."""
    if seconds <= 0:
        raise BudgetExceeded("No computation budget remaining")
    command = [
        sys.executable,
        "-m",
        "greenload.experiments",
        "--worker",
        str(run_dir),
        "--stage",
        stage,
        "--method",
        method,
        "--worker-seconds",
        str(seconds),
        "--cache-dir",
        str(cache_dir),
    ]
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
    started = time.monotonic()
    with (run_dir / "console.log").open("a") as log:
        log.write("\n" + " ".join(command) + "\n")
        log.flush()
        try:
            completed = subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            write_json(
                run_dir / "status.json",
                {
                    "status": "timed_out",
                    "stage": stage,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            raise BudgetExceeded("Worker stopped at computation deadline") from exc
    if completed.returncode:
        status_path = run_dir / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get("status") == "timed_out":
            raise BudgetExceeded(status.get("error", "Worker exhausted computation budget"))
        write_json(
            status_path,
            {
                "status": "failed",
                "stage": stage,
                "returncode": completed.returncode,
                "error": (run_dir / "console.log").read_text()[-4000:],
            },
        )
        raise RuntimeError(f"Worker failed; see {run_dir / 'console.log'}")


def run_search(
    root: Path,
    cfg: Config = DEFAULT,
    budget_seconds=14400.0,
    executor=execute,
):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "search_state.json"
    identity = provenance()
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["source_sha256"] != identity["source_sha256"]:
            raise ValueError("Source changed since this search; use a new experiment directory")
        if state["base_config"] != cfg.to_dict():
            raise ValueError("Search configuration differs from the saved configuration")
        if state["phase"] == "complete":
            generate_report(root, state)
            return state
        if state["budget_seconds"] != budget_seconds:
            raise ValueError("Resume with the original wall-clock budget")
        if any(row["status"] == "running" for row in state["attempts"].values()):
            # A hard process kill cannot flush elapsed time. Charge time since the
            # last durable update conservatively, including any unattended downtime.
            state["elapsed_seconds"] += max(
                0, time.time() - state.get("saved_at_unix", time.time())
            )
    else:
        state = {
            "base_config": cfg.to_dict(),
            "source_sha256": identity["source_sha256"],
            "budget_seconds": budget_seconds,
            "elapsed_seconds": 0.0,
            "phase": "search",
            "attempts": {},
            "incumbent": None,
            "initial": None,
            "group_index": 0,
            "rounds": [],
            "no_improvement_rounds": 0,
            "round_start_cost": None,
            "round_new_attempts": 0,
        }
        write_json(root / "provenance.json", identity)
        write_json(root / "base_config.json", cfg.to_dict())
    invocation_start = time.monotonic()
    previous_elapsed = state["elapsed_seconds"]
    search_seconds = budget_seconds * 0.75  # final quarter reserved for confirmation/reporting
    work_seconds = budget_seconds - min(60, budget_seconds * 0.02)

    def elapsed():
        return previous_elapsed + time.monotonic() - invocation_start

    def persist():
        state["elapsed_seconds"] = elapsed()
        state["saved_at_unix"] = time.time()
        write_json(state_path, state)

    def attempt(settings, method="gru", split="val", limit=None):
        run_id = ("gru-" + fingerprint(settings.to_dict())) if method == "gru" else "baseline"
        run_dir = root / run_id
        run_dir.mkdir(exist_ok=True)
        write_json(run_dir / "config.json", settings.to_dict())
        if not (run_dir / "provenance.json").exists():
            write_json(run_dir / "provenance.json", identity)
        key = f"{run_id}:{split}:{method}"
        path = run_dir / f"{split}_{method}_metrics.json"
        if key in state["attempts"] and state["attempts"][key]["status"] == "complete":
            if not path.exists():
                raise ValueError(f"Cached result is missing: {path}")
            return run_id, json.loads(path.read_text())
        seconds = min(work_seconds, work_seconds if limit is None else limit) - elapsed()
        if seconds <= 0:
            raise BudgetExceeded("Stage budget exhausted")
        print(f"Starting {key}; {seconds / 60:.1f} minutes left in stage", flush=True)
        prior_attempt = state["attempts"].get(key)
        retries = list(prior_attempt.get("previous_attempts", [])) if prior_attempt else []
        if prior_attempt:
            retries.append(
                {
                    name: value
                    for name, value in prior_attempt.items()
                    if name not in ("previous_attempts", "config")
                }
            )
        state["attempts"][key] = {
            "run_id": run_id,
            "method": method,
            "split": split,
            "seed": settings.seed,
            "config": settings.to_dict(),
            "status": "running",
            "previous_attempts": retries,
            "change_from_incumbent": (
                {
                    name: [
                        state["attempts"][f"{state['incumbent']}:val:gru"]["config"][name],
                        value,
                    ]
                    for name, value in settings.to_dict().items()
                    if state["attempts"][f"{state['incumbent']}:val:gru"]["config"][name] != value
                }
                if state["incumbent"]
                else {}
            ),
        }
        persist()
        started = time.monotonic()
        try:
            stage = "test" if split == "test" else ("train" if method == "gru" else "baseline")
            executor(run_dir, stage, method, seconds, root / "oracle_cache")
            result = json.loads(path.read_text())
            if not result["scheduling"]["diagnostics"]["valid"]:
                raise ValueError("Schedule failed feasibility checks")
            state["attempts"][key].update(
                status="complete",
                total_cost=result["scheduling"]["total_cost"],
                mean_nrmse=result["mean_nrmse"],
                gap_pct=result["scheduling"]["optimality_gap_pct"],
            )
            print(
                f"Completed {key}: cost={result['scheduling']['total_cost']:.2f}, "
                f"mean NRMSE={result['mean_nrmse']:.4f}",
                flush=True,
            )
            return run_id, result
        except Exception as exc:
            state["attempts"][key].update(
                status="timed_out" if isinstance(exc, BudgetExceeded) else "failed",
                error=repr(exc),
            )
            if isinstance(exc, BudgetExceeded):
                raise
            print(f"Failed {key}: {exc}", flush=True)
            return run_id, None
        finally:
            state["attempts"][key]["elapsed_seconds"] = time.monotonic() - started
            persist()

    try:
        if state["phase"] == "search":
            for method in BASELINES:
                attempt(cfg, method, limit=search_seconds)
            if state["initial"] is None:
                initial_id, initial = attempt(cfg, limit=search_seconds)
                if initial is None:
                    raise RuntimeError(
                        "Initial corrected GRU failed; cannot establish an incumbent"
                    )
                state["initial"] = state["incumbent"] = initial_id
                state["round_start_cost"] = initial["scheduling"]["total_cost"]
                persist()
            while state["phase"] == "search":
                current = Config.load(root / state["incumbent"] / "config.json")
                incumbent_path = root / state["incumbent"] / "val_gru_metrics.json"
                incumbent = json.loads(incumbent_path.read_text())
                _, alternatives = GROUPS[state["group_index"]]
                # Freeze this group's anchor so every alternative changes only this coordinate.
                anchor_key = "group_anchor"
                anchor = Config(**state.get(anchor_key, current.to_dict()))
                state[anchor_key] = anchor.to_dict()
                persist()
                for changes in alternatives:
                    candidate_cfg = replace(anchor, **changes)
                    candidate_id = "gru-" + fingerprint(candidate_cfg.to_dict())
                    key = f"{candidate_id}:val:gru"
                    if candidate_id == state["incumbent"]:
                        continue
                    if key in state["attempts"] and state["attempts"][key]["status"] in (
                        "complete",
                        "failed",
                    ):
                        # Reconsider a cached neighbor, since the incumbent may have moved.
                        if state["attempts"][key]["status"] == "failed":
                            continue
                        candidate = json.loads(
                            (root / candidate_id / "val_gru_metrics.json").read_text()
                        )
                    else:
                        state["round_new_attempts"] += 1
                        candidate_id, candidate = attempt(candidate_cfg, limit=search_seconds)
                    if candidate and better(candidate, incumbent):
                        state["incumbent"], incumbent = candidate_id, candidate
                        state["attempts"][key]["selection_reason"] = "lower_validation_cost"
                    elif candidate:
                        state["attempts"][key].setdefault(
                            "selection_reason", "no_validation_cost_improvement"
                        )
                    persist()
                state.pop(anchor_key, None)
                state["group_index"] += 1
                if state["group_index"] == len(GROUPS):
                    cost = incumbent["scheduling"]["total_cost"]
                    improved = meaningful_gain(state["round_start_cost"], cost)
                    state["no_improvement_rounds"] = (
                        0 if improved else state["no_improvement_rounds"] + 1
                    )
                    state["rounds"].append(
                        {
                            "round": len(state["rounds"]) + 1,
                            "starting_cost": state["round_start_cost"],
                            "ending_cost": cost,
                            "meaningful_improvement": improved,
                            "new_attempts": state["round_new_attempts"],
                        }
                    )
                    reason = stopping_reason(
                        state["no_improvement_rounds"],
                        state["round_new_attempts"] > 0,
                        elapsed(),
                        search_seconds,
                    )
                    state.update(
                        group_index=0,
                        round_start_cost=cost,
                        round_new_attempts=0,
                    )
                    if reason:
                        state["stopping_reason"] = reason
                        state["phase"] = "confirmation"
                persist()
    except BudgetExceeded:
        state["stopping_reason"] = "search_time_budget"
        state["phase"] = "confirmation"
        persist()
    except Exception as exc:
        state["error"] = repr(exc)
        persist()
        generate_report(root, state)
        raise

    try:
        if state["initial"] is not None:
            selection_path = root / "selection.json"
            if not selection_path.exists():
                write_json(
                    selection_path,
                    {
                        "run_id": state["incumbent"],
                        "config": Config.load(root / state["incumbent"] / "config.json").to_dict(),
                        "criterion": "validation MPC cost; mean forecast NRMSE breaks cost ties",
                        "stopping_reason": state["stopping_reason"],
                        "test_used_for_selection": False,
                        "source_sha256": state["source_sha256"],
                    },
                )
            selection = json.loads(selection_path.read_text())
            if selection["run_id"] != state["incumbent"]:
                raise ValueError("Frozen selection disagrees with search state")
            chosen = Config(**selection["config"])
            # Leave time for test evaluation. Incomplete seed runs remain visible in the ledger.
            remaining = max(0, budget_seconds - elapsed())
            confirmation_limit = budget_seconds - min(600, remaining * 0.25)
            if state["phase"] == "confirmation":
                for seed in (42, 43, 44):
                    for settings in dict.fromkeys(
                        (replace(chosen, seed=seed), replace(cfg, seed=seed))
                    ):
                        try:
                            attempt(settings, limit=confirmation_limit)
                        except BudgetExceeded:
                            state["confirmation_incomplete"] = True
                            break
                    else:
                        continue
                    break
                state["phase"] = "test"
                persist()
            # Freeze was written before this point. Test results never change chosen settings.
            if state["phase"] == "test":
                test_settings = [chosen, cfg]
                for seed in (42, 43, 44):
                    test_settings.extend((replace(chosen, seed=seed), replace(cfg, seed=seed)))
                seen = set()
                for settings in test_settings:
                    run_id = "gru-" + fingerprint(settings.to_dict())
                    if run_id in seen:
                        continue
                    seen.add(run_id)
                    key = f"{run_id}:val:gru"
                    if state["attempts"].get(key, {}).get("status") != "complete":
                        continue
                    attempt(settings, split="test")
                for method in BASELINES:
                    attempt(cfg, method, split="test")
        else:
            state["confirmation_incomplete"] = True
    except BudgetExceeded:
        state["final_evaluation_incomplete"] = True
    finally:
        state["phase"] = "complete"
        persist()
        generate_report(root, state)
        persist()
    return state


def generate_report(root, state):
    rows = list(state["attempts"].values())
    columns = (
        "run_id",
        "split",
        "method",
        "seed",
        "status",
        "total_cost",
        "gap_pct",
        "mean_nrmse",
        "elapsed_seconds",
        "selection_reason",
        "history_hours",
        "hidden_size",
        "num_layers",
        "dropout",
        "learning_rate",
        "normalization",
        "decoder",
    )
    with (root / "leaderboard.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {**row["config"], **row}
            for row in sorted(
                rows, key=lambda row: (row["split"], row.get("total_cost", float("inf")))
            )
        )
    lines = [
        "# GreenLoad experiment report",
        "",
        f"Search stopping reason: **{state.get('stopping_reason', 'in progress')}**.",
        f"Total computation: {state['elapsed_seconds'] / 60:.1f} minutes; "
        f"budget: {state['budget_seconds'] / 3600:.2f} hours.",
        "",
        "Selection uses validation MPC cost. The final test period is never used for tuning.",
        "All reported schedules must complete the same workload, obey capacities and causality, "
        "and use the same finite-period reserve rule as the offline oracle.",
        "Historical pre-tuning metrics use a different protocol and are not comparable.",
        "",
        f"Initial GRU: `{state.get('initial')}`. Selected GRU: `{state.get('incumbent')}`.",
        "",
    ]
    if state.get("incumbent"):
        settings = Config.load(root / state["incumbent"] / "config.json")
        lines.extend(
            [
                "Selected setup: "
                + ", ".join(
                    f"{name}={getattr(settings, name)}"
                    for name in (
                        "history_hours",
                        "hidden_size",
                        "num_layers",
                        "dropout",
                        "learning_rate",
                        "normalization",
                        "decoder",
                    )
                )
                + ".",
                "",
            ]
        )
    for split in ("val", "test"):
        lines += [
            f"## {'Validation' if split == 'val' else 'Final test'}",
            "",
            "| Run / method | Seed | MPC cost | Oracle gap | Mean NRMSE |",
            "|---|---:|---:|---:|---:|",
        ]
        valid = sorted(
            [row for row in rows if row["split"] == split and row["status"] == "complete"],
            key=lambda row: row["total_cost"],
        )
        for row in valid:
            label = row["method"] if row["method"] != "gru" else row["run_id"]
            lines.append(
                f"| {label} | {row['seed']} | {row['total_cost']:.2f} | "
                f"{row['gap_pct']:.3f}% | {row['mean_nrmse']:.4f} |"
            )
        lines.append("")
        if state.get("incumbent") and state.get("initial"):
            selected = state["attempts"].get(f"{state['incumbent']}:{split}:gru", {})
            initial = state["attempts"].get(f"{state['initial']}:{split}:gru", {})
            if selected.get("status") == initial.get("status") == "complete":
                delta = (initial["total_cost"] - selected["total_cost"]) / initial["total_cost"]
                lines += [
                    f"Selected GRU cost reduction relative to the initial corrected GRU: "
                    f"**{100 * delta:.3f}%**.",
                    "",
                ]
    selected_test_path = root / str(state.get("incumbent")) / "test_gru_metrics.json"
    if selected_test_path.exists():
        selected_test = json.loads(selected_test_path.read_text())
        selected_cost = selected_test["scheduling"]["total_cost"]
        comparisons = {}
        for method in BASELINES:
            path = root / "baseline" / f"test_{method}_metrics.json"
            if path.exists():
                comparisons[method] = json.loads(path.read_text())
        if comparisons:
            strongest = min(comparisons, key=lambda m: comparisons[m]["scheduling"]["total_cost"])
            best_cost = comparisons[strongest]["scheduling"]["total_cost"]
            difference = (selected_cost / best_cost - 1) * 100
            lines += [
                f"Lowest-cost simple baseline on the final test: **{strongest}**. "
                f"Selected GRU cost relative to that baseline: **{difference:+.3f}%**.",
                "",
            ]
        if "forecast" in selected_test:
            lines += [
                "## Selected GRU forecast errors",
                "",
                "| Signal | NRMSE | RMSE | MAE |",
                "|---|---:|---:|---:|",
            ]
            for signal, score in selected_test["forecast"].items():
                lines.append(
                    f"| {signal} | {score['nrmse']:.4f} | {score['rmse']:.4f} | "
                    f"{score['mae']:.4f} |"
                )
            lines.append("")
    lines += ["## Seed confirmation", ""]
    for label, reference in (
        ("Selected", state.get("incumbent")),
        ("Initial", state.get("initial")),
    ):
        if not reference:
            continue
        settings = Config.load(root / reference / "config.json")
        for split in ("val", "test"):
            values = []
            completed_seeds = []
            for seed in (42, 43, 44):
                run_id = "gru-" + fingerprint(replace(settings, seed=seed).to_dict())
                row = state["attempts"].get(f"{run_id}:{split}:gru", {})
                if row.get("status") == "complete":
                    values.append(row["total_cost"])
                    completed_seeds.append(seed)
            if values:
                std = f"{np.std(values, ddof=1):.2f}" if len(values) > 1 else "not estimable"
                lines.append(
                    f"- {label}, {split}: seeds {completed_seeds}; mean cost "
                    f"{np.mean(values):.2f}; sample standard deviation {std}."
                )
    lines += ["", "## Search rounds", ""]
    for record in state["rounds"]:
        lines.append(
            f"- Round {record['round']}: {record['starting_cost']:.2f} → "
            f"{record['ending_cost']:.2f}; {record['new_attempts']} new setups; "
            f"meaningful improvement: {record['meaningful_improvement']}."
        )
    failures = [row for row in rows if row["status"] != "complete"]
    if failures or state.get("confirmation_incomplete") or state.get("final_evaluation_incomplete"):
        lines += ["", "## Incomplete or failed work", ""]
        for row in failures:
            lines.append(
                f"- {row['run_id']} / {row['split']} / {row['method']}: "
                f"{row['status']}; {row.get('error', '')}."
            )
        if state.get("confirmation_incomplete"):
            lines.append("- Not all confirmation seeds completed within the reserved time.")
        if state.get("final_evaluation_incomplete"):
            lines.append("- Final evaluation was interrupted by the total computation limit.")
    lines += [
        "",
        "A time-budget stop does not establish a performance plateau. Improvements apply "
        "to this fixed synthetic dataset and scheduling policy; they do not establish real-data "
        "performance or a global optimum.",
        "",
        "Full settings and changes are in search_state.json and each run's config.json. "
        "Training curves, checkpoint metadata, unbounded forecasts, allocations, and solver "
        "diagnostics are saved per run. leaderboard.csv includes every attempted evaluation.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))
    from greenload.reporting import generate_figures

    generate_figures(root, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/experiments/tuning-180d"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--budget-hours", type=float, default=4.0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--stage", default="train", choices=("train", "baseline", "test"))
    parser.add_argument("--method", default="gru")
    parser.add_argument("--worker-seconds", type=float, default=3600)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/oracle_cache"))
    args = parser.parse_args()
    if args.worker:
        try:
            worker(
                args.worker,
                args.stage,
                args.method,
                time.monotonic() + args.worker_seconds,
                args.cache_dir,
            )
        except Exception as exc:
            write_json(
                args.worker / "status.json",
                {
                    "status": "timed_out" if isinstance(exc, BudgetExceeded) else "failed",
                    "stage": args.stage,
                    "error": repr(exc),
                },
            )
            raise
        return
    cfg = Config.load(args.config) if args.config else DEFAULT
    if args.smoke_test:
        cfg = smoke_config(cfg)
    if args.budget_hours <= 0:
        parser.error("--budget-hours must be positive")
    run_search(args.run_dir, cfg, args.budget_hours * 3600)


if __name__ == "__main__":
    main()
