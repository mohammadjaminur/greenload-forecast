import dataclasses
import json

import numpy as np
import pytest
import torch

from greenload.artifacts import BudgetExceeded, prepare_data, write_json
from greenload.config import SIGNALS, Config, smoke_config
from greenload.data.dataset import SignalScaler, inputs_at
from greenload.data.generate_synthetic_data import generate_raw_series
from greenload.evaluate import evaluate_run, forecast_scores, gru_predictions
from greenload.experiments import (
    better,
    meaningful_gain,
    run_search,
    stopping_reason,
)
from greenload.models.baselines import persistence
from greenload.models.gru_forecaster import make_model
from greenload.train import train_run


def test_seasonal_baseline_alignment():
    raw = np.arange(1000, dtype=float)[:, None]
    origin = np.array([200])
    for period in (24, 168):
        predicted = persistence(raw, origin, 24, period)
        np.testing.assert_array_equal(predicted[0, 0], np.arange(201 - period, 225 - period))
    np.testing.assert_array_equal(persistence(raw, origin, 24)[0, 0], [200] * 24)


@pytest.mark.parametrize("decoder", ["direct", "autoregressive"])
def test_model_uses_history_and_known_calendar_without_targets(decoder):
    cfg = dataclasses.replace(smoke_config(), decoder=decoder)
    model = make_model(cfg).eval()
    raw = generate_raw_series(cfg)["price"]
    source, future = inputs_at(raw, np.array([219, 220]), cfg)
    with torch.no_grad():
        pred, _ = model(source, future=future, target=None)
    assert pred.shape == (20, 1, cfg.window_size)
    assert torch.isfinite(pred).all()


def test_training_checkpoint_roundtrip_and_test_guard(tmp_path):
    cfg = smoke_config()
    run_dir = tmp_path / "run"
    train_run(cfg, run_dir)
    raw = generate_raw_series(cfg)
    from greenload.artifacts import array_checksum

    predictions, _ = gru_predictions(run_dir, raw, np.array([219, 220]), cfg, array_checksum(raw))
    checkpoint = torch.load(run_dir / "checkpoints/price.pt", weights_only=True, map_location="cpu")
    scaler = SignalScaler.from_dict(checkpoint["scaler"])
    source, future = inputs_at(scaler.normalize(raw["price"]), np.array([219, 220]), cfg)
    model = make_model(Config(**checkpoint["config"])).eval()
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        expected, _ = model(source, future=future)
    np.testing.assert_allclose(
        predictions["price"],
        scaler.predictions_to_raw(expected.numpy().reshape(2, 10, cfg.window_size)),
        atol=1e-5,
    )
    history = json.loads((run_dir / "history_price.json").read_text())
    assert checkpoint["best_val_loss"] == min(row["val_loss"] for row in history)
    assert checkpoint["best_epoch"] == 1
    with pytest.raises(ValueError, match="frozen selection"):
        evaluate_run(run_dir, "test")
    metric = evaluate_run(run_dir, "val")
    assert metric["scheduling"]["diagnostics"]["valid"]
    assert abs(metric["scheduling"]["diagnostics"]["remaining_backlog"]) < 1e-4


def test_seed_and_length_do_not_change_dataset_parameters():
    cfg = smoke_config()
    a = generate_raw_series(cfg)
    b = generate_raw_series(dataclasses.replace(cfg, seed=44))
    c = generate_raw_series(dataclasses.replace(cfg, total_hours=400))
    for signal in SIGNALS:
        np.testing.assert_array_equal(a[signal], b[signal])
        np.testing.assert_array_equal(a[signal], c[signal][: cfg.total_hours])


def test_dataset_checksum_detects_tampering(tmp_path):
    from greenload.artifacts import load_data

    run_dir = tmp_path / "run"
    raw, metadata = prepare_data(smoke_config(), run_dir)
    raw["price"][0, 0] += 10
    np.savez(run_dir / metadata["path"], **raw)
    with pytest.raises(ValueError, match="checksum"):
        load_data(run_dir)


def test_forecast_metrics_exclude_targets_outside_period():
    raw = np.arange(100, dtype=float)[:, None]
    origins = np.array([89, 90, 91, 92])
    predictions = persistence(raw, origins, 4)
    a = forecast_scores(predictions, raw, origins, 90, 93)
    raw[93:] = 1e9
    b = forecast_scores(predictions, raw, origins, 90, 93)
    assert a == b


def test_stop_rules_and_cost_priority():
    assert meaningful_gain(100, 99.4)
    assert not meaningful_gain(100, 99.6)
    assert stopping_reason(3, True, 0, 10) == "validation_plateau"
    assert stopping_reason(0, False, 0, 10) == "untried_neighbor_exhaustion"
    assert stopping_reason(0, True, 11, 10) == "search_time_budget"
    old = {"scheduling": {"total_cost": 100, "diagnostics": {"valid": True}}, "mean_nrmse": 1}
    new = {"scheduling": {"total_cost": 99, "diagnostics": {"valid": True}}, "mean_nrmse": 2}
    assert better(new, old)
    new["scheduling"]["diagnostics"]["valid"] = False
    assert not better(new, old)


def fake_executor(run_dir, stage, method, seconds, cache_dir):
    del seconds, cache_dir
    cfg = Config.load(run_dir / "config.json")
    split = "test" if stage == "test" else "val"
    if split == "test":
        assert (run_dir.parent / "selection.json").exists()
    # Smaller GRUs win so at least one accepted coordinate change is exercised.
    cost = 1000.0 + cfg.hidden_size + cfg.history_hours / 100 + cfg.seed / 1000
    metric = {
        "method": method,
        "split": split,
        "mean_nrmse": 0.5,
        "scheduling": {
            "total_cost": cost,
            "optimality_gap_pct": 1.0,
            "diagnostics": {"valid": True},
        },
    }
    write_json(run_dir / f"{split}_{method}_metrics.json", metric)


def test_search_resumption_and_frozen_test_selection(tmp_path):
    cfg = smoke_config()
    state = run_search(tmp_path, cfg, 1000, executor=fake_executor)
    assert state["phase"] == "complete"
    assert state["incumbent"]
    assert state["rounds"]
    assert (tmp_path / "report.md").exists()

    def forbidden(*args):
        raise AssertionError("Completed attempts must not be rerun")

    resumed = run_search(tmp_path, cfg, 1000, executor=forbidden)
    assert resumed == state


def test_interrupted_worker_is_logged_and_not_promoted(tmp_path):
    def timeout(*args):
        raise BudgetExceeded("forced timeout")

    state = run_search(tmp_path, smoke_config(), 1, executor=timeout)
    assert state["initial"] is None
    assert state["stopping_reason"] == "search_time_budget"
    assert all(row["status"] == "timed_out" for row in state["attempts"].values())
    assert not (tmp_path / "selection.json").exists()


def test_partial_search_resumes_without_repeating_baselines(tmp_path):
    completed = []

    def interrupt_after_baselines(run_dir, stage, method, seconds, cache_dir):
        if method == "gru":
            raise KeyboardInterrupt()
        completed.append(method)
        fake_executor(run_dir, stage, method, seconds, cache_dir)

    with pytest.raises(KeyboardInterrupt):
        run_search(tmp_path, smoke_config(), 1000, executor=interrupt_after_baselines)
    assert len(completed) == 4

    def resume(run_dir, stage, method, seconds, cache_dir):
        if stage == "baseline":
            raise AssertionError("Validation baseline was already completed")
        fake_executor(run_dir, stage, method, seconds, cache_dir)

    state = run_search(tmp_path, smoke_config(), 1000, executor=resume)
    assert state["phase"] == "complete"
    assert state["incumbent"]


def test_best_checkpoint_is_middle_epoch_not_last(tmp_path, monkeypatch):
    import greenload.train as training
    from greenload.data.dataset import build_all_splits

    cfg = dataclasses.replace(smoke_config(), num_epochs=3, patience=1)
    splits = build_all_splits(generate_raw_series(cfg), cfg)
    calls = 0
    middle_states = {}

    def controlled_validation(model, source, future, settings):
        nonlocal calls
        signal = SIGNALS[calls // 3]
        epoch = calls % 3
        if epoch == 1:
            middle_states[signal] = {
                name: value.clone() for name, value in model.state_dict().items()
            }
        calls += 1
        return splits[signal].val.target + (2.0, 1.0, 3.0)[epoch]

    monkeypatch.setattr(training, "inference", controlled_validation)
    train_run(cfg, tmp_path / "run")
    for signal in SIGNALS:
        saved = torch.load(tmp_path / "run/checkpoints" / f"{signal}.pt", weights_only=True)
        assert saved["best_epoch"] == 2
        for name, value in middle_states[signal].items():
            torch.testing.assert_close(saved["model"][name], value)


def test_ridge_selection_and_validation_predictions_ignore_test_values():
    from greenload.models.baselines import ridge_predict

    cfg = smoke_config()
    raw = generate_raw_series(cfg)["workload"]
    origins = np.array([219, 250])
    first, first_details = ridge_predict(raw, origins, cfg)
    raw[cfg.train_hours + cfg.val_hours :] = 1e12
    second, second_details = ridge_predict(raw, origins, cfg)
    np.testing.assert_array_equal(first, second)
    assert first_details == second_details
