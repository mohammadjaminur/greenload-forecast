import dataclasses

import numpy as np
import torch

from greenload.config import DEFAULT, smoke_config
from greenload.data.dataset import build_all_splits, inputs_at
from greenload.data.generate_synthetic_data import generate_raw_series


def test_split_shapes():
    cfg = smoke_config()
    splits = build_all_splits(generate_raw_series(cfg), cfg)
    for signal, num_loc in (("workload", 1), ("price", 10), ("water", 10), ("carbon", 10)):
        for name in ("train", "val", "test"):
            split = getattr(splits[signal], name)
            start, end = cfg.bounds(name)
            assert split.num_windows == end - start - cfg.window_size + 1
            assert split.source.shape == (
                split.num_windows * num_loc,
                cfg.history_hours,
                cfg.input_size,
            )
            assert split.target.shape == (split.num_windows * num_loc, 1, cfg.window_size)


def test_target_periods_are_disjoint_and_history_can_cross_boundary():
    cfg = smoke_config()
    raw = generate_raw_series(cfg)
    splits = build_all_splits(raw, cfg)
    split = splits["price"]
    target_times = {}
    for name in ("train", "val", "test"):
        data = getattr(split, name)
        target_times[name] = set(
            (data.origins[:, None] + np.arange(1, cfg.window_size + 1)).flatten()
        )
        start, end = cfg.bounds(name)
        assert min(target_times[name]) == start
        assert max(target_times[name]) == end - 1
        denormalized = split.scaler.denormalize(
            data.source[:, :, 0]
            .numpy()
            .reshape(data.num_windows, cfg.num_loc, -1)
            .transpose(0, 2, 1)
        )
        np.testing.assert_allclose(
            denormalized[0], raw["price"][start - cfg.history_hours : start], atol=1e-4
        )
    assert target_times["train"].isdisjoint(target_times["val"])
    assert target_times["val"].isdisjoint(target_times["test"])


def test_scaler_excludes_validation_and_test():
    cfg = smoke_config()
    raw = generate_raw_series(cfg)
    first = build_all_splits(raw, cfg)["price"].scaler
    raw["price"][cfg.train_hours :] += 100000
    second = build_all_splits(raw, cfg)["price"].scaler
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.std, second.std)
    assert np.isclose(first.mean, raw["price"][: cfg.train_hours].mean())


def test_per_location_scaler_roundtrip():
    cfg = dataclasses.replace(smoke_config(), normalization="per_location")
    raw = generate_raw_series(cfg)["carbon"]
    from greenload.data.dataset import build_signal_splits

    scaler = build_signal_splits(raw, cfg).scaler
    np.testing.assert_allclose(scaler.denormalize(scaler.normalize(raw)), raw)
    np.testing.assert_allclose(scaler.mean, raw[: cfg.train_hours].mean(axis=0))


def test_inputs_never_read_future_observations():
    cfg = smoke_config()
    raw = generate_raw_series(cfg)["price"]
    origin = cfg.train_hours
    first = inputs_at(raw, np.array([origin]), cfg)
    raw[origin + 1 :] = -1e9
    second = inputs_at(raw, np.array([origin]), cfg)
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a, b)


def test_history_candidates_share_identical_training_origins():
    raw = generate_raw_series(DEFAULT)
    from greenload.data.dataset import build_signal_splits

    a = build_signal_splits(raw["workload"], dataclasses.replace(DEFAULT, history_hours=24))
    b = build_signal_splits(raw["workload"], dataclasses.replace(DEFAULT, history_hours=168))
    np.testing.assert_array_equal(a.train.origins, b.train.origins)
