"""Sliding-window dataset construction for the seq2seq forecasters.

Three bugs/gaps in the original data pipeline are fixed here:

1. **Train/val leakage.** The original code built all overlapping 24h windows
   from the training block, shuffled them, and only then sliced the first 250
   as "train" and the next 38 as "val". Since adjacent windows overlap by up
   to 23/24 hours, many "val" windows were near-duplicates of "train" windows,
   giving the one model that tracked validation loss an optimistic signal.
   Fixed by splitting hours chronologically *before* windowing.
2. **Test-set truncation.** The original evaluation notebook used only 19 of
   the ~121 available test windows with no stated reason. Fixed by using the
   full test pool.
3. **No input/target normalization.** The four signals have wildly different
   natural scales (workload ~0-8, water WUE ~0.3-3.6, carbon intensity
   ~50-840). Training all four with one shared learning rate on raw values
   makes the large-scale signals (carbon especially) converge far slower --
   in practice, training collapses to predicting a near-constant value. The
   original notebooks worked around this by hand-tuning a different learning
   rate per signal (1e-6/1e-4/1e-5/1e-5) without addressing the underlying
   scale mismatch (and, tellingly, also had several never-finished "_Scaled"
   notebook variants that attempted a fix and were abandoned). Standardizing
   every signal to zero mean / unit variance -- using statistics computed
   from the training split only, so the model never sees test-set statistics
   -- removes the need for per-signal tuning entirely: one learning rate
   works for all four.

Shape convention: every signal (workload has `num_loc=1`, the others have
`num_loc=10`) is exposed to the model with location folded into the batch
dimension -- `(num_windows * num_loc, 1, 1)` for the source step and
`(num_windows * num_loc, 1, window_size)` for the target -- so the GRU always
sees a single real time step per call, for every signal. The original code
fed the 10-location axis into the GRU as a fake length-10 *sequence* for the
price/water/carbon models (but not for workload), which meant "locations"
were being recurrently processed through a shared hidden state as if they
were consecutive time steps -- a bug in its own right, and one that would
have made scheduled sampling need three different implementations. Folding
location into batch fixes both problems at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from greenload.config import DEFAULT, Config


@dataclass
class SignalScaler:
    """Z-score scaler; mean/std are computed from the training split only."""

    mean: float
    std: float

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


@dataclass
class WindowedSplit:
    """Model-ready (normalized) tensors for one split (train/val/test) of one signal."""

    source: torch.Tensor  # (N * num_loc, 1, 1)
    target: torch.Tensor  # (N * num_loc, 1, window_size)
    num_windows: int
    num_loc: int


@dataclass
class SignalSplits:
    train: WindowedSplit
    val: WindowedSplit
    test: WindowedSplit
    scaler: SignalScaler


def _build_windows(signal: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """signal: (T, num_loc). Returns source (N, num_loc), target (N, window_size, num_loc)."""
    length = signal.shape[0]
    num_windows = length - window_size
    if num_windows <= 0:
        raise ValueError(f"signal too short ({length}) for window_size={window_size}")
    source = signal[:num_windows]
    target = np.stack([signal[i + 1 : i + window_size + 1] for i in range(num_windows)], axis=0)
    return source, target


def _to_windowed_split(source: np.ndarray, target: np.ndarray) -> WindowedSplit:
    num_windows, num_loc = source.shape
    window_size = target.shape[1]
    src_t = torch.tensor(source, dtype=torch.float32).reshape(num_windows * num_loc, 1, 1)
    tgt_t = (
        torch.tensor(target, dtype=torch.float32)
        .permute(0, 2, 1)  # (N, window_size, num_loc) -> (N, num_loc, window_size)
        .reshape(num_windows * num_loc, 1, window_size)
    )
    return WindowedSplit(source=src_t, target=tgt_t, num_windows=num_windows, num_loc=num_loc)


def build_signal_splits(raw_signal: np.ndarray, cfg: Config = DEFAULT) -> SignalSplits:
    """Chronologically split one raw (T, num_loc) signal into train/val/test WindowedSplits,
    normalized using statistics computed from the training block only.
    """
    train_block = raw_signal[: cfg.train_block_hours]
    test_block = raw_signal[cfg.train_block_hours :]

    scaler = SignalScaler(mean=float(train_block.mean()), std=float(train_block.std()) or 1.0)
    train_block = scaler.normalize(train_block)
    test_block = scaler.normalize(test_block)

    tv_source, tv_target = _build_windows(train_block, cfg.window_size)
    n_train, n_val = cfg.num_train_windows, cfg.num_val_windows
    expected = n_train + n_val
    if tv_source.shape[0] != expected:
        raise ValueError(
            f"train block produced {tv_source.shape[0]} windows, expected {expected}. "
            "Adjust `train_block_hours`/`num_train_windows`/`num_val_windows` in config, "
            "or the requested `total_hours`."
        )

    test_source, test_target = _build_windows(test_block, cfg.window_size)

    return SignalSplits(
        train=_to_windowed_split(tv_source[:n_train], tv_target[:n_train]),
        val=_to_windowed_split(tv_source[n_train:], tv_target[n_train:]),
        test=_to_windowed_split(test_source, test_target),
        scaler=scaler,
    )


def build_all_splits(
    raw_series: dict[str, np.ndarray], cfg: Config = DEFAULT
) -> dict[str, SignalSplits]:
    """raw_series: output of generate_synthetic_data.generate_raw_series.

    Returns {signal_name: SignalSplits}.
    """
    return {name: build_signal_splits(series, cfg) for name, series in raw_series.items()}


def iter_batches(
    split: WindowedSplit, batch_size: int, num_loc: int, shuffle: bool, generator: torch.Generator
):
    """Yield (source, target) mini-batches of `batch_size` *windows* (each window
    contributes `num_loc` rows), grouping all of a window's locations together
    so a mini-batch never splits a window's per-location rows across batches.
    """
    order = (
        torch.randperm(split.num_windows, generator=generator)
        if shuffle
        else torch.arange(split.num_windows)
    )
    for start in range(0, split.num_windows, batch_size):
        window_idx = order[start : start + batch_size]
        row_idx = (window_idx.unsqueeze(1) * num_loc + torch.arange(num_loc)).reshape(-1)
        yield split.source[row_idx], split.target[row_idx]
