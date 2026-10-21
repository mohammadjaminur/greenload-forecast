"""Serializable settings shared by training, inference, and experiment search."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

SIGNALS = ("workload", "price", "water", "carbon")


@dataclass(frozen=True)
class Config:
    seed: int = 42  # model initialization, minibatches, scheduled sampling
    data_seed: int = 42  # deliberately independent of the training seed
    num_loc: int = 10
    window_size: int = 24
    max_cap: float = 1.0
    total_hours: int = 4320
    train_hours: int = 2880
    val_hours: int = 720
    history_hours: int = 48
    common_history_hours: int = 168  # identical training origins across candidates
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    decoder: str = "direct"
    normalization: str = "global"
    calendar_features: bool = True
    batch_size: int = 64
    num_epochs: int = 60
    patience: int = 8
    min_delta: float = 1e-4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    ss_warmup_epochs: int = 5
    ss_max_sampling_prob: float = 0.65
    device: str = "cpu"
    torch_threads: int = 2
    l1_water: float = 60.0
    l2_carbon: float = 1500.0
    workload_upper_fraction: float = 0.95

    def __post_init__(self):
        if not (0 < self.history_hours <= self.common_history_hours < self.train_hours):
            raise ValueError("Require 0 < history <= common history < training hours")
        if min(self.val_hours, self.total_hours - self.train_hours - self.val_hours) < (
            self.window_size
        ):
            raise ValueError("Validation and test periods must each contain a full horizon")
        if self.train_hours - self.common_history_hours < self.window_size:
            raise ValueError("Training period must contain a full target horizon")
        if self.decoder not in ("direct", "autoregressive"):
            raise ValueError("Unknown decoder")
        if self.normalization not in ("global", "per_location"):
            raise ValueError("Unknown normalization")
        if not (0 < self.workload_upper_fraction <= 1):
            raise ValueError("Invalid workload upper bound")
        if min(self.num_loc, self.num_epochs, self.batch_size, self.patience) < 1:
            raise ValueError("Counts must be positive")
        if min(self.window_size, self.hidden_size, self.num_layers, self.torch_threads) < 1:
            raise ValueError("Model and horizon sizes must be positive")
        if self.max_cap <= 0 or self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("Capacity, learning rate, and gradient clipping must be positive")
        if not 0 <= self.dropout < 1 or not 0 <= self.ss_max_sampling_prob <= 1:
            raise ValueError("Invalid dropout or scheduled sampling probability")

    @property
    def input_size(self) -> int:
        return 5 if self.calendar_features else 1

    @property
    def workload_upper_bound(self) -> float:
        return self.num_loc * self.max_cap * self.workload_upper_fraction

    def bounds(self, split: str) -> tuple[int, int]:
        return {
            "train": (self.common_history_hours, self.train_hours),
            "val": (self.train_hours, self.train_hours + self.val_hours),
            "test": (self.train_hours + self.val_hours, self.total_hours),
        }[split]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> Config:
        return cls(**json.loads(Path(path).read_text()))


DEFAULT = Config()


def smoke_config(cfg: Config = DEFAULT) -> Config:
    return replace(
        cfg,
        total_hours=300,
        train_hours=220,
        val_hours=40,
        history_hours=24,
        hidden_size=8,
        num_epochs=1,
        batch_size=16,
    )
