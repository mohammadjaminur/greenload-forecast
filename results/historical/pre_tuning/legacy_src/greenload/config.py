"""Central configuration for data generation, training, and evaluation.

All magic numbers that were previously copy-pasted across a dozen notebooks
live here once, so a hyperparameter change only needs to happen in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SIGNALS = ("workload", "price", "water", "carbon")


@dataclass(frozen=True)
class Config:
    seed: int = 42

    # --- problem size ---
    num_loc: int = 10
    window_size: int = 24  # forecast horizon, hours
    max_cap: float = 1.0  # max workload fraction routable to one location per hour

    # --- synthetic data volume (kept close to the original dataset's
    # proportions so results stay comparable to the numbers this project
    # was originally built against) ---
    total_hours: int = 457
    train_block_hours: int = 312  # first `train_block_hours` -> train+val pool
    # remaining (total_hours - train_block_hours) hours -> test pool
    num_train_windows: int = 250
    num_val_windows: int = 38
    # test pool uses every window it can produce (train_block_hours .. total_hours),
    # not an arbitrary truncation.

    # --- forecaster training ---
    batch_size: int = 12
    num_epochs: int = 25
    num_layers: int = 3
    dropout: float = 0.45
    device: str = "cpu"

    hidden_sizes: dict[str, int] = field(
        default_factory=lambda: {"workload": 512, "price": 512, "water": 256, "carbon": 256}
    )
    learning_rates: dict[str, float] = field(
        default_factory=lambda: {"workload": 1e-4, "price": 1e-4, "water": 1e-4, "carbon": 1e-4}
    )

    # --- scheduled sampling (fixes the exposure-bias bug: the original model
    # was trained with pure teacher forcing but evaluated with closed-loop
    # autoregressive rollout) ---
    ss_warmup_epochs: int = 5  # pure teacher forcing for the first N epochs
    ss_max_sampling_prob: float = 0.65  # sampling probability is capped here, never reaches 1.0

    # --- MPC scheduler objective weights: price + l1*max(water) + l2*max(carbon) ---
    l1_water: float = 60.0
    l2_carbon: float = 1500.0


DEFAULT = Config()
