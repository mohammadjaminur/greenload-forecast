"""Zero-order-hold (persistence) baseline forecaster.

Predicts that every future step equals the last observed value. It has no
learnable parameters and needs no training. It exists so the GRU forecaster's
value can be judged against a trivial baseline rather than only against a
perfect-foresight oracle -- see `evaluate.py` and the README's results table.
"""

from __future__ import annotations

import torch


class NaiveForecaster:
    """Drop-in replacement for GRUForecaster's inference-time call signature."""

    def __init__(self, horizon: int):
        self.horizon = horizon

    def __call__(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        sampling_prob: float = 0.0,
        future: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        return x[:, -1:, :1].expand(-1, -1, self.horizon).clone(), None
