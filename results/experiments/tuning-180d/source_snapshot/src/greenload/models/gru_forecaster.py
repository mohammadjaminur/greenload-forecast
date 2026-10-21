"""History-encoding GRU with direct or scheduled-sampling autoregressive decoding."""

from __future__ import annotations

import torch
from torch import nn

from greenload.config import Config


def scheduled_sampling_prob(epoch: int, cfg: Config, total_epochs: int) -> float:
    if epoch < cfg.ss_warmup_epochs:
        return 0.0
    span = max(total_epochs - cfg.ss_warmup_epochs - 1, 1)
    return min(
        cfg.ss_max_sampling_prob, cfg.ss_max_sampling_prob * (epoch - cfg.ss_warmup_epochs) / span
    )


class GRUForecaster(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        horizon: int,
        dropout: float,
        input_size: int = 1,
        output_size: int = 1,
        decoder: str = "autoregressive",
    ):
        super().__init__()
        if output_size != 1:
            raise ValueError("Each model predicts one signal per location")
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, 1)
        self.direct = nn.Sequential(
            nn.Linear(hidden_size + 4 * horizon, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon),
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        self.input_size = input_size
        self.decoder = decoder

    def init_hidden(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        sampling_prob: float = 0.0,
        future: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, hidden = self.gru(x, hidden)
        if future is None:
            future = x.new_zeros((len(x), self.horizon, 4))
        if self.decoder == "direct":
            features = torch.cat([hidden[-1], future.flatten(1)], dim=1)
            return self.direct(features).unsqueeze(1), hidden

        previous = x[:, -1:, :1]
        outputs = []
        for step in range(self.horizon):
            step_input = previous
            if self.input_size > 1:
                step_input = torch.cat([previous, future[:, step : step + 1]], dim=-1)
            out, hidden = self.gru(step_input, hidden)
            pred = self.linear(out)
            outputs.append(pred)
            if target is None:
                previous = pred  # gradients through deployment-like rollouts
            elif sampling_prob <= 0:
                previous = target[:, :, step : step + 1]
            else:
                use_prediction = torch.rand(len(x), 1, 1, device=x.device) < sampling_prob
                previous = torch.where(use_prediction, pred.detach(), target[:, :, step : step + 1])
        return torch.cat(outputs, dim=2), hidden


def make_model(cfg: Config) -> GRUForecaster:
    return GRUForecaster(
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        horizon=cfg.window_size,
        dropout=cfg.dropout,
        input_size=cfg.input_size,
        decoder=cfg.decoder,
    ).to(cfg.device)
