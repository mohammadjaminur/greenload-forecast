"""GRU-based seq2seq forecaster, with a scheduled-sampling fix for exposure bias.

The original notebook's forecaster was trained with pure teacher forcing (the
true future value is fed at every step) but evaluated with closed-loop
autoregressive rollout (the model's own prediction is fed back in) -- the
original author's own inline comment on that closed-loop code path read
"the performance is very poor". That is the textbook symptom of *exposure
bias*: the model never sees its own mistakes during training, so small errors
compound once it has to. Scheduled sampling (Bengio et al., 2015) fixes this
by occasionally feeding the model its own prediction during training too, on
a schedule that ramps up over epochs.

`horizon` is passed explicitly to the constructor (the original code read a
module-global `window_size` from inside the model's forward method).
"""

from __future__ import annotations

import torch
from torch import nn

from greenload.config import Config


def scheduled_sampling_prob(epoch: int, cfg: Config, total_epochs: int) -> float:
    """Monotonically non-decreasing sampling probability for a given epoch.

    - Epochs < cfg.ss_warmup_epochs: 0.0 (pure teacher forcing). Early in
      training the model's own predictions are close to noise; feeding that
      back in from epoch 1 would slow convergence for no benefit.
    - After warmup: linear ramp up to `cfg.ss_max_sampling_prob`, capped
      there rather than reaching 1.0 (fully closed-loop training on a small
      model over a short run tends to destabilize training).
    """
    if epoch < cfg.ss_warmup_epochs:
        return 0.0
    remaining_span = max(total_epochs - cfg.ss_warmup_epochs, 1)
    progress = (epoch - cfg.ss_warmup_epochs) / remaining_span
    return min(cfg.ss_max_sampling_prob, cfg.ss_max_sampling_prob * progress)


class GRUForecaster(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        horizon: int,
        dropout: float,
        input_size: int = 1,
        output_size: int = 1,
    ):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

    def init_hidden(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        sampling_prob: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 1, 1) current observed value.
            hidden: optional initial hidden state.
            target: optional (B, 1, horizon) true future values. If given,
                used as the teacher-forcing / scheduled-sampling source. If
                None, the rollout is fully closed-loop autoregressive -- this
                is the real inference/deployment path, and what `evaluate.py`
                always uses.
            sampling_prob: probability, per example per step, of feeding the
                model's own last prediction instead of the true target value.
                Only meaningful when `target` is given; ignored otherwise.
                Pass 0.0 (the default) for plain teacher forcing, e.g. when
                computing validation loss.

        Returns:
            predictions (B, 1, horizon), final hidden state.
        """
        step_input = x
        outputs: list[torch.Tensor] = []
        for t in range(self.horizon):
            out, hidden = self.gru(step_input, hidden)
            pred = self.linear(out)  # (B, 1, 1)
            outputs.append(pred)
            if t == self.horizon - 1:
                break

            if target is None:
                step_input = pred.detach()
            else:
                true_next = target[:, :, t].unsqueeze(-1)  # (B, 1, 1)
                if sampling_prob <= 0:
                    step_input = true_next
                else:
                    use_own_pred = (
                        torch.rand(pred.shape[0], device=pred.device) < sampling_prob
                    ).view(-1, 1, 1)
                    step_input = torch.where(use_own_pred, pred.detach(), true_next)

        return torch.cat(outputs, dim=2), hidden
