"""Train the 4 GRU forecasters (workload / price / water / carbon).

Trains two variants of every signal so the README can report an honest
before/after comparison:

  * "baseline" -- pure teacher forcing throughout, matching the original
    notebook's training procedure (the one whose closed-loop rollout the
    original author's own comment called "very poor").
  * "fixed"    -- scheduled sampling (see `models/gru_forecaster.py`).

Both variants are evaluated identically later, in `evaluate.py`, via fully
closed-loop autoregressive rollout -- the real inference/deployment path.

Usage:
    python -m greenload.train
    python -m greenload.train --smoke-test   # 1 epoch, tiny data, small model (for CI)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from greenload.config import DEFAULT, SIGNALS, Config
from greenload.data.dataset import SignalSplits, build_all_splits, iter_batches
from greenload.data.generate_synthetic_data import generate_raw_series
from greenload.models.gru_forecaster import GRUForecaster, scheduled_sampling_prob


def smoke_config(cfg: Config) -> Config:
    """Tiny, fast configuration used only for the CI smoke test."""
    return dataclasses.replace(
        cfg,
        total_hours=100,
        train_block_hours=60,
        num_train_windows=30,
        num_val_windows=6,
        num_epochs=1,
        batch_size=6,
        hidden_sizes=dict.fromkeys(SIGNALS, 8),
    )


def train_one_model(
    signal: str,
    splits: SignalSplits,
    cfg: Config,
    use_scheduled_sampling: bool,
    device: str,
) -> tuple[GRUForecaster, list[dict]]:
    model = GRUForecaster(
        hidden_size=cfg.hidden_sizes[signal],
        num_layers=cfg.num_layers,
        horizon=cfg.window_size,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rates[signal])
    criterion = nn.MSELoss()
    generator = torch.Generator().manual_seed(cfg.seed)

    num_loc = splits.train.num_loc
    history = []

    for epoch in range(cfg.num_epochs):
        model.train()
        sampling_prob = (
            scheduled_sampling_prob(epoch, cfg, cfg.num_epochs) if use_scheduled_sampling else 0.0
        )

        batch_losses = []
        for source, target in iter_batches(
            splits.train, cfg.batch_size, num_loc, shuffle=True, generator=generator
        ):
            source, target = source.to(device), target.to(device)
            optimizer.zero_grad()
            pred, _ = model(source, target=target, sampling_prob=sampling_prob)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_source = splits.val.source.to(device)
            val_target = splits.val.target.to(device)
            val_pred, _ = model(val_source, target=val_target, sampling_prob=0.0)
            val_loss = criterion(val_pred, val_target).item()

        train_loss = float(np.mean(batch_losses))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "sampling_prob": sampling_prob,
            }
        )
        print(
            f"[{signal:8s}] epoch {epoch + 1}/{cfg.num_epochs}  "
            f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
            f"sampling_prob={sampling_prob:.2f}"
        )

    return model, history


def plot_loss_curves(all_history: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, signal in zip(axes.flat, SIGNALS, strict=True):
        for mode, style in (("baseline", "--"), ("fixed", "-")):
            hist = all_history[mode][signal]
            epochs = [h["epoch"] + 1 for h in hist]
            ax.plot(epochs, [h["val_loss"] for h in hist], style, label=f"{mode} val loss")
        ax.set_title(signal)
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Validation loss: baseline (pure teacher forcing) vs. fixed (scheduled sampling)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-out", type=Path, default=Path("data/raw_series.npz"))
    parser.add_argument("--smoke-test", action="store_true", help="tiny/fast run for CI")
    args = parser.parse_args()

    cfg = smoke_config(DEFAULT) if args.smoke_test else DEFAULT
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    raw_series = generate_raw_series(cfg)
    args.data_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.data_out, **raw_series, hours=cfg.total_hours, seed=cfg.seed)

    all_splits = build_all_splits(raw_series, cfg)

    checkpoints_dir = args.out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    all_history: dict[str, dict[str, list[dict]]] = {"baseline": {}, "fixed": {}}
    start = time.time()
    for mode, use_ss in (("baseline", False), ("fixed", True)):
        mode_dir = checkpoints_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        for signal in SIGNALS:
            model, history = train_one_model(signal, all_splits[signal], cfg, use_ss, cfg.device)
            all_history[mode][signal] = history
            torch.save(model.state_dict(), mode_dir / f"{signal}.pt")
    elapsed = time.time() - start
    print(f"Training complete in {elapsed:.1f}s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "training_history.json", "w") as f:
        json.dump(all_history, f, indent=2)

    if not args.smoke_test:
        figures_dir = args.out_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_loss_curves(all_history, figures_dir / "loss_curves.png")

    history_path = args.out_dir / "training_history.json"
    print(f"Saved checkpoints to {checkpoints_dir}, history to {history_path}")


if __name__ == "__main__":
    main()
