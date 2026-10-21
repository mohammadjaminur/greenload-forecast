"""Train and checkpoint four forecasters using deployment-mode validation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from greenload.artifacts import (
    BudgetExceeded,
    check_deadline,
    prepare_data,
    provenance,
    write_json,
)
from greenload.config import DEFAULT, SIGNALS, Config, smoke_config
from greenload.data.dataset import build_signal_splits, iter_batches
from greenload.models.gru_forecaster import make_model, scheduled_sampling_prob


def inference(model, source, future, cfg):
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(source), cfg.batch_size * 10):
            pred, _ = model(
                source[start : start + cfg.batch_size * 10].to(cfg.device),
                future=future[start : start + cfg.batch_size * 10].to(cfg.device),
                target=None,
            )
            outputs.append(pred.cpu())
    return torch.cat(outputs)


def train_run(
    cfg: Config,
    run_dir: Path,
    deadline: float | None = None,
    datasets_dir: Path | None = None,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if config_path.exists() and Config.load(config_path) != cfg:
        raise ValueError("Refusing to overwrite a run with different settings")
    write_json(config_path, cfg.to_dict())
    identity = provenance()
    provenance_path = run_dir / "provenance.json"
    if provenance_path.exists():
        if json.loads(provenance_path.read_text())["source_sha256"] != identity["source_sha256"]:
            raise ValueError("Source changed since this training run; use a new run directory")
    else:
        write_json(provenance_path, identity)
    raw, metadata = prepare_data(cfg, run_dir, datasets_dir)
    torch.set_num_threads(cfg.torch_threads)
    summary_path = run_dir / "training.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    started = time.monotonic()
    write_json(run_dir / "status.json", {"status": "training"})
    try:
        for index, signal in enumerate(SIGNALS):
            checkpoint_path = run_dir / "checkpoints" / f"{signal}.pt"
            if signal in summary and summary[signal]["complete"] and checkpoint_path.exists():
                continue
            check_deadline(deadline)
            torch.manual_seed(cfg.seed + index)
            np.random.seed(cfg.seed + index)
            torch.use_deterministic_algorithms(True)
            history_path = run_dir / f"history_{signal}.json"
            if history_path.exists():
                archive = run_dir / f"history_{signal}_interrupted_{time.time_ns()}.json"
                history_path.rename(archive)
            splits = build_signal_splits(raw[signal], cfg)
            model = make_model(cfg)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
            )
            generator = torch.Generator().manual_seed(cfg.seed + index)
            criterion = nn.MSELoss()
            history, best_loss, patience_best, bad_epochs = [], float("inf"), float("inf"), 0
            best_epoch = None
            signal_start = time.monotonic()
            for epoch in range(cfg.num_epochs):
                check_deadline(deadline)
                model.train()
                sampling = (
                    scheduled_sampling_prob(epoch, cfg, cfg.num_epochs)
                    if cfg.decoder == "autoregressive"
                    else 0.0
                )
                loss_sum, count = 0.0, 0
                for source, target, future in iter_batches(
                    splits.train, cfg.batch_size, splits.train.num_loc, True, generator
                ):
                    check_deadline(deadline)
                    optimizer.zero_grad(set_to_none=True)
                    pred, _ = model(
                        source.to(cfg.device),
                        target=target.to(cfg.device),
                        sampling_prob=sampling,
                        future=future.to(cfg.device),
                    )
                    loss = criterion(pred, target.to(cfg.device))
                    if not torch.isfinite(loss):
                        raise ValueError(f"Nonfinite training loss: {signal}")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                    optimizer.step()
                    loss_sum += loss.item() * len(source)
                    count += len(source)
                check_deadline(deadline)
                val_pred = inference(model, splits.val.source, splits.val.future, cfg)
                val_loss = criterion(val_pred, splits.val.target).item()
                if not np.isfinite(val_loss):
                    raise ValueError(f"Nonfinite validation loss: {signal}")
                history.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": loss_sum / count,
                        "val_loss": val_loss,
                        "sampling_prob": sampling,
                        "learning_rate": cfg.learning_rate,
                        "elapsed_seconds": time.monotonic() - signal_start,
                        "validation": "deployment_inference_without_targets",
                    }
                )
                write_json(run_dir / f"history_{signal}.json", history)
                if val_loss < best_loss:
                    best_loss, best_epoch = val_loss, epoch + 1
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = checkpoint_path.with_suffix(".tmp")
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "config": cfg.to_dict(),
                            "scaler": splits.scaler.to_dict(),
                            "signal": signal,
                            "best_epoch": best_epoch,
                            "best_val_loss": best_loss,
                            "dataset_checksum": metadata["checksum"],
                        },
                        temporary,
                    )
                    temporary.replace(checkpoint_path)
                if val_loss < patience_best - cfg.min_delta:
                    patience_best, bad_epochs = val_loss, 0
                else:
                    bad_epochs += 1
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(
                        f"{run_dir.name} {signal} epoch={epoch + 1} "
                        f"train={loss_sum / count:.5f} val={val_loss:.5f}",
                        flush=True,
                    )
                if bad_epochs >= cfg.patience:
                    break
            # Checkpoint stores the best epoch, never the last epoch by accident.
            model.load_state_dict(
                torch.load(checkpoint_path, map_location=cfg.device, weights_only=True)["model"]
            )
            summary[signal] = {
                "complete": True,
                "best_epoch": best_epoch,
                "best_val_loss": best_loss,
                "epochs": len(history),
                "elapsed_seconds": time.monotonic() - signal_start,
                "stopping_reason": (
                    "early_stopping" if bad_epochs >= cfg.patience else "max_epochs"
                ),
            }
            write_json(summary_path, summary)
        write_json(
            run_dir / "status.json",
            {
                "status": "trained",
                "elapsed_seconds_this_invocation": time.monotonic() - started,
            },
        )
    except Exception as exc:
        write_json(
            run_dir / "status.json",
            {
                "status": "timed_out" if isinstance(exc, BudgetExceeded) else "failed",
                "error": repr(exc),
                "elapsed_seconds_this_invocation": time.monotonic() - started,
            },
        )
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", "--out-dir", type=Path, default=Path("results/manual"))
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    cfg = Config.load(args.config) if args.config else DEFAULT
    cfg = smoke_config(cfg) if args.smoke_test else cfg
    train_run(cfg, args.run_dir)


if __name__ == "__main__":
    main()
