"""Atomic experiment artifacts, deadlines, and reproducible dataset identity."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from greenload.config import SIGNALS, Config
from greenload.data.generate_synthetic_data import generate_raw_series


class BudgetExceeded(RuntimeError):
    pass


def check_deadline(deadline: float | None):
    if deadline is not None and time.monotonic() >= deadline:
        raise BudgetExceeded("Wall-clock computation budget exhausted")


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def array_checksum(raw: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in SIGNALS:
        arr = np.ascontiguousarray(raw[name], dtype="<f8")
        digest.update(name.encode())
        digest.update(str(arr.shape).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def data_settings(cfg: Config) -> dict:
    return {
        "generator": "independent-streams-v2",
        "data_seed": cfg.data_seed,
        "total_hours": cfg.total_hours,
        "num_loc": cfg.num_loc,
        "max_cap": cfg.max_cap,
        "workload_upper_fraction": cfg.workload_upper_fraction,
    }


def prepare_data(cfg: Config, run_dir: Path, datasets_dir: Path | None = None):
    datasets_dir = datasets_dir or run_dir.parent / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    settings = data_settings(cfg)
    path = datasets_dir / (fingerprint(settings) + ".npz")
    if not path.exists():
        np.savez_compressed(path, **generate_raw_series(cfg))
    with np.load(path) as stored:
        raw = {name: stored[name] for name in SIGNALS}
    metadata = {
        "settings": settings,
        "checksum": array_checksum(raw),
        "path": os.path.relpath(path.resolve(), run_dir.resolve()),
        "split_hours": {name: cfg.bounds(name) for name in ("train", "val", "test")},
        "normalization_fit_hours": [0, cfg.train_hours],
    }
    prior = run_dir / "dataset.json"
    if prior.exists() and json.loads(prior.read_text()) != json.loads(json.dumps(metadata)):
        raise ValueError("Existing run has different dataset metadata")
    write_json(prior, metadata)
    return raw, metadata


def load_data(run_dir: Path):
    metadata = json.loads((run_dir / "dataset.json").read_text())
    with np.load(run_dir / metadata["path"]) as stored:
        raw = {name: stored[name] for name in SIGNALS}
    if array_checksum(raw) != metadata["checksum"]:
        raise ValueError("Dataset checksum does not match the training run")
    return raw, metadata


def provenance() -> dict:
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
        return result.stdout.strip()

    versions = {}
    for package in ("torch", "numpy", "scipy", "cvxpy", "clarabel", "pytest", "ruff"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {
        "revision": git("rev-parse", "HEAD"),
        "working_tree": git("status", "--short"),
        "source_diff": git("diff", "--", "src", "tests", "pyproject.toml", "Makefile"),
        "source_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": versions,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
