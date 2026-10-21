"""Procedural synthetic data generator.

Produces hourly workload / electricity-price / water-usage-effectiveness /
carbon-intensity traces for `num_loc` synthetic datacenter locations. This
project intentionally does NOT ship or depend on any real trace data (grid
carbon/water/price recordings, or real workload logs) -- everything here is
generated from a fixed seed so results are fully reproducible without any
external or private data dependency.

Design goals (see README "Engineering fixes" section for why each matters):
  * Locations are genuinely distinct (drawn from location-indexed RNG
    substreams), not near-duplicates -- otherwise the MPC scheduler's
    location-choice problem becomes trivial.
  * Each channel mixes daily + weekly seasonality with AR(1)-correlated noise
    (not i.i.d. per-hour noise), so there is real short-horizon structure for
    the GRU forecaster to learn, but enough unpredictability over a 24-step
    horizon that exposure bias is visible and worth fixing.
  * Non-negativity is enforced by flooring at generation time, not by the
    original codebase's `price[price < 0] *= -1` sign-flip hack.
  * Workload amplitude is tuned so rolling 24h utilization stays in a
    feasible, non-degenerate band relative to total fleet capacity
    (see `tests/test_synthetic_data.py` for the enforced invariant).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from greenload.config import DEFAULT, Config

# (baseline_low, baseline_high, daily_amp_frac, weekly_amp_frac, ar_coef, noise_std_frac)
_CHANNEL_PARAMS = {
    "price": (20.0, 80.0, 0.25, 0.10, 0.85, 0.06),
    "water": (0.2, 3.0, 0.20, 0.08, 0.85, 0.06),
    "carbon": (50.0, 700.0, 0.20, 0.08, 0.85, 0.06),
}


def _ar1_noise(rng: np.random.Generator, length: int, std: float, ar_coef: float) -> np.ndarray:
    """Autocorrelated noise: noise[t] = ar_coef * noise[t-1] + eps[t]."""
    eps_std = std * np.sqrt(1 - ar_coef**2)  # keeps stationary std ~= `std`
    eps = rng.normal(0.0, eps_std, size=length)
    noise = np.empty(length)
    noise[0] = eps[0]
    for t in range(1, length):
        noise[t] = ar_coef * noise[t - 1] + eps[t]
    return noise


def _seasonal_signal(
    rng: np.random.Generator,
    hours: int,
    baseline: float,
    daily_amp: float,
    weekly_amp: float,
    ar_coef: float,
    noise_std: float,
    phase: float,
) -> np.ndarray:
    t = np.arange(hours)
    daily = daily_amp * np.sin(2 * np.pi * t / 24 + phase)
    weekly = weekly_amp * np.sin(2 * np.pi * t / (24 * 7) + phase / 2)
    noise = _ar1_noise(rng, hours, noise_std, ar_coef)
    signal = baseline * (1 + daily + weekly) + baseline * noise
    return np.clip(signal, a_min=baseline * 0.05, a_max=None)


def generate_location_channel(
    rng: np.random.Generator, hours: int, num_loc: int, channel: str
) -> np.ndarray:
    """Generate an (hours, num_loc) array for price/water/carbon."""
    lo, hi, daily_amp, weekly_amp, ar_coef, noise_std = _CHANNEL_PARAMS[channel]
    out = np.empty((hours, num_loc))
    for loc in range(num_loc):
        loc_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        baseline = loc_rng.uniform(lo, hi)
        phase = loc_rng.uniform(0, 2 * np.pi)
        out[:, loc] = _seasonal_signal(
            loc_rng, hours, baseline, daily_amp, weekly_amp, ar_coef, noise_std, phase
        )
    return out


def generate_workload(
    rng: np.random.Generator,
    hours: int,
    num_loc: int,
    max_cap: float,
    target_utilization: float = 0.55,
) -> np.ndarray:
    """Generate a single global (hours, 1) workload trace, business-hours-shaped.

    Amplitude is scaled so the rolling 24h sum sits around `target_utilization`
    of total fleet capacity (`num_loc * max_cap * 24`) -- feasible for the MPC
    scheduler, never degenerately zero, never so large the receding-horizon
    cumulative-workload constraint becomes infeasible.
    """
    fleet_capacity_per_hour = num_loc * max_cap
    target_mean = target_utilization * fleet_capacity_per_hour
    raw = _seasonal_signal(
        rng,
        hours,
        baseline=target_mean,
        daily_amp=0.35,
        weekly_amp=0.15,
        ar_coef=0.85,
        noise_std=0.08,
        phase=0.0,
    )
    floor = 0.05 * fleet_capacity_per_hour
    return np.clip(raw, a_min=floor, a_max=fleet_capacity_per_hour * 0.95).reshape(hours, 1)


def generate_raw_series(cfg: Config = DEFAULT, hours: int | None = None, seed: int | None = None):
    """Return dict of raw (unwindowed) synthetic hourly series.

    Keys: 'workload' (T,1), 'price'/'water'/'carbon' (T, num_loc).
    """
    hours = hours or cfg.total_hours
    seed = cfg.seed if seed is None else seed
    root_rng = np.random.default_rng(seed)

    series = {
        "workload": generate_workload(root_rng, hours, cfg.num_loc, cfg.max_cap),
    }
    for channel in ("price", "water", "carbon"):
        series[channel] = generate_location_channel(root_rng, hours, cfg.num_loc, channel)
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=DEFAULT.total_hours)
    parser.add_argument("--seed", type=int, default=DEFAULT.seed)
    parser.add_argument("--out", type=Path, default=Path("data/raw_series.npz"))
    args = parser.parse_args()

    series = generate_raw_series(hours=args.hours, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **series, hours=args.hours, seed=args.seed)
    for name, arr in series.items():
        print(
            f"{name:10s} shape={arr.shape} min={arr.min():.3f} "
            f"max={arr.max():.3f} mean={arr.mean():.3f}"
        )
    print(f"Saved synthetic series to {args.out}")


if __name__ == "__main__":
    main()
