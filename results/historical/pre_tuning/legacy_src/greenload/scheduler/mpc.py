"""Convex-optimization workload scheduler: receding-horizon MPC + offline oracle.

Ported and cleaned from the original research codebase's `utils/solve.py`.
Only the receding-horizon MPC solver and the offline (perfect-foresight)
oracle baseline are kept -- the original file also contained a structurally
different online Lyapunov drift-plus-penalty scheduler (~35% of that file)
that doesn't take predictions as input at all. That's a different control
paradigm and is intentionally out of scope for this predict-then-schedule
showcase; see the README's "what's not included" note.

Objective (both solvers): minimize
    price_cost + l1 * max_over_locations(water_cost) + l2 * max_over_locations(carbon_cost)
subject to per-location hourly capacity and (for MPC) a cumulative
"can't allocate more than has arrived yet" workload constraint.

Unlike the original, `num_loc` is always derived from the input arrays'
shape rather than hardcoded to 10.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np


def _cost_terms(y: cp.Variable, price: np.ndarray, water: np.ndarray, carbon: np.ndarray):
    price_cost = cp.sum(cp.multiply(y, price))
    water_cost = cp.norm(cp.sum(cp.multiply(y, water), axis=1), p="inf")
    carbon_cost = cp.norm(cp.sum(cp.multiply(y, carbon), axis=1), p="inf")
    return price_cost, water_cost, carbon_cost


def offline_oracle(
    price: np.ndarray,
    water: np.ndarray,
    carbon: np.ndarray,
    workload: np.ndarray,
    l1: float,
    l2: float,
    max_cap: float = 1.0,
    verbose: bool = False,
) -> tuple[float, np.ndarray]:
    """Perfect-foresight upper-bound baseline: solve once with the true future.

    Args:
        price, water, carbon: (num_loc, num_hours) true signals.
        workload: (1, num_hours) true workload trace.
        l1, l2: water / carbon objective weights.
        max_cap: max workload fraction routable to one location per hour.

    Returns: (optimal_cost, allocation) where allocation is (num_loc, num_hours).
    """
    num_loc, num_hours = price.shape
    y = cp.Variable((num_loc, num_hours), nonneg=True)
    price_cost, water_cost, carbon_cost = _cost_terms(y, price, water, carbon)
    total_cost = price_cost + l1 * water_cost + l2 * carbon_cost

    constraints = [y <= max_cap]
    for i in range(num_hours):
        constraints += [cp.sum(y[:, : i + 1]) <= cp.sum(workload[:, : i + 1])]
    constraints += [cp.sum(y) == cp.sum(workload)]

    problem = cp.Problem(cp.Minimize(total_cost), constraints)
    problem.solve(verbose=verbose)
    return problem.value, y.value


def mpc_solver(
    price_window: np.ndarray,
    water_window: np.ndarray,
    carbon_window: np.ndarray,
    workload_window: np.ndarray,
    historical_cost: tuple[float, np.ndarray, np.ndarray],
    l1: float,
    l2: float,
    max_cap: float = 1.0,
    verbose: bool = False,
) -> tuple[float, np.ndarray]:
    """One receding-horizon MPC re-solve, over a lookahead window.

    Args:
        price_window, water_window, carbon_window: (num_loc, horizon+1) --
            the true current value in column 0, forecast for the rest.
        workload_window: (1, horizon+1), same convention.
        historical_cost: (historical_price_cost, historical_water_cost_per_loc,
            historical_carbon_cost_per_loc) accumulated from all previous
            re-solves, so the running total (not just this window) is what's
            actually being optimized against the water/carbon fairness cap.
        l1, l2: water / carbon objective weights.

    Returns: (optimal_cost, allocation) where allocation is (num_loc, horizon+1);
        in the receding-horizon loop only column 0 (this hour's decision) is applied.
    """
    hist_price, hist_water, hist_carbon = historical_cost
    num_loc, horizon = price_window.shape
    y = cp.Variable((num_loc, horizon), nonneg=True)

    price_cost = cp.sum(cp.multiply(y, price_window)) + hist_price
    water_cost = cp.norm(cp.sum(cp.multiply(y, water_window), axis=1) + hist_water, p="inf")
    carbon_cost = cp.norm(cp.sum(cp.multiply(y, carbon_window), axis=1) + hist_carbon, p="inf")
    total_cost = price_cost + l1 * water_cost + l2 * carbon_cost

    constraints = [y <= max_cap]
    for i in range(horizon):
        constraints += [cp.sum(y[:, : i + 1]) <= cp.sum(workload_window[:, : i + 1])]
    constraints += [cp.sum(y) == cp.sum(workload_window)]

    problem = cp.Problem(cp.Minimize(total_cost), constraints)
    problem.solve(verbose=verbose)
    return problem.value, y.value


def distribute_remaining_workload(window: np.ndarray, remaining: float) -> tuple[np.ndarray, float]:
    """Carry any workload that couldn't be scheduled last hour into this window,
    spilling into later hours (capped at 10 units/hour) if it doesn't fit in hour 0.
    """
    window = window.copy()
    i = 0
    while remaining != 0:
        if i >= window.shape[1]:
            break
        window[0, i] += remaining
        if window[0, i] > 10:
            remaining = window[0, i] - 10
            window[0, i] = 10
            i += 1
        else:
            remaining = 0
    return window, remaining


def evaluate_allocation(
    allocation: np.ndarray,
    price: np.ndarray,
    water: np.ndarray,
    carbon: np.ndarray,
    l1: float,
    l2: float,
) -> dict[str, float]:
    """Compute price / water / carbon / total cost for a realized (num_loc, num_hours)
    allocation against the true signals, mirroring the MPC objective.
    """
    price_per_loc = np.multiply(price, allocation).sum(axis=1)
    water_per_loc = np.multiply(water, allocation).sum(axis=1)
    carbon_per_loc = np.multiply(carbon, allocation).sum(axis=1)

    price_cost = float(np.sum(price_per_loc))
    water_cost = float(l1 * np.linalg.norm(water_per_loc, ord=np.inf))
    carbon_cost = float(l2 * np.linalg.norm(carbon_per_loc, ord=np.inf))

    return {
        "price_cost": price_cost,
        "water_cost": water_cost,
        "carbon_cost": carbon_cost,
        "total_cost": price_cost + water_cost + carbon_cost,
    }


def run_receding_horizon(
    predicted_price: np.ndarray,
    predicted_water: np.ndarray,
    predicted_carbon: np.ndarray,
    predicted_workload: np.ndarray,
    true_price: np.ndarray,
    true_water: np.ndarray,
    true_carbon: np.ndarray,
    true_workload: np.ndarray,
    l1: float,
    l2: float,
    max_cap: float = 1.0,
    verbose: bool = False,
) -> np.ndarray:
    """Run the full receding-horizon MPC loop over a test trace.

    At each hour `i`, combines the true current value with a `horizon`-step
    forecast (`predicted_*[i]`), re-solves the MPC problem, applies only the
    first hour's decision, and accumulates realized cost so the next re-solve
    optimizes against the running total.

    Within its own lookahead window, the solver is free to plan to serve this
    hour's workload later (in one of the forecast columns) rather than right
    now -- that's a legitimate MPC planning choice. But only column 0 is ever
    realized; the rest of the plan is thrown away and re-solved next hour. So
    any workload the solver *planned* to defer must be explicitly carried
    forward as backlog into the next hour's window, or it silently vanishes
    (never actually served, but also never counted as a cost) -- which makes
    a forecaster look artificially cheap simply for causing the solver to
    defer more. Backlog is tracked the same way the original codebase's
    `mpc_allocation` did: true workload arrived so far, minus workload
    actually realized so far.

    Args:
        predicted_*: (num_hours, num_loc, horizon) forecasts made at each hour.
        true_*: (num_loc, num_hours) [or (1, num_hours) for workload] realized signals.

    Returns: realized allocation, (num_loc, num_hours).
    """
    num_hours = true_price.shape[1]
    num_loc = true_price.shape[0]

    hist_price, hist_water, hist_carbon = 0.0, np.zeros(num_loc), np.zeros(num_loc)
    action_sum = 0.0
    cap_carryover = 0.0
    realized = []

    for i in range(num_hours):
        price_window = np.concatenate([true_price[:, [i]], predicted_price[i]], axis=1)
        water_window = np.concatenate([true_water[:, [i]], predicted_water[i]], axis=1)
        carbon_window = np.concatenate([true_carbon[:, [i]], predicted_carbon[i]], axis=1)
        workload_window = np.concatenate([true_workload[:, [i]], predicted_workload[i]], axis=1)

        backlog = float(true_workload[:, :i].sum()) - action_sum + cap_carryover
        workload_window, cap_carryover = distribute_remaining_workload(workload_window, backlog)

        _, y = mpc_solver(
            price_window,
            water_window,
            carbon_window,
            workload_window,
            (hist_price, hist_water, hist_carbon),
            l1=l1,
            l2=l2,
            max_cap=max_cap,
            verbose=verbose,
        )

        action = y[:, 0]
        hist_price += float(np.sum(action * true_price[:, i]))
        hist_water += action * true_water[:, i]
        hist_carbon += action * true_carbon[:, i]
        action_sum += float(np.sum(action))
        realized.append(action)

    return np.stack(realized, axis=1)
