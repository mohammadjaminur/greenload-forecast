"""Capacity-safe MPC and a matching finite-period perfect-foresight oracle."""

from __future__ import annotations

from collections import Counter

import cvxpy as cp
import numpy as np

from greenload.artifacts import check_deadline

SOLVER = "CLARABEL"


class SchedulingError(RuntimeError):
    pass


def _solve(problem):
    try:
        problem.solve(solver=SOLVER, warm_start=True, tol_gap_abs=1e-7, tol_feas=1e-8)
    except cp.error.SolverError as exc:
        raise SchedulingError(f"Solver failed: {exc}") from exc
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise SchedulingError(f"Solver status: {problem.status}")
    return problem.status


class HorizonProblem:
    """Reuse the parameterized convex program across hours of the same horizon."""

    def __init__(self, num_loc, hours, l1, l2, max_cap):
        self.y = cp.Variable((num_loc, hours), nonneg=True)
        self.price = cp.Parameter((num_loc, hours), nonneg=True)
        self.water = cp.Parameter((num_loc, hours), nonneg=True)
        self.carbon = cp.Parameter((num_loc, hours), nonneg=True)
        self.arrivals = cp.Parameter(hours, nonneg=True)
        self.hist_water = cp.Parameter(num_loc, nonneg=True)
        self.hist_carbon = cp.Parameter(num_loc, nonneg=True)
        self.minimum_service = cp.Parameter(nonneg=True)
        price_cost = cp.sum(cp.multiply(self.y, self.price))
        water_cost = cp.max(cp.sum(cp.multiply(self.y, self.water), axis=1) + self.hist_water)
        carbon_cost = cp.max(cp.sum(cp.multiply(self.y, self.carbon), axis=1) + self.hist_carbon)
        objective = (price_cost + l1 * water_cost + l2 * carbon_cost) / max(l1, l2, 1)
        cumulative = cp.cumsum(cp.sum(self.y, axis=0))
        self.problem = cp.Problem(
            cp.Minimize(objective),
            [
                self.y <= max_cap,
                cumulative <= self.arrivals,
                cp.sum(self.y) == self.arrivals[-1],
                cp.sum(self.y[:, 0]) >= self.minimum_service,
            ],
        )
        self.scale = max(l1, l2, 1)

    def solve(self, price, water, carbon, workload, historical, minimum_service=0.0):
        self.price.value = price
        self.water.value = water
        self.carbon.value = carbon
        self.arrivals.value = np.cumsum(workload.reshape(-1))
        self.hist_water.value = np.maximum(historical[1], 0)
        self.hist_carbon.value = np.maximum(historical[2], 0)
        self.minimum_service.value = minimum_service
        status = _solve(self.problem)
        return self.problem.value * self.scale + historical[0], self.y.value, status


def offline_oracle(
    price,
    water,
    carbon,
    workload,
    l1,
    l2,
    max_cap=1.0,
    verbose=False,
    workload_upper_bound=None,
    forecast_horizon=None,
):
    del verbose
    num_loc, num_hours = price.shape
    y = cp.Variable((num_loc, num_hours), nonneg=True)
    cumulative = cp.cumsum(cp.sum(y, axis=0))
    arrivals = np.cumsum(workload.reshape(-1))
    constraints = [y <= max_cap, cumulative <= arrivals, cp.sum(y) == arrivals[-1]]
    if workload_upper_bound is not None:
        if forecast_horizon is None:
            raise ValueError("Oracle reserve requires a forecast horizon")
        slack = num_loc * max_cap - workload_upper_bound
        if slack < 0 or np.max(workload) > workload_upper_bound + 1e-8:
            raise ValueError("Actual workload exceeds the declared upper bound")
        reserve = np.minimum(forecast_horizon, np.arange(num_hours - 1, -1, -1)) * slack
        constraints.append(cumulative >= arrivals - reserve)
    objective = (
        cp.sum(cp.multiply(y, price))
        + l1 * cp.max(cp.sum(cp.multiply(y, water), axis=1))
        + l2 * cp.max(cp.sum(cp.multiply(y, carbon), axis=1))
    ) / max(l1, l2, 1)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    _solve(problem)
    if y.value is None:
        raise SchedulingError("Oracle returned no allocation")
    return float(problem.value * max(l1, l2, 1)), y.value


def mpc_solver(
    price_window,
    water_window,
    carbon_window,
    workload_window,
    historical_cost,
    l1,
    l2,
    max_cap=1.0,
    verbose=False,
    minimum_service=0.0,
):
    del verbose
    problem = HorizonProblem(price_window.shape[0], price_window.shape[1], l1, l2, max_cap)
    value, allocation, _ = problem.solve(
        price_window,
        water_window,
        carbon_window,
        workload_window,
        historical_cost,
        minimum_service,
    )
    return value, allocation


def distribute_remaining_workload(window, remaining, fleet_capacity=10.0):
    """Compatibility helper; remaining work is carried once, never counted twice."""
    window = window.copy()
    for index in range(window.shape[1]):
        added = min(max(0.0, fleet_capacity - window[0, index]), max(remaining, 0.0))
        window[0, index] += added
        remaining -= added
    return window, max(remaining, 0.0)


def evaluate_allocation(allocation, price, water, carbon, l1, l2):
    price_cost = float(np.sum(price * allocation))
    water_cost = float(l1 * np.max(np.sum(water * allocation, axis=1)))
    carbon_cost = float(l2 * np.max(np.sum(carbon * allocation, axis=1)))
    return {
        "price_cost": price_cost,
        "water_cost": water_cost,
        "carbon_cost": carbon_cost,
        "total_cost": price_cost + water_cost + carbon_cost,
    }


def allocation_diagnostics(allocation, workload, max_cap):
    cumulative = np.cumsum(allocation.sum(axis=0))
    arrived = np.cumsum(workload.reshape(-1))
    backlog = arrived - cumulative
    result = {
        "arrived_workload": float(arrived[-1]),
        "served_workload": float(cumulative[-1]),
        "remaining_backlog": float(backlog[-1]),
        "max_backlog": float(np.max(backlog)),
        "max_capacity_violation": float(max(0, np.max(allocation) - max_cap)),
        "max_causality_violation": float(max(0, np.max(-backlog))),
        "max_nonnegativity_violation": float(max(0, -np.min(allocation))),
    }
    result["valid"] = bool(
        abs(result["remaining_backlog"]) <= 1e-4
        and max(
            result[k]
            for k in (
                "max_capacity_violation",
                "max_causality_violation",
                "max_nonnegativity_violation",
            )
        )
        <= 1e-5
    )
    return result


def _project_numerical_action(action, lower, upper, max_cap):
    """Correct tiny solver residuals only; large violations invalidate the run."""
    clipped = np.clip(action, 0, max_cap)
    target = np.clip(clipped.sum(), lower, upper)
    if clipped.sum() > target:
        clipped *= target / clipped.sum()
    else:
        missing = target - clipped.sum()
        for index in range(len(clipped)):
            added = min(max_cap - clipped[index], missing)
            clipped[index] += added
            missing -= added
    correction = float(np.abs(clipped - action).sum())
    if correction > 1e-3:
        raise SchedulingError(f"Allocation required excessive numerical correction: {correction}")
    return clipped, correction


def run_receding_horizon(
    predicted_price,
    predicted_water,
    predicted_carbon,
    predicted_workload,
    true_price,
    true_water,
    true_carbon,
    true_workload,
    l1,
    l2,
    max_cap=1.0,
    verbose=False,
    workload_upper_bound=None,
    return_diagnostics=False,
    deadline=None,
):
    del verbose
    num_loc, num_hours = true_price.shape
    horizon = predicted_price.shape[-1]
    fleet = num_loc * max_cap
    bound = fleet if workload_upper_bound is None else workload_upper_bound
    if not (0 < bound <= fleet) or np.max(true_workload) > bound + 1e-8:
        raise ValueError("Workload violates the declared per-hour upper bound")
    slack = fleet - bound
    hist_price, hist_water, hist_carbon = 0.0, np.zeros(num_loc), np.zeros(num_loc)
    backlog, corrections, clipped_forecasts = 0.0, 0.0, 0
    realized, statuses, problems = [], Counter(), {}
    for hour in range(num_hours):
        check_deadline(deadline)
        future_hours = min(horizon, num_hours - hour - 1)
        length = future_hours + 1
        windows = []
        for truth, prediction in (
            (true_price, predicted_price),
            (true_water, predicted_water),
            (true_carbon, predicted_carbon),
            (true_workload, predicted_workload),
        ):
            future = prediction[hour, :, :future_hours]
            if not np.isfinite(future).all():
                raise SchedulingError("Nonfinite forecast")
            maximum = bound if truth is true_workload else np.inf
            bounded = np.clip(future, 0, maximum)
            clipped_forecasts += int(np.count_nonzero(future != bounded))
            windows.append(np.concatenate([truth[:, hour : hour + 1], bounded], axis=1))
        available = backlog + float(true_workload[0, hour])
        windows[3][0, 0] = available
        # Reserve enough capacity for any allowed future arrival. This invariant
        # guarantees feasibility and terminal completion without future observations.
        minimum_service = max(0.0, available - future_hours * slack)
        if minimum_service > fleet + 1e-6:
            raise SchedulingError("Terminal reserve became infeasible")
        if length not in problems:
            problems[length] = HorizonProblem(num_loc, length, l1, l2, max_cap)
        _, y, status = problems[length].solve(
            *windows,
            (hist_price, hist_water, hist_carbon),
            minimum_service,
        )
        statuses[status] += 1
        action, correction = _project_numerical_action(y[:, 0], minimum_service, available, max_cap)
        corrections += correction
        hist_price += float(np.sum(action * true_price[:, hour]))
        hist_water += action * true_water[:, hour]
        hist_carbon += action * true_carbon[:, hour]
        backlog = max(0.0, available - float(action.sum()))
        realized.append(action)
    allocation = np.stack(realized, axis=1)
    diagnostics = allocation_diagnostics(allocation, true_workload, max_cap)
    diagnostics.update(
        {
            "solver_status_counts": dict(statuses),
            "solver": SOLVER,
            "numerical_correction_mass": corrections,
            "clipped_forecast_values": clipped_forecasts,
            "reserve_policy": (
                "backlog <= remaining_lookahead_hours * (fleet_capacity - arrival_bound)"
            ),
            "workload_upper_bound": bound,
        }
    )
    if not diagnostics["valid"]:
        raise SchedulingError(f"Invalid realized schedule: {diagnostics}")
    return (allocation, diagnostics) if return_diagnostics else allocation
