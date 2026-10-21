# Experiment protocol and artifact guide

The executable defaults are in `greenload.config.Config`. The CLI saves the complete
configuration; evaluation reads that configuration and checkpoint scalers.

## Timeline and information available

All timestamps are synthetic integer hours. Training observations occupy `[0, 2880)`,
validation `[2880, 3600)`, and test `[3600, 4320)`. Training targets start at hour 168
so every candidate and seasonal baseline can use the same origins.

At origin `t`, history includes hours `t-L+1` through `t`, and targets are hours
`t+1` through `t+24`. Entire training/early-stopping target windows must lie inside
one period. For example, the last training origin is 2855, with targets 2856–2879;
the first validation origin is 2879, with targets 2880–2903. Historical observations
can overlap across boundaries; target periods cannot.

Forecast reporting additionally includes partial windows near the endpoint, masking
all targets outside the period. The scheduler acts on every actual hour of the
period: it observes the current signals and receives forecasts for subsequent hours.
Calendar sine/cosine features use absolute hour and require no future measurements.

Ridge regularization uses validation forecast loss. GRU early stopping uses inference
without targets; configuration search uses validation MPC cost. The frozen selection
is written before any final-test metric is computed. Training-seed confirmation does
not change the selected configuration. Test performance never triggers new tuning.

## Capacity reserve and equal service

Let fleet capacity be C, the known per-hour workload bound be U, and backlog before
the current arrival be B. With current arrival A and h future hours left in the
shrinking lookahead, require the current allocation total x to satisfy:

    x >= max(0, B + A - h * (C - U))
    x <= B + A

This maintains backlog after the action at most `h * (C-U)`. Initially backlog is
zero. Since actual arrivals are bounded by U, the required current service never
exceeds C, including when the endpoint approaches. At the endpoint h is zero, so
all remaining work is served. Future workload forecasts are clipped into `[0,U]`.
The rule uses the known bound, not future actual arrivals.

The offline oracle uses exactly the same cumulative-arrival, per-location capacity,
and backlog-reserve constraints, with actual future signals. Its objective is a
lower bound on the feasible realized MPC objective. The reserve is a conservative
benchmark policy; this implementation does not claim per-job maximum delay limits.

Solver failures or excessive numerical residual corrections fail the candidate.
Small numerical action projections are logged. An incomplete workload or an allocation
violating capacities, nonnegativity, or causality cannot be selected.

## What each artifact contains

- `search_state.json`: every attempted evaluation, configuration changes from the
  incumbent, costs/errors, status/failure, elapsed time, round history and stopping rule.
- `selection.json`: immutable validation-selected setup, selection criterion and source identity.
- `leaderboard.csv`: validation/test costs, oracle gaps, errors and key hyperparameters.
- `report.md` and `figures/`: measured comparisons and training/forecast plots.
- `config.json`: data, model, optimizer, sampling, inference and scheduler settings.
- `dataset.json`: generator version, independent data seed, split boundaries,
  normalization fitting interval and canonical array checksum.
- `provenance.json`: revision, source hash, working-tree diff, Python/platform and core dependencies.
- `history_<signal>.json`: per-epoch weighted training loss, deployment-mode validation
  loss, sampling probability, learning rate and cumulative training seconds.
- `training.json`: best epoch, best validation loss, epoch count and stopping reason per signal.
- `checkpoints/<signal>.pt`: best model weights, config, scaler and dataset identity.
- `<split>_<method>_metrics.json`: raw forecast RMSE/MAE/NRMSE, horizon and location
  breakdowns, ranking errors, objective components, oracle comparison and solver diagnostics.
- `<split>_<method>_predictions.npz`: unbounded forecasts, origins and realized allocations.
- `console.log` and `status.json`: command execution, progress, failure or timeout information.

NRMSE divides RMSE by the standard deviation of actual values in that evaluation
period; the aggregate gives each signal equal weight. Per-location NRMSE is also
reported. Curves of normalized training/validation loss use each setup's own scaler;
compare forecast errors in raw units when normalizations differ.

Checkpoints and numerical arrays remain local and are ignored by Git. All data can
be regenerated from logged settings and verified by checksum. Original results and
source snapshots are in `results/historical/pre_tuning/`.

## Search, resumption and limits

Each full coordinate round considers histories, sizes, layer/dropout pairs, learning
rates, normalizations and decoder type. Completed settings are cached. The incumbent
can improve by less than 0.5%, but only a round gain of at least 0.5% resets the
meaningful-improvement patience. Three full rounds without such a gain constitute
a validation plateau; exhaustion of untried neighbors is recorded separately.

The four-hour default reserves three hours for search and the final hour for seed
confirmation, final testing and reporting. Individual workers have hard timeouts;
training and scheduling also check deadlines internally. Reporting has a small time
reserve. A budget stop is never labeled as a plateau.

Resume by repeating the original command and run directory. Completed signal training
and evaluations are reused. An interrupted signal restarts deterministically; its
old history is archived and previous attempt details are retained. A source or base
configuration mismatch requires a new experiment directory. After an abrupt process
kill, unflushed time since the last saved state is charged conservatively, potentially
including downtime. This avoids exceeding the original authorized budget on resume.
