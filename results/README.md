# Results

The current, corrected experiment is in [experiments/tuning-180d/](experiments/tuning-180d/README.md).

- [Full report](experiments/tuning-180d/report.md): validation/test metrics and seed confirmation.
- [Interpretation](experiments/tuning-180d/analysis.md): what improved and what remains unproven.
- [Leaderboard](experiments/tuning-180d/leaderboard.csv): all 26 evaluations.
- [Summary](experiments/tuning-180d/summary.json): machine-readable headline comparisons.
- [Artifact audit](experiments/tuning-180d/artifact_audit.json): data, checkpoint, and allocation checks.
- [Selected configuration](../configs/selected.json): the setup frozen using validation cost.

## Measured outcome

On the fixed synthetic test month, the selected GRU reduces MPC cost by 10.81% and
mean forecast NRMSE by 73.87% relative to persistence for seed 42. Across three model
seeds, its mean cost remains 10.67% below persistence. Every evaluated method serves
the same workload completely.

The small additional tuning gain over the corrected initial GRU is not consistent
across seeds. Ridge has the lowest observed test cost. These qualifications are part
of the result, rather than reasons to rerun selection against the final test month.

## Historical artifacts

Root-level `metrics.json`, `training_history.json`, and `figures/` belong to the
**legacy 457-hour experiment**. They are preserved for traceability and are not the
current results. Copies and original source snapshots are in
[historical/pre_tuning/](historical/pre_tuning/README.md). Different data and terminal
service rules prevent a direct cost comparison with the corrected experiment.

## Local-only numerical artifacts

Checkpoints (`.pt`), generated datasets and prediction/allocation arrays (`.npz`),
and smoke-test outputs remain local and are ignored by Git. Published artifacts
include metrics, configurations, epoch histories, console logs, plots, audits,
dependency versions, and an executable source snapshot. Run a new experiment directory
to regenerate numerical artifacts; the demo notebook can display the committed plots
without them.
