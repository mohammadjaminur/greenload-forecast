# Interpretation of the completed experiment

## What improved

On the untouched synthetic test month, the frozen selected GRU reduces realized MPC cost by 10.81% and mean forecast NRMSE by 73.87% compared with persistence. Across all three training seeds, its mean cost remains 10.67% below persistence. These are comparisons under one corrected protocol, not comparisons with historical costs.

The corrected setup uses six months of data, multi-hour observed histories, calendar features, a compact GRU, training-only scaling, disjoint target periods, deployment-mode early stopping, and equal terminal service. The original experiment had 457 observations, one-step inputs, larger models, overlapping targets, validation-assisted normalization, and teacher-forced validation. Those changes were made together to establish a valid benchmark; this experiment does not isolate the causal contribution of each correction.

## Forecast accuracy and scheduling differ

The controlled history comparison uses the same data, origins, seed, architecture, optimizer and scheduling policy. Only history length changes (with normal early stopping):

| History | Validation MPC cost | Mean NRMSE | Run minutes |
|---|---:|---:|---:|
| 24h | 229,016,451.64 | 0.1934 | 3.77 |
| 48h | 229,378,563.32 | 0.1872 | 6.20 |
| 168h | 229,179,042.90 | 0.1826 | 23.10 |

The full-week history gives the lowest aggregate forecasting error among the searched GRUs, but the 24-hour history gives the lowest validation scheduling cost. Forecast errors and geographic ranking diagnostics are retained per signal/location/horizon; aggregate NRMSE alone cannot explain scheduling regret. The MPC objective remains heavily weighted toward carbon, with its original weights fixed.

## The small tuning gain is not robust

For seed 42, the 24-hour selection improves test cost by 0.133% over the initial 48-hour setup. This is below the predeclared 0.5% meaningful-improvement threshold. Across seeds 42/43/44:

| Setup | Test mean cost | Sample standard deviation |
|---|---:|---:|
| Initial 48h | 214,127,477.85 | 31,456.76 |
| Selected 24h | 214,142,229.70 | 281,336.47 |

The selected setup has 0.0069% higher mean test cost and considerably greater seed variation. Its forecast NRMSE is also higher than the initial setup's in the seed-42 test comparison. The evidence supports correcting the experimental setup and beating persistence; it does not support claiming a reliable further gain from the 24-hour tuning choice. The configuration was frozen before confirmation/test results and was not changed afterward.

Ridge is the strongest observed baseline: its seed-42 test objective is 213,724,872.68, versus 213,817,370.51 for the selected GRU. Keeping GRU as the learned model was a user-selected constraint; the report does not conceal the stronger simple baseline.

## Stopping and reproducibility

Eight distinct GRU configurations were evaluated. A second coordinate sweep reused completed results and found no new improving neighbor, so the search stopped at `untried_neighbor_exhaustion`. This is a local coordinate-search stop, not exhaustion of all joint parameter combinations and not a three-round plateau or a time-budget stop. Four additional configurations supplied seeds 43/44 for the initial and selected setups; all 48 signal models and 26 evaluations completed without failure.

All 26 schedules passed the full-period audit. Validation methods served 4057.3022438 workload units and test methods served 3910.0404781 units; maximum absolute terminal backlog was below 1e-11. All results share the same canonical dataset checksum. Checkpoints match their minimum deployment-mode validation loss, and the selection file predates the first test metric.

The source snapshot and dependency record support reproduction. Configurations, histories, raw forecast arrays, checkpoints, solver diagnostics and leaderboard are retained locally. Scientific plots are in `figures/`. Three training seeds describe initialization sensitivity on one fixed synthetic dataset; they do not establish uncertainty across real datasets.
