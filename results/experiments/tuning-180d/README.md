# Corrected 180-day experiment

This directory contains the completed experiment: eight searched GRU setups, four additional training-seed repeats, 48 signal models, and 26 evaluations in 79.3 minutes.

## Read first

- [Report](report.md): metrics and the recorded stopping reason.
- [Interpretation](analysis.md): history ablations, seed sensitivity, and limitations.
- [Summary](summary.json) and [leaderboard](leaderboard.csv): machine-readable results.
- [Frozen selection](selection.json) and [initial configuration](base_config.json).
- [Artifact audit](artifact_audit.json) and [training-only data profile](training_data_profile.json).
- [Exact source snapshot](source_snapshot/) and [runtime provenance](provenance.json).

## Figures

- [MPC objective cost](figures/test_mpc_cost.png)
- [Forecast RMSE by horizon](figures/test_forecast_rmse.png)
- [Training and deployment-validation curves](figures/training_validation_curves.png)
- [First test-window forecast example](figures/example_test_forecast.png)

## Training runs

Each run directory includes configuration, data identity, training history, best-epoch summary, validation metrics, and normalized-path console logs. Test metrics exist only for the initial and selected setups across seeds 42/43/44.

| Run | Seed | History | Hidden | Layers | Learning rate | Normalization | Decoder |
|---|---:|---:|---:|---:|---:|---|---|
| [gru-5c108e2b06f1f75b](gru-5c108e2b06f1f75b/config.json) | 42 | 48h | 64 | 1 | 0.001 | global | direct |
| [gru-7086bbf538dcca2e](gru-7086bbf538dcca2e/config.json) | 42 | 24h | 64 | 1 | 0.001 | global | direct |
| [gru-5f878bda6559d67c](gru-5f878bda6559d67c/config.json) | 42 | 168h | 64 | 1 | 0.001 | global | direct |
| [gru-c16c71192624bc3e](gru-c16c71192624bc3e/config.json) | 42 | 24h | 32 | 1 | 0.001 | global | direct |
| [gru-22d045ba523e9abb](gru-22d045ba523e9abb/config.json) | 42 | 24h | 64 | 2 | 0.001 | global | direct |
| [gru-f4d23721a898f7a2](gru-f4d23721a898f7a2/config.json) | 42 | 24h | 64 | 1 | 0.0003 | global | direct |
| [gru-759fb4c19eaaa161](gru-759fb4c19eaaa161/config.json) | 42 | 24h | 64 | 1 | 0.001 | per_location | direct |
| [gru-3ad4606f6cb814c3](gru-3ad4606f6cb814c3/config.json) | 42 | 24h | 64 | 1 | 0.001 | global | autoregressive |
| [gru-b569ac2733a38e28](gru-b569ac2733a38e28/config.json) | 43 | 24h | 64 | 1 | 0.001 | global | direct |
| [gru-db175bd2c7d3150e](gru-db175bd2c7d3150e/config.json) | 43 | 48h | 64 | 1 | 0.001 | global | direct |
| [gru-ef12e5cfdbf3db49](gru-ef12e5cfdbf3db49/config.json) | 44 | 24h | 64 | 1 | 0.001 | global | direct |
| [gru-b80a37b3fdaa96a3](gru-b80a37b3fdaa96a3/config.json) | 44 | 48h | 64 | 1 | 0.001 | global | direct |

[Simple baseline metrics and regularization choices](baseline/) are stored separately.

## Reproduce

From the repository root, use a new directory for an independent run:

```bash
python -m greenload.experiments --budget-hours 4 --run-dir results/experiments/reproduction
```

The default `tuning-180d` directory is already complete and returns cached metrics. Model weights and numerical arrays are local-only; the committed JSON, CSV, histories, and figures remain readable without retraining. See [the configurations guide](../../../configs/README.md) to train one setup.

Runtime provenance records the original revision plus working-tree changes at execution time. The source hash and committed source snapshot identify the actual code used. Console command paths were normalized to be relative to the repository for publication; numerical results were unchanged.
