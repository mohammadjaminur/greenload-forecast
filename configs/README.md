# Experiment configurations

`initial.json` is the corrected 48-hour starting setup.
`selected.json` is frozen from validation MPC cost before final-test evaluation.
It is not selected by final-test results or by choosing the best confirmation seed.

Train it with:

```bash
python -m greenload.train --config configs/selected.json --run-dir results/selected
python -m greenload.evaluate --run-dir results/selected --split val
```

The model seed is independent of `data_seed`. The complete search, seed comparisons,
and final-test evidence are in `results/experiments/tuning-180d/report.md`.
