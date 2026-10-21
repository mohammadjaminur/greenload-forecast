# GreenLoad experiment report

Search stopping reason: **untried_neighbor_exhaustion**.
Total computation: 79.3 minutes; budget: 4.00 hours.

Selection uses validation MPC cost. The final test period is never used for tuning.
All reported schedules must complete the same workload, obey capacities and causality, and use the same finite-period reserve rule as the offline oracle.
Historical pre-tuning metrics use a different protocol and are not comparable.

Initial GRU: `gru-5c108e2b06f1f75b`. Selected GRU: `gru-7086bbf538dcca2e`.

Selected setup: history_hours=24, hidden_size=64, num_layers=1, dropout=0.0, learning_rate=0.001, normalization=global, decoder=direct.

## Validation

| Run / method | Seed | MPC cost | Oracle gap | Mean NRMSE |
|---|---:|---:|---:|---:|
| gru-7086bbf538dcca2e | 42 | 229016451.64 | 1.185% | 0.1934 |
| gru-759fb4c19eaaa161 | 42 | 229019052.85 | 1.186% | 0.1866 |
| gru-22d045ba523e9abb | 42 | 229115321.83 | 1.228% | 0.1895 |
| ridge | 42 | 229135398.49 | 1.237% | 0.1757 |
| gru-5f878bda6559d67c | 42 | 229179042.90 | 1.256% | 0.1826 |
| gru-c16c71192624bc3e | 42 | 229294084.96 | 1.307% | 0.1945 |
| gru-db175bd2c7d3150e | 43 | 229358748.56 | 1.336% | 0.1883 |
| gru-b80a37b3fdaa96a3 | 44 | 229364103.90 | 1.338% | 0.1887 |
| gru-5c108e2b06f1f75b | 42 | 229378563.32 | 1.345% | 0.1872 |
| gru-ef12e5cfdbf3db49 | 44 | 229488843.19 | 1.393% | 0.1944 |
| gru-f4d23721a898f7a2 | 42 | 229514269.95 | 1.405% | 0.1982 |
| gru-b569ac2733a38e28 | 43 | 229606217.04 | 1.445% | 0.1919 |
| gru-3ad4606f6cb814c3 | 42 | 229750855.39 | 1.509% | 0.2038 |
| seasonal_weekly | 42 | 230660013.98 | 1.911% | 0.2635 |
| seasonal_daily | 42 | 231777914.33 | 2.405% | 0.3041 |
| persistence | 42 | 253920796.95 | 12.188% | 0.7639 |

Selected GRU cost reduction relative to the initial corrected GRU: **0.158%**.

## Final test

| Run / method | Seed | MPC cost | Oracle gap | Mean NRMSE |
|---|---:|---:|---:|---:|
| ridge | 42 | 213724872.68 | 1.365% | 0.1817 |
| gru-7086bbf538dcca2e | 42 | 213817370.51 | 1.409% | 0.2021 |
| gru-5c108e2b06f1f75b | 42 | 214102570.63 | 1.544% | 0.1965 |
| gru-db175bd2c7d3150e | 43 | 214117034.95 | 1.551% | 0.1965 |
| gru-b80a37b3fdaa96a3 | 44 | 214162827.98 | 1.573% | 0.1965 |
| gru-ef12e5cfdbf3db49 | 44 | 214304366.71 | 1.640% | 0.2009 |
| gru-b569ac2733a38e28 | 43 | 214304951.89 | 1.640% | 0.1994 |
| seasonal_weekly | 42 | 214582577.37 | 1.772% | 0.2703 |
| seasonal_daily | 42 | 216474084.04 | 2.669% | 0.3166 |
| persistence | 42 | 239725058.86 | 13.697% | 0.7733 |

Selected GRU cost reduction relative to the initial corrected GRU: **0.133%**.

Lowest-cost simple baseline on the final test: **ridge**. Selected GRU cost relative to that baseline: **+0.043%**.

## Selected GRU forecast errors

| Signal | NRMSE | RMSE | MAE |
|---|---:|---:|---:|
| workload | 0.2992 | 0.4587 | 0.3636 |
| price | 0.1677 | 3.2134 | 2.4612 |
| water | 0.1588 | 0.1183 | 0.0896 |
| carbon | 0.1826 | 34.3596 | 25.9685 |

## Seed confirmation

- Selected, val: seeds [42, 43, 44]; mean cost 229370503.96; sample standard deviation 312184.14.
- Selected, test: seeds [42, 43, 44]; mean cost 214142229.70; sample standard deviation 281336.47.
- Initial, val: seeds [42, 43, 44]; mean cost 229367138.59; sample standard deviation 10250.03.
- Initial, test: seeds [42, 43, 44]; mean cost 214127477.85; sample standard deviation 31456.76.

## Search rounds

- Round 1: 229378563.32 → 229016451.64; 7 new setups; meaningful improvement: False.
- Round 2: 229016451.64 → 229016451.64; 0 new setups; meaningful improvement: False.

A time-budget stop does not establish a performance plateau. Improvements apply to this fixed synthetic dataset and scheduling policy; they do not establish real-data performance or a global optimum.

Full settings and changes are in search_state.json and each run's config.json. Training curves, checkpoint metadata, unbounded forecasts, allocations, and solver diagnostics are saved per run. leaderboard.csv includes every attempted evaluation.
