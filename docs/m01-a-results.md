# M01-A Event Flow residual result

Status: **complete, gate failed; M01-B is not started**

This report is a self-owned local Protocol-v2 result. It does not use the
LB142 reference for training, residual construction, alpha selection, or the
candidate gate. The LB142 correlation field is therefore `N/A`.

## Canonical residual OOF

The canonical artifact contains 885,936 rows covering months 21-70 exactly
once. The five blocks cover ten months each:

| Block | Rows | Source train end | Baseline cosine |
|---|---:|---:|---:|
| R21_30 | 177,647 | 20 | 0.146058868 |
| R31_40 | 177,098 | 30 | 0.138139850 |
| R41_50 | 177,945 | 40 | 0.142924910 |
| R51_60 | 177,542 | 50 | 0.143515387 |
| R61_70 | 175,704 | 60 | 0.154758789 |

All sample IDs are unique, every prediction and target is finite, and
`source_train_end < prediction_month`. The T4 residual view is limited to
months 21-50.

## M01-A outer replay

Event Flow uses 24 streaming features: signed order/trade volume per second,
signed volume per event, and event counts per second over 5/15/30/60 second
windows. Quote OFI, geometry, fast-slow differences, and interactions are not
included.

| Outer | Baseline | Final | Delta | Alpha | Best iteration | Beta |
|---|---:|---:|---:|---:|---:|---:|
| PSEUDO | 0.142550340 | 0.142550340 | +0.000000000 | 0.00 | 84 | 0.000401903 |
| H2 | 0.141861992 | 0.142376506 | +0.000514514 | 0.07 | 59 | 0.000380418 |
| T3 | 0.143549308 | 0.144325835 | +0.000776527 | 0.08 | 108 | 0.000370296 |
| T4 | 0.157053101 | 0.157650872 | +0.000597771 | 0.08 | 108 | 0.000370296 |
| Mean delta |  |  | +0.000472203 |  |  |  |

All four predictions are finite. The valid-to-replay diagnostics passed the
distribution gate: standard-deviation ratios are 0.962, 1.020, 1.001, 1.126;
absolute-p99 ratios are 0.932, 1.001, 1.017, 1.186.

## Gate decision

- PSEUDO delta requirement `>= +0.0015`: **failed** (`0.000000000`).
- Positive outer folds: **3/4**.
- Worst delta requirement `>= -0.0005`: passed (`0.000000000`).
- Finite predictions and drift limits: passed.
- Final M01-A gate: **failed**.

M01-A is recorded as a valid negative result for the Event Flow-only feature
family. M01-B-F are not run. The next independent method family is M02
Market-Centered / Market-Depth Geometry; any future M01 work requires a new,
explicit experiment decision.

## Reproducibility

- M01-A code/config revision: `5e93f3f`
- Event Flow feature hash: `b77a0a2a40418e40d28a0ce16e111649887f9b459a36cf7d76a71656ce2a2455`
- Metric: uncentered cosine similarity
- Alpha grid: `0.00-0.30`, step `0.01`
- CatBoost: maximum 3000 iterations, early stopping 200, seed 2026
- No competition submission was created.
