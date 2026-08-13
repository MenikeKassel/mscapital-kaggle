# M02-base Geometry results

Run status: **complete, not promoted**. This is the M02-base local protocol
experiment; temporal dependency/lead-lag subexperiments were not run.
no Kaggle competition submission was created and LB142 was not used for
selection, training, or gating.

Feature artifact:

- rows: `1,257,637`
- features: `21` (20 geometry values plus `lob_quote_missing`)
- feature hash: `3d2cf0efeca9e6d8bb890ab584b32f7a4b911ae193d07e5ebce828b3076a20b1`
- missing quote rows retained: `0` (all 219 apparent misses in the first run
  were corrected as cross-Arrow-batch continuation rows)

| Outer | Rows | Baseline | M02 final | Delta | Alpha | Best iteration | Beta | Prediction corr. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PSEUDO | 672,948 | 0.142550340 | 0.143045011 | +0.000494671 | 0.10 | 167 | 0.000401903 | 0.997517 |
| H2 | 177,542 | 0.141861992 | 0.142178732 | +0.000316740 | 0.05 | 20 | 0.000380418 | 0.999850 |
| T3 | 177,542 | 0.143549308 | 0.143965793 | +0.000416486 | 0.05 | 36 | 0.000370296 | 0.999465 |
| T4 | 175,704 | 0.157053101 | 0.157709848 | +0.000656746 | 0.05 | 36 | 0.000370296 | 0.999558 |

Mean delta: `+0.000471161`; all four outer folds are positive. Every
prediction is finite and every drift ratio is within the configured range.
The candidate gate nevertheless fails because PSEUDO improvement is below
the required `+0.0015` threshold. Following the frozen route, M02 geometry is
recorded as a negative/insufficient result and M02 extensions plus M03 remain
deferred; the next research decision is documented separately rather than
automatically launching another experiment.
