# M02-base Geometry results

Run status: **complete, not promoted**. This is the M02-base local protocol
experiment; temporal dependency/lead-lag subexperiments were not run.
no Kaggle competition submission was created and LB142 was not used for
selection, training, or gating.

Feature artifact:

- rows: `1,257,637`
- features: `21` (20 geometry values plus `lob_quote_missing`)
- feature hash: `3d2cf0efeca9e6d8bb890ab584b32f7a4b911ae193d07e5ebce828b3076a20b1`
- missing/invalid latest quotes: `5,804` (retained as zero-valued geometry with
  `lob_quote_missing=1`)

| Outer | Rows | Baseline | M02 final | Delta | Alpha | Best iteration | Beta | Prediction corr. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PSEUDO | 672,948 | 0.142550340 | 0.143025766 | +0.000475426 | 0.09 | 145 | 0.000401903 | 0.997990 |
| H2 | 177,542 | 0.141861992 | 0.142389314 | +0.000527322 | 0.07 | 44 | 0.000380418 | 0.999385 |
| T3 | 177,542 | 0.143549308 | 0.144100186 | +0.000550879 | 0.07 | 74 | 0.000370296 | 0.998801 |
| T4 | 175,704 | 0.157053101 | 0.157679166 | +0.000626064 | 0.07 | 74 | 0.000370296 | 0.998997 |

Mean delta: `+0.000544923`; all four outer folds are positive. Every
prediction is finite and every drift ratio is within the configured range.
The candidate gate nevertheless fails because PSEUDO improvement is below
the required `+0.0015` threshold. Following the frozen route, M02 geometry is
recorded as a negative/insufficient result and M02 extensions plus M03 remain
deferred; the next research decision is documented separately rather than
automatically launching another experiment.
