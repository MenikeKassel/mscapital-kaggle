# M02-T temporal Geometry results

Status: **complete, gate failed; M02 family closed**.

This is a self-owned local Protocol-v2 result. LB142 was not used for feature
construction, residual targets, alpha selection, or gating. No competition
submission was created.

Feature artifact:

- rows: `1,257,637`
- features: `85` (`21` M02-base + `64` fixed temporal features)
- feature hash: `88f3c56f5b572e16bde0f92aec8cd578a116762a85cd5344f24b36591868e59d`
- base feature hash: `3d2cf0efeca9e6d8bb890ab584b32f7a4b911ae193d07e5ebce828b3076a20b1`
- temporal artifact values hash: `642b76a6add4e9d43b3c0be1a59fafbff2555657216f6872983b1a9198ab175a`
- build revision: `263aafb` (`Mask uncovered temporal quotes and verify M02 attribution`)
- invalid raw quote rows skipped/forward-filled: `935,238`
- as-of grid: one-second points from 60 seconds before prediction through 0

| Outer | Baseline | M02-base | M02-T final | M02-T delta | Delta vs M02-base | Alpha | Best iteration | M02-base corr. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PSEUDO | 0.142550340 | 0.143025766 | 0.142918425 | +0.000368085 | -0.000107340 | 0.07 | 146 | 0.999312 |
| H2 | 0.141861992 | 0.142389314 | 0.142239354 | +0.000377361 | -0.000149961 | 0.05 | 30 | 0.999724 |
| T3 | 0.143549308 | 0.144100186 | 0.144196422 | +0.000647115 | +0.000096236 | 0.07 | 85 | 0.999270 |
| T4 | 0.157053101 | 0.157679166 | 0.157878499 | +0.000825397 | +0.000199333 | 0.07 | 85 | 0.999395 |

Mean M02-T delta vs frozen baseline: `+0.000554490`.
Mean M02-T delta vs M02-base: `+0.000009567`.
All four predictions are finite, all four outer folds are positive, and all
drift ratios are within the configured range. The gate fails because PSEUDO
improvement is below `+0.0015`.

| Outer | Prediction std | Std ratio | Absolute-p99 ratio | Baseline correlation |
|---|---:|---:|---:|---:|
| PSEUDO | 0.965325 | 0.961691 | 0.920950 | 0.998940 |
| H2 | 1.034129 | 1.031934 | 1.018885 | 0.999811 |
| T3 | 1.013106 | 1.006109 | 1.016049 | 0.998778 |
| T4 | 1.135006 | 1.127167 | 1.176753 | 0.998975 |

All ratios remain inside the frozen `[0.67, 1.50]` and `[0.50, 2.00]` bounds.

Each fold manifest also records hashes for the aligned M02-base
`sample_id/month/target/pred` arrays; the summary command reloads those arrays
and replays the M02-base score, M02-T delta, and attribution correlation.

Decision: M02-base and M02-T are recorded as positive-but-insufficient
representation results. The M02 family is closed; no further Geometry window
or lead-lag variants will be run. The next method family is M03 depth-2 Path
Signature, under a separate implementation decision.
