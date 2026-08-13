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
- temporal artifact values hash: `6f0b851024ecd9a59539589f477c7c984b26cb22aaef869ab4a503e7cd16d6a9`
- build revision: `70625b1` (`Count only usable temporal quote updates`)
- invalid raw quote rows skipped/forward-filled: `935,238`
- as-of grid: one-second points from 60 seconds before prediction through 0

| Outer | Baseline | M02-base | M02-T final | M02-T delta | Delta vs M02-base | Alpha | Best iteration | M02-base corr. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PSEUDO | 0.142550340 | 0.143025766 | 0.142949075 | +0.000398735 | -0.000076690 | 0.09 | 211 | 0.999286 |
| H2 | 0.141861992 | 0.142389314 | 0.142231458 | +0.000369466 | -0.000157856 | 0.05 | 30 | 0.999726 |
| T3 | 0.143549308 | 0.144100186 | 0.144287608 | +0.000738301 | +0.000187422 | 0.08 | 103 | 0.999149 |
| T4 | 0.157053101 | 0.157679166 | 0.157967162 | +0.000914060 | +0.000287996 | 0.08 | 103 | 0.999305 |

Mean M02-T delta vs frozen baseline: `+0.000605141`.
Mean M02-T delta vs M02-base: `+0.000060218`.
All four predictions are finite, all four outer folds are positive, and all
drift ratios are within the configured range. The gate fails because PSEUDO
improvement is below `+0.0015`.

Decision: M02-base and M02-T are recorded as positive-but-insufficient
representation results. The M02 family is closed; no further Geometry window
or lead-lag variants will be run. The next method family is M03 depth-2 Path
Signature, under a separate implementation decision.
