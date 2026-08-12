# C1 Clean RealMLP-v2a Results

Completed on 2026-08-13 using four isolated Kaggle Tesla P100 runs. All
reported scores use uncentered cosine and unrounded outer-validation targets.

| Outer | Refit months | Valid months | Rows | Cosine | Pearson | Pred mean | Pred std | Target std | Best epoch | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PSEUDO | 0-32 | 33-70 | 672,948 | 0.136040611 | 0.136731280 | -0.001639910 | 0.013822925 | 0.002534166 | 10/10 | 685.7 |
| H2 | 0-40 | 51-60 | 177,542 | 0.135809991 | 0.135693703 | -0.001940488 | 0.015048571 | 0.002495371 | 10/10 | 868.3 |
| T3 | 0-50 | 51-60 | 177,542 | 0.136877769 | 0.136681996 | -0.001806254 | 0.015530599 | 0.002495371 | 10/10 | 1040.9 |
| T4 | 0-50 | 61-70 | 175,704 | 0.145643884 | 0.148139920 | -0.001619431 | 0.016830027 | 0.002981453 | 10/10 | 1033.9 |
| Arithmetic mean | - | - | - | **0.138593064** | - | - | - | - | - | - |

The four outers are correlated temporal stress tests, not four independent
samples. The arithmetic mean is descriptive and must not be interpreted as an
independent-sample confidence estimate.

## Legacy comparison

- Legacy RealMLP PSEUDO cosine: `0.138559701`.
- Clean PSEUDO cosine: `0.136040611`.
- Clean minus legacy: `-0.002519090`.
- Clean-versus-legacy PSEUDO prediction Pearson: `0.958997615`.
- Legacy targets and clean month-33-70 targets were exactly equal before the
  comparison was computed.

This is the finite-optimism outcome: removing protocol leakage lowers the
PSEUDO estimate, but the RealMLP signal remains material. C1 therefore does
not invalidate RealMLP and supports proceeding to targeted C2 diagnostics.

## Regime evidence

H2 and T3 share the same month-51-60 outer validation period. Adding months
41-50 to the refit history increases P100 cosine by `0.001067778`. T3 and T4
share the same inner split, best progress, refit history, preprocessing state,
and model configuration; moving the validation period from 51-60 to 61-70
increases cosine by `0.008766115`. The late regime is substantially more
predictable for this representation.

All three distinct inner searches selected epoch 10 of 10 and their tune
scores were still rising at the boundary. C1 preserves the frozen 10-epoch
schedule. C2 should treat a longer-schedule ceiling check as a diagnostic
before interpreting optimizer, mask, or target-rounding ablations.

## Integrity checks

- Every outer prediction has a unique `sample_id`, the exact registered month
  range and row count, and no NaN or infinite values.
- Downloaded IDs, months, and targets match the competition label table
  exactly after stable `sample_id` sorting.
- Preprocessing, target-aware feature selection, category vocabularies,
  quantile boundaries, robust scaling, and RQ codebooks are fit only on the
  permitted inner or refit history.
- Rewriting outer-validation features and targets leaves inner/refit
  preprocessing hashes, RQ state, training inputs, and best progress
  unchanged (covered by a regression test).
- H2 was repeated on P100 with an identical score. Local RTX runs are retained
  only as cross-machine smoke evidence and are not mixed into the formal mean.
- Formal manifests record Git/config/data/feature hashes, split sizes,
  diagnostics, best step/progress, runtime, and the P100 software environment.
- No Kaggle competition submission was created by C1.

Predictions, model weights, kernel logs, credentials, and machine-specific
paths are intentionally excluded from the public repository.
