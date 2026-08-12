# C3 Clean Table v2 Results

C3 completed on 2026-08-13. All four registered outer folds used the same
frozen v5 representation and model recipe under protocol-v2. Stopping points
were selected only on each fold's inner tune months; every component and the
MLP scaler were then reinitialized and refitted on registered history before
outer prediction.

## Formal outer results

| Outer | Clean Table | Pearson | Pred std | Target std | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| PSEUDO | 0.135051054 | 0.135373333 | 0.000347514 | 0.002534166 | 992.6 |
| H2 | 0.134216095 | 0.133928759 | 0.000343033 | 0.002495371 | 1,589.4 |
| T3 | 0.135453873 | 0.135209994 | 0.000344951 | 0.002495371 | 2,240.3 |
| T4 | 0.150388711 | 0.152205417 | 0.000393530 | 0.002981453 | 2,181.0 |
| Arithmetic mean | **0.138777433** | - | - | - | - |

The four outers are correlated temporal stress tests, not four independent
samples. The arithmetic mean is descriptive only.

H2 and T3 score exactly the same m51-60 targets. Extending refit history from
m0-40 to m0-50 improved Clean Table by `+0.001237778`. This supports the
strictly historical expanding-window design and does not use the shared outer
target to tune a production choice.

## Component diagnostics

| Outer | LGBM | CatBoost | MLP (3 seeds) | Fixed blend |
|---|---:|---:|---:|---:|
| PSEUDO | 0.126948593 | 0.130996297 | 0.128172985 | **0.135051054** |
| H2 | 0.130417383 | 0.130554764 | 0.127380225 | **0.134216095** |
| T3 | 0.131031722 | 0.130202717 | 0.128005265 | **0.135453873** |
| T4 | 0.135833465 | 0.142991986 | 0.145002171 | **0.150388711** |

The historical `0.2 LGBM + 0.5 CatBoost + 0.3 MLP` blend beat every component
on all four outers. C3 therefore keeps it as the frozen Table component. No
outer score was used to alter these internal weights.

## Frozen stopping points

| Outer | LGBM | CatBoost | MLP seeds 2026 / 7 / 123 |
|---|---:|---:|---:|
| PSEUDO | 723 | 2,057 | 15 / 12 / 11 |
| H2 | 514 | 3,506 | 9 / 4 / 12 |
| T3 | 1,004 | 3,872 | 13 / 13 / 9 |
| T4 | 1,004 | 3,872 | 13 / 13 / 9 |

T3 and T4 share the same inner and refit histories. Their best steps and both
inner/refit MLP scaler hashes matched exactly, while their outer predictions
were produced for different registered month ranges.

## Historical and RealMLP context

- The preserved legacy v5 PSEUDO artifact recomputes to `0.134870712`.
- Clean Table PSEUDO changed by only `+0.000180342` and has prediction
  correlation `0.992194123` with legacy v5. The historical Table result was not
  a protocol artifact.
- Clean Table and the accepted 30-epoch RealMLP have outer prediction
  correlations of `0.8093`, `0.8479`, `0.8408`, and `0.8200` on
  PSEUDO/H2/T3/T4 respectively.
- RealMLP prediction standard deviation is roughly 34-42 times Table's. Raw
  blend weights therefore do not express comparable directional influence;
  C4 must choose among raw, standard-deviation, and RMS scaling on inner tune
  only.
- On T4, Clean Table (`0.150388711`) and 30-epoch RealMLP (`0.150293835`) are
  essentially tied despite using different representations. This makes their
  nested blend the next highest-value experiment.

## Integrity

- All formal runs used source revision
  `56e69cd25b2b2dbdb319dec928e9f81d1181b85e`.
- PSEUDO has 672,948 unique IDs covering m33-70; H2/T3 each have 177,542
  unique IDs covering m51-60; T4 has 175,704 unique IDs covering m61-70.
- All predictions are finite. PSEUDO targets match the historical v5 artifact
  exactly; H2 and T3 IDs, months, and targets also match each other exactly.
- A new single private v1 Kaggle data bundle was used. It contains only the two
  feature parquet files; labels came from the competition mount.
- Prediction arrays, component arrays, histories, logs, model weights, and
  private data remain excluded from the public repository.
- No Kaggle competition submission was created.

C3 is complete. Clean Baseline v2 remains unfrozen until C4 performs nested
cross-model calibration and applies its predeclared all-outer gate.
