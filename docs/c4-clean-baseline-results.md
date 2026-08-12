# C4 Clean Baseline v2 Results

C4 completed on 2026-08-13 and freezes the Clean Baseline v2 recipe. It uses
the accepted 30-epoch RealMLP from C2 and Clean Table from C3 without changing
either component.

For each outer, `raw`, standard-deviation, and RMS scaling plus Table weights
from `0.00` to `1.00` in `0.01` steps were searched only on the registered
inner tune predictions. Learned scales were then applied unchanged to outer
predictions. The Table weight is the directional weight after the selected
component scaling.

## Nested results

| Outer | Inner choice | Table weight | RealMLP | Fold adaptive | Delta |
|---|---|---:|---:|---:|---:|
| PSEUDO | RMS | 0.46 | 0.136379917 | 0.142648649 | +0.006268731 |
| H2 | std | 0.36 | 0.138287880 | 0.141761926 | +0.003474045 |
| T3 | RMS | 0.37 | 0.139785533 | 0.143515387 | +0.003729854 |
| T4 | RMS | 0.37 | 0.150293835 | 0.156923672 | +0.006629838 |

The fold-adaptive arithmetic mean is `0.146212408`, a descriptive improvement
of `+0.005025617` over RealMLP. These outers are correlated temporal stress
tests, not independent samples.

The predeclared default gate passes: all four leakage-safe nested outer
predictions are non-degrading and their mean improvement is greater than
`+0.0005`.

## Frozen production rule

The production method is the mode of the four inner-selected methods and the
production Table weight is the median of the four inner-selected weights:

```text
RMS(RealMLP) * 0.63 + RMS(Table) * 0.37
```

This aggregation happens only after the four nested tests pass. It must not be
scored back on PSEUDO as if it were an untouched outer: T3/T4 inner months
m41-50 overlap the PSEUDO m33-70 outer. Such a cross-fold score would violate
the explicit ban on calibrating among overlapping stress tests.

Numeric production scales are intentionally not frozen from outer or test
distributions. They will be fitted once from the canonical rolling OOF m51-70
segment. The recipe, component versions, method, weight, and scale source are
now frozen; later alpha experiments may not retune them.

## Integrity

- RealMLP/Table IDs, months, and targets matched exactly for every inner and
  outer artifact; all predictions were finite.
- Calibration outputs record the component scales, input prediction hashes,
  selected method/weight, and the frozen production calibration.
- Each outer `predictions.npz` stores its leakage-safe fold-specific nested
  prediction. No cross-fold aggregate is presented as an outer prediction.
- Tests rewrite outer predictions without changing inner method/weight
  selection and reject any component ID misalignment.
- No competition submission, prediction array, model weight, token, or private
  data is included in the public repository.

Clean Baseline v2 is frozen. The next stage is canonical rolling OOF production
followed by M01-A residual alpha.
