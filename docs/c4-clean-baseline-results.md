# C4 Clean Baseline v2 Results

C4 completed and froze Clean Baseline v2 on 2026-08-13. It uses
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

Numeric production scales were fitted only from the canonical rolling OOF
m51-70 segment. T3 supplies the already-completed m51-60 block. A new strictly
historical `R61_70` split uses inner train 0-50, inner tune 51-60, refit 0-60,
and predicts 61-70. Both components were trained on Kaggle P100 kernels.

| Canonical block | Rows | RealMLP | Table | Frozen blend |
|---|---:|---:|---:|---:|
| m51-60 | 177,542 | 0.139785568 | 0.135453873 | 0.143549307 |
| m61-70 | 175,704 | 0.147950590 | 0.150577521 | 0.154614249 |
| concatenated m51-70 | 353,246 | 0.143846440 | 0.143282516 | **0.149173320** |

The frozen RMS scales are:

```text
scale_realmlp = 0.014812425837302948
scale_table   = 0.00035057231611417754

prediction = 0.63 * realmlp / scale_realmlp
           + 0.37 * table   / scale_table
```

The canonical result hash is
`65b9af31e082c2f10a9314af2e8e101067b29ec176c4e6b57d930a2593eac17f`.
Component versions, method, weight, scales, component order, and scale source
are now frozen. Later alpha experiments may not retune them.

## Integrity

- RealMLP/Table IDs, months, and targets matched exactly for every inner and
  outer artifact; all predictions were finite.
- Calibration outputs record the fold-specific scales, input prediction
  hashes, and selected method/weight.
- Each outer `predictions.npz` stores its leakage-safe fold-specific nested
  prediction. No cross-fold aggregate is presented as an outer prediction.
- Tests rewrite outer predictions without changing inner method/weight
  selection and reject any component ID misalignment.
- The canonical scale builder rejects missing months, duplicate IDs, target
  misalignment, non-finite predictions, and non-positive scales. The production
  application function also fixes the component order and validates shape.
- Formal `R61_70` manifests record refit 0-60, valid 61-70, 175,704 unique
  samples, finite predictions, and P100 environments. RealMLP was generated at
  `d295c45`; Table at `7db7341`.
- No competition submission, prediction array, model weight, token, or private
  data is included in the public repository.

Clean Baseline v2 is frozen. The next stage is the remaining canonical rolling
OOF blocks m21-50 for the residual dataset, followed by M01-A.
