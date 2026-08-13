# Remaining mainline methods summary

All three required fixed-recipe residual families were run once on the four
registered outer folds. None passed the common candidate gate because each had
PSEUDO gain below `+0.0015`:

| Method | Mean delta | Positive folds | PSEUDO delta | Decision |
|---|---:|---:|---:|---|
| M03 Path Signature | +0.000203163 | 4/4 | +0.000267273 | failed; closed |
| M04 Optiver Interactions | +0.000476535 | 4/4 | +0.000392311 | failed; closed |
| M05 Market-State KNN | +0.000057317 | 2/4 | +0.000267382 | failed; closed |

M06 is `not_identifiable` and produced no prediction. Because no candidate
passed, the conditional nested alpha ensemble is recorded as:

```json
{"status":"no_candidate","methods":[],"production_weight_freeze":false}
```

No M03 conditional learners, no ensemble, no test prediction, and no Kaggle
competition submission were created. LB142 remains external post-hoc evidence
only.
