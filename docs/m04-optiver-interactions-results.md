# M04 Optiver Interaction Family results

Status: **complete, gate failed; M04 is closed for this execution queue**.

This is a self-owned Protocol-v2 residual experiment. LB142 was not used for
feature construction, residuals, selection, or gating; no competition
submission or test prediction was created.

Feature artifact: 1,257,637 rows, 24 fixed interactions, feature hash
`ce9fc5180dd2c065562000cdb80ecf2606d5320057b2e6e2cc95f8366580af55`.

| Outer | Baseline | Final | Delta | Alpha | Best iteration |
|---|---:|---:|---:|---:|---:|
| PSEUDO | 0.142550340 | 0.142942651 | +0.000392311 | 0.06 | 84 |
| H2 | 0.141861992 | 0.142292065 | +0.000430072 | 0.05 | 20 |
| T3 | 0.143549308 | 0.143962796 | +0.000413488 | 0.04 | 22 |
| T4 | 0.157053101 | 0.157723369 | +0.000670267 | 0.04 | 22 |
| Mean delta |  |  | **+0.000476535** |  |  |

All four folds were finite and positive, with drift ratios inside the frozen
bounds. The gate failed because PSEUDO was below `+0.0015`; M04 is not added to
the conditional ensemble.

Feature artifact revision: `d45ecfb`; formal fold runner revision: `9b117e3`.
CatBoost was fixed at 3000 iterations,
early stopping 200, seed 2026, with the frozen alpha grid.
