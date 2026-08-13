# M05 Historical Market-State KNN results

Status: **complete, gate failed; M05 is closed for this execution queue**.

The state artifact contains 1,257,637 rows and the fixed 16-dimensional state
schema. Historical prototypes were restricted to `source_month < query_month`.
No LB142, test prediction, or competition submission was used.

| Outer | Baseline | Final | Delta | k | Alpha |
|---|---:|---:|---:|---:|---:|
| PSEUDO | 0.142550340 | 0.142817722 | +0.000267382 | 16 | 0.04 |
| H2 | 0.141861992 | 0.141823877 | -0.000038116 | 16 | 0.08 |
| T3 | 0.143549308 | 0.143549308 | 0.000000000 | 8 | 0.00 |
| T4 | 0.157053101 | 0.157053101 | 0.000000000 | 8 | 0.00 |
| Mean delta |  |  | **+0.000057317** |  |  |

All predictions were finite and drift checks passed, but only 2/4 folds were
positive and PSEUDO was below `+0.0015`. M05 is not a candidate for the
conditional ensemble.

Feature artifact revision: `3f10422`; formal fold runner revision: `6a25d48`.
