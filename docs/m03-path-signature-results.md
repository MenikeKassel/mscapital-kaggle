# M03 depth-2 Path Signature results

Status: **complete, gate failed; conditional RealMLP/MLPLOB branches were not started**.

The fixed representation has 112 columns (seven channels, depth 2, four
windows). It was built from the full raw train market/order/transaction files
with an old-to-new one-second as-of grid. Invalid quote rows were filtered by
the frozen usable-book rule; no future quote or test prediction was used.

| Outer | Baseline | Final | Delta | Alpha | Best iteration |
|---|---:|---:|---:|---:|---:|
| PSEUDO | 0.142550340 | 0.142817613 | +0.000267273 | 0.09 | 335 |
| H2 | 0.141861992 | 0.141929104 | +0.000067112 | 0.01 | 4 |
| T3 | 0.143549308 | 0.143805989 | +0.000256681 | 0.03 | 25 |
| T4 | 0.157053101 | 0.157274685 | +0.000221584 | 0.03 | 25 |
| Mean delta |  |  | **+0.000203163** |  |  |

All four predictions were finite, positive, and inside the drift bounds. The
gate failed because PSEUDO was below `+0.0015`; therefore the conditional
RealMLP and MLPLOB branches were correctly not launched.

Feature artifact: 1,257,637 rows, 112 features, feature hash
`05653b212ca52ffa38a15b0e2dacbe289472f0a554cb85afc69508d5b14bf78b`, values
hash `db5d40860f4d7113711e1d534f920b561cd8d39c9603ee7368539c27664da757`,
invalid raw quote rows `199,603,075`. Builder revision: `ee45d01`.

M03 is recorded as a valid negative result for this fixed depth-2 recipe. The
next independent method in the queue is the already-completed M04 report.
