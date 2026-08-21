# 方法族地图

| 路线 | 状态 | 证据数 | 结果 | 终止/下一门 |
|---|---|---|---|---|
| R01-table-baseline | frozen | 15 | Clean Table lineage is historical anchor | R04-realmlp-clean |
| R02-r2-drift | frozen | 6 | R2 retained in 152+73Z | R04-realmlp-clean |
| R03-micro-primitives | frozen | 5 | micro features absorbed into frozen asset | R04-realmlp-clean |
| R04-realmlp-clean | frozen | 16 | Clean Baseline v2 frozen | R17-realmlp-recipe |
| R05-sequence | closed |  | TCN catastrophic LB and weak stable evidence | model_saturation |
| R06-m-residual | closed | 7 | all residual families are insufficient for the gate | insufficient |
| R07-state-conditioned | closed | 3 | E01 positive but PSEUDO below gate; E02/E03 diagnostic | insufficient |
| R08-unsupervised-latent | closed | 5 | SAE/TinyLOBERT/grid/NHP no stable gain | model_saturation |
| R09-hidden-information | external | 17 | LB142 forensic and market history evidence | R20-submissions |
| R10-amplitude-gate | closed | 1 | outer gate non-positive | objective_mismatch |
| R11-scfi-z | frozen | 14 | 152+73Z is production asset; P10 RQ run invalid | R20-submissions |
| R12-geometry-signature | closed |  | M02/M03/M04 below gate; no extension | insufficient |
| R13-p6r-production | closed | 5 | positive but below gate; P6-ORIG anchor retained | insufficient |
| R14-o-to-t | closed | 2 | no reproducible lead-lag | nonstationarity |
| R15-p9-quant | closed | 3 | neutralization insufficient; NCL/V-REx negative | objective_mismatch |
| R16-cancel-eventtime | closed | 5 | Cancel absorbed by Z; Event-Time negative; M55 and neutralization insufficient | redundancy |
| R17-realmlp-recipe | candidate |  | ParamMish/PL/PBLD/schedreg/coslog4 are candidate components | R18-blsm |
| R18-blsm | active | 1 | G0 validates behavior latent existence; G1 pre-registered | R17-realmlp-recipe |
| R19-production-calibration | external |  | external lb142 and self-owned anchors separated | N/A |
| R20-submissions | frozen | 11 | submission ledger is evidence-only | N/A |
