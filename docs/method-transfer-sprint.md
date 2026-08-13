# Method Transfer Sprint

Clean Baseline v2 is the fixed reference point, not the end of the research
program. New representations are evaluated as residual alpha and ranked by
stable blend gain, residual information, low repetition with the baseline and
regime stability.

1. M01 Multi-Level Dynamic OFI (`Cont OFI → Xu MLOFI → Kolm Deep OFI`):
   event flow (M01-A done), quote OFI at L1/L2, liquidity-normalized
   multi-scale dynamics, ΔOFI, fast-slow, OFI velocity, cross-flow
   interactions.
2. M02 Market-Centered LOB Geometry (`JPM Wu et al. arXiv:2110.05479`):
   (price-mid)/spread, cumulative depth, depth slope, relative depth shares,
   entropy/HHI and bid/ask asymmetry. Fixed-dimension summaries for
   RealMLP, no LOB tensor replication. The source has only two book levels,
   so no L3+ or fabricated curvature is used.
3. M03 Path Signature (`Rough Path / Chevyrev-Kormilitzin`): six to eight
   market/event channels, 5/15/30/60s windows, depth 2 before any depth-3
   expansion.
4. M04 Residual Market-State KNN (`Optiver RV 1st "Nearest Neighbors"`):
   historical-only neighbors and residual prediction, judged by blend gain
   rather than standalone score; target corr(RealMLP, KNN) < 0.8.
5. M05 Optiver Interaction Family (`TATC heuristics, no paper`): triplet
   imbalance, market urgency, depth pressure, OFI×spread, OFI×intensity.
   Lowest implementation cost; do not cite a nonexistent paper.
6. M06 Cross-sectional Dynamics (`Cross-Impact OFI arXiv:2112.13213 +
   TATC synthetic index`): rank(OFI/velocity/volatility), market_mean,
   asset − market only; no full relative pass (N006).

MLPLOB is an architecture track after a representation has passed. Label
Horizon is frozen because the shipped labels do not expose legal intermediate
future paths. SSL and TabPFN remain later prototypes; TCN, Mamba and large
Transformer tracks remain frozen.
