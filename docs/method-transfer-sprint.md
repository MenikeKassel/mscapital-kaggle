# Method Transfer Sprint

Clean Baseline v2 is the fixed reference point, not the end of the research
program. New representations are evaluated as residual alpha and ranked by
stable blend gain, residual information, low repetition with the baseline and
regime stability.

1. M01 Dynamic Multi-Level Invariant OFI: event flow, quote OFI at L1/L2,
   liquidity-normalized multi-scale dynamics and cross-flow interactions.
2. M02 Market-Centered LOB Geometry: L1/L2 relative book shape, depth shares,
   slope, entropy/HHI and bid/ask asymmetry. The source has only two book
   levels, so no L3+ or fabricated curvature is used.
3. M03 Path Signature: six to eight market/event channels, 5/15/30/60s
   windows, depth 2 before any depth-3 expansion.
4. M04 Residual Market-State KNN: historical-only neighbors and residual
   prediction, judged by blend gain rather than standalone score.

MLPLOB is an architecture track after a representation has passed. Label
Horizon is frozen because the shipped labels do not expose legal intermediate
future paths. SSL and TabPFN remain later prototypes; TCN, Mamba and large
Transformer tracks remain frozen.
