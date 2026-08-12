# C2 RealMLP Ablation Results

Completed on 2026-08-13. C2 changed one RealMLP training choice at a time
relative to C1 and kept the nested temporal protocol, preprocessing, seed,
architecture, loss, and outer scoring fixed.

## Main result: 30-epoch schedule

The only change was stretching the complete learning-rate, label-noise, and RQ
schedule from 10 to 30 epochs. Each outer still selected its stopping progress
on inner tune, then reinitialized and refit with fold-local preprocessing and
RQ state before predicting outer validation.

### Inner screening (Kaggle P100)

| Inner search | C1 (10 epoch) | 30 epoch | Best epoch | Delta |
|---|---:|---:|---:|---:|
| PSEUDO | 0.133570070 | 0.138207164 | 22/30 | +0.004637093 |
| H2 | 0.130564161 | 0.136019455 | 19/30 | +0.005455294 |
| T3/T4 | 0.136832628 | 0.141059884 | 20/30 | +0.004227256 |
| Arithmetic mean delta | - | - | - | **+0.004773214** |

The frozen inner gate passed with 3/3 positive searches and a worst delta of
`+0.004227256`.

### Formal outer validation (Kaggle P100)

| Outer | C1 | 30 epoch | Delta | Pearson | Corr vs C1 | Pred mean | Pred std | Target std | Best progress | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PSEUDO | 0.136040611 | 0.136379917 | +0.000339307 | 0.137317384 | 0.919564810 | -0.001609819 | 0.011924376 | 0.002534166 | 22/30 | 1,566.1 |
| H2 | 0.135809991 | 0.138287880 | +0.002477889 | 0.138329438 | 0.961836950 | -0.002038311 | 0.014141995 | 0.002495371 | 19/30 | 1,989.9 |
| T3 | 0.136877769 | 0.139785533 | +0.002907765 | 0.139554359 | 0.954993157 | -0.001553541 | 0.014540298 | 0.002495371 | 20/30 | 2,386.8 |
| T4 | 0.145643884 | 0.150293835 | +0.004649951 | 0.152462761 | 0.946900248 | -0.001394438 | 0.016328691 | 0.002981453 | 20/30 | 2,374.3 |
| Arithmetic mean | **0.138593064** | **0.141186791** | **+0.002593728** | - | - | - | - | - | - | - |

The frozen outer gate passed:

- positive outers: `4/4`;
- mean delta: `+0.002593728` (required at least `+0.0003`);
- worst delta: `+0.000339307` (required at least `-0.0005`);
- every prediction was finite and IDs, months, and targets matched C1 exactly.

The outer gain is smaller than the inner screening gain, especially on
PSEUDO. This is why C2 reports the two stages separately. The conclusion is
not that 30 epochs universally adds about `0.0048`; it is that the schedule
change survives all four registered outer stress tests with a descriptive mean
gain of about `0.0026`.

## Single-variable diagnostics (local RTX, inner only)

These candidates were compared to the same-hardware local C1 inner histories.
They were not mixed with the P100 values above.

| Candidate | PSEUDO delta | H2 delta | T3 delta | Mean delta | Positive | Inner gate |
|---|---:|---:|---:|---:|---:|---|
| Full mask (`i::16`) | +0.000063 | +0.000161 | +0.000540 | +0.000255 | 3/3 | Fail |
| No mask | +0.000021 | +0.000746 | +0.000932 | +0.000567 | 3/3 | Pass |
| First-`NTPLinear` optimizer group | -0.000096 | -0.000169 | -0.000182 | -0.000149 | 0/3 | Fail |
| Raw target | -0.000388 | +0.000231 | -0.000271 | -0.000143 | 1/3 | Fail |

No-mask is a weak inner-screen survivor, not a confirmed outer improvement.
It remains an optional follow-up after the main 30-epoch result. Full-mask,
corrected optimizer grouping, and raw-target training do not advance.

## Research interpretation

- C1 did not fail because the RealMLP representation was invalid. Its main
  correctable limitation was a 10-epoch schedule that stopped while all three
  distinct inner curves were still improving.
- The optimizer-grouping quirk is not beneficial merely because correcting it
  looks cleaner in code; the corrected group was negative in all three inner
  searches.
- Target rounding remains the C1 default because removing it failed the gate.
- No-mask may contain a small signal, but its inner gain is roughly one eighth
  of the 30-epoch gain and has no outer evidence yet.
- The four outers are correlated temporal stress tests, not four independent
  samples. Arithmetic means are descriptive only.

## Integrity and publication

- All four formal runs used source revision
  `3f7dc16e2cf809e3c1920107b7cf2d8d4fe2d068` and config hash
  `de2e26a288361c46016a4462c4eee2201db9d05937f120200297f4a980a79da1`.
- Feature hashes differ only where the permitted historical training window
  differs; T3 and T4 share the same feature hash and preprocessing history.
- Prediction NPZ files, model weights, Kaggle logs, OAuth credentials, and
  machine-specific paths are excluded from the public repository.
- C2 did not create a Kaggle competition submission.

The 30-epoch schedule is accepted as the RealMLP candidate for the remaining
Clean Baseline work. Clean Baseline v2 is not frozen until Clean Table and
nested ensemble calibration (C3/C4) are complete.
