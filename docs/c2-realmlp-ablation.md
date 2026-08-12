# C2 RealMLP Ablation Protocol

C2 starts from the completed C1 Clean RealMLP-v2a configuration. It does not
change the nested temporal registry, preprocessing rules, loss, seed, model
width, RQ encoder/head mismatch, or outer scoring protocol unless that item is
the single named ablation.

## D0: training ceiling diagnostic

The three distinct inner searches (PSEUDO, H2, and T3; T4 duplicates T3) run
30 epochs with the complete learning-rate, noise, and RQ schedule stretched to
30 epochs. The diagnostic reads only registered inner train/tune months and
does not refit or score outer validation.

The 30-epoch schedule is eligible for a formal four-outer run only when:

- at least two of the three distinct inner searches improve over their
  same-hardware C1 reference;
- mean inner gain is at least `0.0003`;
- worst inner gain is no lower than `-0.0005`.

This is a screening rule, not a C1 result rewrite. A surviving schedule must
still pass the frozen C2 outer gate below.

## Single-variable matrix

Each single-variable candidate first runs the same three distinct inner
searches (PSEUDO, H2, and T3) and uses the D0 screening gate. Candidates that
fail this pre-screen do not consume four formal outer runs. A survivor becomes
a formal candidate and must then pass the frozen outer gate below.

| ID | Changed field | Candidate | C1 reference |
|---|---|---|---|
| C2-O | optimizer grouping | first `NTPLinear` (`shared.2.weight`) | legacy `shared.0.weight` LayerNorm quirk |
| C2-MF | ensemble mask | full mask, member `i` drops `i::16` | half mask, member `i` drops `i::8` |
| C2-MN | ensemble mask | no mask | half mask |
| C2-TR | training target | raw target | round(4) |

All other fields remain equal to C1. Inner tune chooses best progress; each
formal survivor is then reinitialized, refit with its own fold-local
preprocessing/RQ state, and scored once on each registered outer.

## Frozen outer gate

A single ablation can enter the eventual Clean Baseline configuration only if:

- at least 3 of 4 outer deltas are positive;
- arithmetic mean outer delta is at least `0.0003`;
- worst outer delta is no lower than `-0.0005`;
- IDs, months, and targets align exactly with C1 and predictions are finite.

The outers are correlated temporal stress tests, not independent samples.
Passing candidates remain subject to a combined-configuration confirmation;
individual gains are not assumed additive.

The optional builder `--push` publishes a private Kaggle training kernel, but
no C2 command creates a competition prediction submission. Predictions,
weights, logs, credentials, and machine-specific paths remain private
artifacts.

## Completed result

The 30-epoch schedule passed both frozen gates: 3/3 positive distinct inner
searches with mean delta `+0.004773214`, followed by 4/4 positive outer stress
tests with mean delta `+0.002593728` and worst delta `+0.000339307`.

See [`c2-realmlp-results.md`](c2-realmlp-results.md) for the full diagnostics
and the single-variable ablation outcomes.
