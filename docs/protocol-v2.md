# Protocol-v2

Protocol-v2 is the leakage-safe successor to the frozen numbered scripts.
The only model-selection metric is uncentered cosine:

    dot(pred, target) / (norm(pred) * norm(target))

Centered cosine/Pearson is a diagnostic only. Every learned preprocessing
state is fitted on the training portion of a temporal split and then reused
without refitting on validation or test data. Quantile transforms use a
dedicated missing bin and clip finite out-of-domain values to the nearest
training bin.

The registered outer stress folds are PSEUDO, H2, T3 and T4. Their inner
temporal calibration sets are defined in mscapital.splits; outer validation
rows never select weights, alpha or early-stopping progress.

Generated artifacts belong under output/experiments/ and contain a manifest,
prediction NPZ and a human-readable report. The CLI never submits to Kaggle.
