# C1 Clean RealMLP-v2a

This is the first executable protocol-v2 training milestone. It runs one
outer fold at a time and keeps all feature selection, quantile boundaries,
categorical vocabularies, robust scales, RQ codebooks, best-progress
selection, and model refits inside the permitted historical months.

## Local smoke

From the repository root, expose the source tree and use the project venv:

```powershell
& '.\.venv\Scripts\python' -m pip install -e '.[clean-realmlp]'
$env:PYTHONPATH = "$PWD\src"
& '.\.venv\Scripts\python' -m mscapital clean-realmlp `
  --config '.\configs\clean-realmlp-v2a.json' `
  --outer ALL --device cuda --max-rows-per-month 512
```

The smoke flag is only for shape/finite/leakage checks; its score is not a
competition result. Full local data can be run by omitting
`--max-rows-per-month`, but the formal C1 evidence is the four Kaggle P100
kernels.

## Kaggle kernels

Generate the four self-contained kernel directories:

```powershell
& '.\.venv\Scripts\python' '.\scripts\build_kaggle_c1.py' `
  --repo '.' --output '.\output\kaggle_c1_kernels'
```

Authenticate the Kaggle CLI in the current user environment first. The push
step is intentionally explicit and never submits a competition prediction:

```powershell
& '.\.venv\Scripts\python' -m kaggle auth login
& '.\.venv\Scripts\python' '.\scripts\build_kaggle_c1.py' `
  --repo '.' --push
```

Each kernel writes `inner_predictions.npz`, `predictions.npz`,
`training_history.json`, `manifest.json`, and `report.md` to its Kaggle
working directory. Download those directories into the artifact root, then
summarize locally:

```powershell
& '.\.venv\Scripts\python' -m mscapital summarize-clean-realmlp `
  --artifact-root '.\output\experiments' `
  --experiment-id clean-realmlp-v2a `
  --legacy-pseudo '.\output\rlps_v12\realmlp_pseudo_pred.npz'
```

The summary reports the four correlated outer stress tests, their arithmetic
mean, and the PSEUDO-only comparison with legacy RealMLP. It does not freeze
Clean Baseline v2 or launch C2 ablations.

The completed C1 evidence and interpretation are recorded in
[`c1-clean-realmlp-results.md`](c1-clean-realmlp-results.md). Predictions and
model weights remain private artifacts and are not committed.
