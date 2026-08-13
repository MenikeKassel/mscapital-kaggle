# M02-base Market-Centered LOB Geometry

M02 is the first follow-up after M01-A failed its PSEUDO gate. It uses only
the latest L1/L2 quote per labelled sample and the frozen residual CatBoost
protocol. L1 relative prices are omitted because they are constant at ±0.5;
no L3+, curvature, or large HLOB graph features are constructed.

The production feature artifact is generated locally from the raw market
Feather in a streaming reader. Samples with no quote rows are retained with
zero-valued geometry and an explicit `lob_quote_missing` indicator. The
artifact is ignored by Git and is not a competition submission.

Commands:

```powershell
python -m mscapital build-geometry-file `
  --market "$env:MSCAP_DATA_ROOT\raw\train\market.feather" `
  --labels "$env:MSCAP_DATA_ROOT\raw\train\label.feather" `
  --output "$env:MSCAP_ARTIFACT_ROOT\m02_geometry_features\geometry_train.parquet"

python -m mscapital run-m02 `
  --canonical-oof "$env:MSCAP_ARTIFACT_ROOT\canonical_residual_oof\canonical_residual_oof.npz" `
  --features "$env:MSCAP_ARTIFACT_ROOT\m02_geometry_features\geometry_train.parquet" `
  --baseline-root "$env:MSCAP_ARTIFACT_ROOT\c4_protocol_closed_final\clean-baseline-v2" `
  --output-root "$env:MSCAP_ARTIFACT_ROOT\m02_formal" --outer ALL
```

The four-fold M02-base result is recorded separately in
`docs/m02-geometry-results.md`. Temporal dependency/lead-lag subexperiments
are intentionally not included in this result.
