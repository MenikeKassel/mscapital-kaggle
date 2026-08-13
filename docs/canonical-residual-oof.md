# Canonical Clean Baseline rolling OOF

This stage creates the only baseline OOF artifact allowed for residual research.
It is not an average of overlapping outer folds. Each sample is predicted once
by a model refit strictly before its month.

| Block | Inner train | Inner tune | Refit | Predict |
|---|---:|---:|---:|---:|
| R21_30 | 0-10 | 11-20 | 0-20 | 21-30 |
| R31_40 | 0-20 | 21-30 | 0-30 | 31-40 |
| R41_50 | 0-30 | 31-40 | 0-40 | 41-50 |
| R51_60 | 0-40 | 41-50 | 0-50 | 51-60 |
| R61_70 | 0-50 | 51-60 | 0-60 | 61-70 |

For every block, RealMLP and Table are independently trained under their frozen
C2/C3 configurations. The Clean Baseline rule is fixed to RMS and Table weight
`0.37`, but each block learns its two RMS scales from that block's historical
inner-tune predictions. Production scales from months 51-70 are never replayed
onto earlier blocks.

Build one verified block:

```powershell
python -m mscapital build-clean-baseline-oof-block `
  --realmlp-dir output/rolling_components/realmlp/R21_30 `
  --table-dir output/rolling_components/table/R21_30 `
  --split R21_30 `
  --output-root output/canonical_oof_blocks
```

Merge all five formal block directories:

```powershell
python -m mscapital build-residual-oof `
  --block R21_30=output/canonical_oof_blocks/R21_30 `
  --block R31_40=output/canonical_oof_blocks/R31_40 `
  --block R41_50=output/canonical_oof_blocks/R41_50 `
  --block R51_60=output/canonical_oof_blocks/R51_60 `
  --block R61_70=output/canonical_oof_blocks/R61_70 `
  --output output/canonical_residual_oof/canonical_residual_oof.npz
```

The loader verifies component identity/config/months/data fingerprints, block
manifest hashes, exact ten-month coverage, per-array hashes, globally unique
`sample_id`, finite target/prediction, and `source_train_end < month`. Smoke
blocks are marked explicitly and cannot enter the canonical merge.

Outer residual views are fixed as follows:

- PSEUDO: months 21-32.
- H2: months 21-40.
- T3 and T4: months 21-50. T4 cannot read months 51-60.

Predictions and manifests remain under ignored `output/`; no command in this
workflow creates a Kaggle competition submission.
