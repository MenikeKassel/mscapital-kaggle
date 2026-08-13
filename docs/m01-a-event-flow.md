# M01-A Event Flow residual experiment

M01-A is the first Method Transfer experiment after Clean Baseline v2. It does
not compute Quote OFI, Geometry, fast-slow differences, or interactions.

For Order and Trade separately, and for 5/15/30/60 second lookbacks, it builds:

- signed volume per second;
- signed volume per event;
- event count per second.

Order signs are bid add `+`, bid cancel `-`, ask add `-`, ask cancel `+`.
Trade signs are bid `+`, ask `-`. Events with negative
`seconds_before_predict` are excluded. Empty windows and samples with no event
rows receive zero. No cumulative flow from different windows is subtracted.

The production builder uses Polars lazy streaming group aggregation over the
raw Feather files rather than the small-array Python grouping reference:

```powershell
python -m mscapital build-event-flow `
  --order D:/data/raw/train/order.feather `
  --transaction D:/data/raw/train/transaction.feather `
  --labels D:/data/raw/train/label.feather `
  --output output/m01a_features/event_flow_train.parquet
```

Residual CatBoost uses the frozen `configs/m01-a.json`: 3000 maximum
iterations, 200-round early stopping, and seed 2026. For each outer fold it:

1. loads only the visible canonical rolling OOF and estimates
   `beta=(p dot y)/(p dot p)`;
2. trains on the registered residual inner-train months;
3. selects the earliest CatBoost best iteration and alpha from `0.00-0.30` in
   `0.01` increments on residual inner-tune;
4. refits at that iteration on every OOF row visible to the outer;
5. applies the inner-tune RMS scales and alpha to frozen Clean Baseline outer
   predictions plus the residual prediction.

The selection API has no outer-target argument. Outer target is read only after
selection/refit to produce replay diagnostics and the candidate gate.

```powershell
python -m mscapital run-m01a `
  --canonical-oof output/canonical_residual_oof/canonical_residual_oof.npz `
  --features output/m01a_features/event_flow_train.parquet `
  --baseline-root output/c4_protocol_closed_final/clean-baseline-v2 `
  --output-root output/m01a_formal `
  --outer ALL
```

The gate requires PSEUDO delta at least `+0.0015`, at least three positive
outers, worst delta at least `-0.0005`, finite final predictions, validation to
outer standard-deviation ratio in `[0.67, 1.50]`, and absolute-p99 ratio in
`[0.50, 2.00]`. The experiment stops after M01-A regardless of outcome; M01-B
is a separate decision.

LB142 is a test-only reference in the local assets and has no trustworthy
outer-valid `sample_id` mapping. Its outer correlation is therefore reported as
N/A rather than fabricated. Baseline correlation is always reported.
