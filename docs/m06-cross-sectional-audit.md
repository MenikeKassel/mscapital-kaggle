# M06 Cross-sectional Dynamics audit

Status: **not_identifiable; frozen without predictions**.

The train and test schemas expose `sample_id` and engineered features, but no
shared semantic asset key together with a shared time key. Therefore a
deployable train/test cross-sectional group cannot be reconstructed without
guessing from `sample_id` or using future test-unavailable ranks.

The audit result is:

```json
{
  "status": "not_identifiable",
  "reason": "no reproducible asset/time cross-section in train and test",
  "prediction_artifact": null
}
```

M06 is complete for the current schema and will not generate a prediction or
enter the residual ensemble.
