# -*- coding: utf-8 -*-
import numpy as np
import polars as pl

OUT = r"D:\mscapital-kaggle\output"

raw = pl.read_parquet(rf"{OUT}\p5b_scfi\raw_ot_agg.parquet").with_columns(pl.all().fill_null(0.0))
raw_np = raw.select([c for c in raw.columns if c.startswith(("ob_", "tb_"))]).to_numpy()
raw_names = [c for c in raw.columns if c.startswith(("ob_", "tb_"))]
bad = [(raw_names[i], int(np.isinf(raw_np[:, i]).sum()), int(np.isnan(raw_np[:, i]).sum()))
       for i in range(raw_np.shape[1]) if np.isinf(raw_np[:, i]).sum() or np.isnan(raw_np[:, i]).sum()]
print("rawagg non-finite cols:", bad[:15], "total", len(bad))

ms = pl.read_parquet(rf"{OUT}\m05_state\market_state_train.parquet")
ms_np = ms.select([c for c in ms.columns if c not in ("sample_id", "month", "target")]).to_numpy()
ms_names = [c for c in ms.columns if c not in ("sample_id", "month", "target")]
bad2 = [(ms_names[i], int(np.isinf(ms_np[:, i]).sum()), int(np.isnan(ms_np[:, i]).sum()))
        for i in range(ms_np.shape[1]) if np.isinf(ms_np[:, i]).sum() or np.isnan(ms_np[:, i]).sum()]
print("mstate non-finite cols:", bad2[:15], "total", len(bad2))

# 检查 builder 里可能产生 inf 的派生列: 在 polars 里 0/0 → null? inf?
if bad:
    print("rawagg sample values:", raw.select([b[0] for b in bad[:5]]).head(3).to_dict(as_series=False))
