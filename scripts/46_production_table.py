# -*- coding: utf-8 -*-
"""
Clean Table v2 生产推理: 全量(0-70)训练 + test 预测
复用 src 组件 (CleanTableConfig / apply_r2 / _lgb_params / _cat_params / _train_mlp_seed / _table_blend)
输出: table_test_pred.npz (647,896)
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\mscapital-kaggle\src")
from mscapital.models.clean_table import (
    CleanTableConfig, load_table_frame, apply_r2, _lgb_params, _cat_params,
    _train_mlp_seed, _table_blend, StandardClip,
)
from mscapital.metrics import cosine_uncentered

DATA = r"D:\mscapital-forecasting\data\processed"
RAW = r"D:\mscapital-forecasting\data\raw"
OUT = r"D:\mscapital-forecasting\data\processed\p12_out"

t0 = time.time()
cfg = CleanTableConfig()
print("cfg:", cfg.__dict__, flush=True)

# 1. train frame
frame = load_table_frame(
    f"{DATA}/train_features.parquet", f"{DATA}/micro_features_train.parquet",
    f"{RAW}/train/label.feather",
)
x_all, y_all = frame.values, frame.target
print(f"train {x_all.shape} ({time.time()-t0:.0f}s)", flush=True)

# 2. test frame (与 load_table_frame 相同 pipeline)
te = pd.read_parquet(f"{DATA}/test_features.parquet")
micro_te = pd.read_parquet(f"{DATA}/micro_features_test.parquet")
official = [c for c in te.columns if c != "sample_id"]
micro_names = [c for c in micro_te.columns if c != "sample_id"]
from mscapital.models.clean_table import R2_REPLACEMENTS
r2_frame_te = te[["sample_id", *official]].copy()
for output, (numerator, denominator, offset) in R2_REPLACEMENTS.items():
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_frame_te[output] = (
            r2_frame_te[numerator].to_numpy(dtype=np.float64)
            / (r2_frame_te[denominator].to_numpy(dtype=np.float64) + offset)
        )
merged_te = r2_frame_te.merge(micro_te, on="sample_id", how="left", validate="one_to_one", sort=False)
merged_te[micro_names] = merged_te[micro_names].fillna(0.0)
order_te = np.argsort(te["sample_id"].to_numpy(), kind="stable")
te_ids = te["sample_id"].to_numpy()[order_te].astype(np.int64)
missing = [c for c in frame.feature_names if c not in merged_te.columns]
assert not missing, f"missing cols: {missing}"
x_test = merged_te[list(frame.feature_names)].to_numpy(dtype=np.float64)[order_te]
print(f"test {x_test.shape} ({time.time()-t0:.0f}s)", flush=True)

# 3. 全量训练
import lightgbm as lgb
from catboost import CatBoostRegressor

N_ITER = 5000
print("LGBM full...", flush=True)
m = lgb.train(_lgb_params(cfg), lgb.Dataset(x_all, y_all), N_ITER)
p_lgb = m.predict(x_test)
print(f"  lgb done ({time.time()-t0:.0f}s)", flush=True)

print("CatBoost full...", flush=True)
m = CatBoostRegressor(**_cat_params(cfg, N_ITER))
m.fit(x_all, y_all)
p_cat = m.predict(x_test)
print(f"  cat done ({time.time()-t0:.0f}s)", flush=True)

print("MLP full 3seeds...", flush=True)
scaler = StandardClip().fit(x_all)
xs, xst = scaler.transform(x_all), scaler.transform(x_test)
mlp_preds = []
for seed in cfg.mlp_seeds:
    p, _, _ = _train_mlp_seed(xs, y_all, xst, cfg, seed)
    mlp_preds.append(p)
    print(f"  mlp seed{seed} done ({time.time()-t0:.0f}s)", flush=True)
p_mlp = np.mean(mlp_preds, axis=0)

# 4. blend
p_blend = _table_blend({"lgb": p_lgb, "cat": p_cat, "mlp": p_mlp})
np.savez(f"{OUT}/clean_table_test_pred.npz", pred=p_blend, lgb=p_lgb, cat=p_cat, mlp=p_mlp, test_ids=te_ids)
print(f"\nsaved clean_table_test_pred.npz ({len(p_blend):,})", flush=True)
print(f"mean={p_blend.mean():.2e} std={p_blend.std():.2e}", flush=True)
print("ALL DONE", flush=True)
