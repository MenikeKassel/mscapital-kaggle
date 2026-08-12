# -*- coding: utf-8 -*-
"""
Exp B1: 特征组消融 (Leave-One-Group-Out)
RQ: 增量预测力来自哪组特征? (窗口统计 / EWM / 交叉 / 成交 / 订单 / 盘口基础)
协议: CV1 (train m0-50 / valid m51-70), 官方参数, 每次仅移除一组
"""
import re
import time
import numpy as np
import polars as pl
import lightgbm as lgb

FEAT = r"D:\mscapital-forecasting\data\processed\train_features.parquet"
N_THREADS = 12

tr = pl.read_parquet(FEAT)
all_feats = [c for c in tr.columns if c not in ("sample_id", "month", "target")]

def is_window(f):
    return bool(re.search(r"_(15|45|60|120|180)$", f)) and "_ewm_" not in f

def is_ewm(f):
    return "_ewm_" in f

def is_cross(f):
    return f.startswith("x_")

def is_tx(f):
    return f.startswith("t_")

def is_ord(f):
    return f.startswith("o_")

def is_mkt(f):
    return f.startswith("m_")

# 互斥分组: 优先级 交叉(x_) > 窗口 > EWM > 前缀
GROUPS = {
    "交叉(x_)": [f for f in all_feats if is_cross(f)],
    "窗口统计(非EWM/x)": [f for f in all_feats if is_window(f) and not is_cross(f)],
    "EWM(非x)": [f for f in all_feats if is_ewm(f) and not is_cross(f)],
    "成交基础(t_)": [f for f in all_feats if is_tx(f) and not is_window(f) and not is_ewm(f)],
    "订单基础(o_)": [f for f in all_feats if is_ord(f) and not is_window(f) and not is_ewm(f)],
    "盘口基础(m_)": [f for f in all_feats if is_mkt(f) and not is_window(f) and not is_ewm(f)],
}
for name, feats in GROUPS.items():
    print(f"组 [{name}]: {len(feats)} 特征", flush=True)
n_covered = sum(len(v) for v in GROUPS.values())
assert n_covered == len(all_feats), f"分组覆盖 {n_covered}/{len(all_feats)}"
seen = set()
for name, feats in GROUPS.items():
    dup = seen & set(feats)
    assert not dup, f"组 [{name}] 与已有重叠: {dup}"
    seen |= set(feats)

PARAMS = dict(
    objective="regression", metric="rmse",
    learning_rate=0.02, num_leaves=32, min_data_in_leaf=300,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    lambda_l2=5.0, max_bin=255, verbose=-1, num_threads=N_THREADS, seed=0)

def cos_uncenter(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def run(feats):
    tr_df = tr.filter(pl.col("month") <= 50)
    va_df = tr.filter((pl.col("month") > 50) & (pl.col("month") <= 70))
    X_tr = tr_df.select(feats).to_numpy().astype(np.float32)
    y_tr = tr_df["target"].to_numpy().astype(np.float32)
    X_va = va_df.select(feats).to_numpy().astype(np.float32)
    y_va = va_df["target"].to_numpy().astype(np.float32)
    dtr = lgb.Dataset(X_tr, y_tr)
    dva = lgb.Dataset(X_va, y_va, reference=dtr)
    model = lgb.train(PARAMS, dtr, num_boost_round=10000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200)])
    p_va = model.predict(X_va, num_iteration=model.best_iteration)
    return cos_uncenter(p_va, y_va), model.best_iteration

BASE = 0.130204  # A2/B1 全集 CV1
print(f"\nbaseline (90特征) = {BASE:.6f}", flush=True)

results = []
for name, feats in GROUPS.items():
    remain = [f for f in all_feats if f not in feats]
    t0 = time.time()
    c, it = run(remain)
    drop = BASE - c
    results.append((name, len(feats), len(remain), c, drop, it, time.time() - t0))
    print(f"移除[{name}]: n={len(remain)} cos={c:.6f} (drop={drop:+.6f}) iter={it} ({time.time()-t0:.1f}s)", flush=True)

print("\n=== Exp B1 汇总 (按drop排序, drop越大贡献越大) ===")
for name, nf, nr, c, drop, it, dt in sorted(results, key=lambda r: -r[4]):
    print(f"{drop:+.6f}  移除[{name}] ({nf}特征) -> cos={c:.6f}")
