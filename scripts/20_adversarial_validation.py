# -*- coding: utf-8 -*-
"""
P0-2: Adversarial Validation (GPT 拍板执行)
三组: A: m0-50 vs m51-70 (train内部漂移) / B: m51-70 vs test / C: full train vs test
输出: AUC, 特征重要性TOP20, 特征组漂移表, Predictive×Drift 四象限数据
"""
import numpy as np
import polars as pl
import lightgbm as lgb
from scipy import stats

FEAT_TRAIN = r"D:\mscapital-forecasting\data\processed\train_features.parquet"
FEAT_TEST = r"D:\mscapital-forecasting\data\processed\test_features.parquet"
N_THREADS = 6

tr = pl.read_parquet(FEAT_TRAIN)
te = pl.read_parquet(FEAT_TEST)
all_feats = [c for c in tr.columns if c not in ("sample_id", "month", "target")]
X_tr = tr.select(all_feats).to_numpy().astype(np.float32)
m_tr = tr["month"].to_numpy().astype(np.int32)
X_te = te.select(all_feats).to_numpy().astype(np.float32)
print(f"train {X_tr.shape} test {X_te.shape}", flush=True)

def auc(y_true, y_score):
    n_pos = int(y_true.sum()); n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    r = stats.rankdata(y_score)
    return float((r[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def adv_run(name, Xa, Xb, label_a=0):
    """Xa=domain0, Xb=domain1"""
    X = np.vstack([Xa, Xb])
    y = np.concatenate([np.zeros(len(Xa), dtype=np.int32), np.ones(len(Xb), dtype=np.int32)])
    params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, verbose=-1, num_threads=N_THREADS, seed=0)
    # 简单切分做早停 (80/20)
    idx = np.random.RandomState(0).permutation(len(X))
    cut = int(len(X) * 0.8)
    dtr = lgb.Dataset(X[idx[:cut]], y[idx[:cut]])
    dva = lgb.Dataset(X[idx[cut:]], y[idx[cut:]])
    m = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(50)])
    p = m.predict(X, num_iteration=m.best_iteration)
    a = auc(y, p)
    imp = m.feature_importance(importance_type="gain")
    order = np.argsort(-imp)
    top = [(all_feats[i], float(imp[i])) for i in order[:20]]
    # 特征组聚合
    groups = {}
    for i, f in enumerate(all_feats):
        if f.startswith("x_"):
            g = "x_cross"
        elif f.startswith("t_"):
            g = "t_transaction"
        elif f.startswith("o_"):
            g = "o_order"
        elif "_ewm_" in f:
            g = "m_ewm"
        elif "_15" in f or "_45" in f or "_60" in f or "_120" in f or "_180" in f:
            g = "m_window"
        else:
            g = "m_book"
        groups.setdefault(g, 0.0)
        groups[g] += float(imp[i])
    tot = sum(groups.values())
    print(f"\n=== {name} ===")
    print(f"AUC = {a:.4f}  (n_train={len(Xa):,} n_test={len(Xb):,})")
    print(f"TOP20 特征 (gain):")
    for f, v in top:
        print(f"  {f}: {v:.1f}")
    print("特征组漂移 (gain占比):")
    for g, v in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"  {g}: {v/tot*100:.1f}%")
    return a, dict(groups), top

# 三组
m0_50 = m_tr <= 50
m51_70 = (m_tr > 50) & (m_tr <= 70)
rA = adv_run("A: m0-50 vs m51-70 (train内部漂移)", X_tr[m0_50], X_tr[m51_70])
rB = adv_run("B: m51-70 vs test (近期 vs test)", X_tr[m51_70], X_te)
rC = adv_run("C: full train vs test", X_tr, X_te)

print("\n\n========== P0-2 汇总 ==========")
print(f"A (train内部): AUC={rA[0]:.4f}")
print(f"B (近期vs test): AUC={rB[0]:.4f}")
print(f"C (全train vs test): AUC={rC[0]:.4f}")
if rC[0] > 0.7:
    print("→ 存在明显 covariate drift; 高漂移特征需识别")
else:
    print("→ train/test 特征分布接近 (AUC<=0.7)")

# 特征组四象限: C 组漂移占比 vs 预测重要性 (预测重要性用 gain 汇总, 来自之前的实验)
print("\n特征组漂移 vs 预测价值 (C组 gain 占比):")
for g, v in sorted(rC[1].items(), key=lambda x: -x[1]):
    print(f"  {g}: {v/sum(rC[1].values())*100:.1f}%")
