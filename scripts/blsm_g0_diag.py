# -*- coding: utf-8 -*-
"""BLSM-G0 诊断: 行为特征是否独立于现有状态?
回答 brief 的 6 问 (被 month/activity/volatility/order_count 主导? latent 稳定?)
方法: 1) 行为特征 PCA 降维 2) 检验 PC 是否可用现有 152 特征/月/活动度预测 (R2/AUC)
     → 若 PC 高度可被现有代理解释 => FAIL (只是重新编码)
     → 若 PC 有独立方差 => EXIST (行为状态可能真存在)
     + 行为特征本身与 target 的 IC 快照 (不训练, 只看方向)
"""
import numpy as np, polars as pl, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

# ---------- 载入 ----------
blsm = pl.read_parquet(r"D:\mscapital-forecasting\data\processed\blsm_g0_train.parquet")
f726 = pl.read_parquet(r"D:\mscapital-forecasting\data\processed\f0726_train_f32.parquet")
lab = pl.read_ipc(r"D:\mscapital-forecasting\data\raw\train\label.feather")
# 统一 sample_id dtype
blsm = blsm.with_columns(pl.col("sample_id").cast(pl.Int32))
f726 = f726.with_columns(pl.col("sample_id").cast(pl.Int32))
lab = lab.with_columns(pl.col("sample_id").cast(pl.Int32))
beh = blsm.select([c for c in blsm.columns if c != "sample_id"]).to_pandas()

# activity/vol proxy 从 152 取 (现有基线代理)
proxy = f726.select(["t_cnt" if "t_cnt" in f726.columns else "t_sec_rowcount_near_far_ratio_60",
                     "m_mid_std","m_rv","m_imb_mean_60","o_n_45" if "o_n_45" in f726.columns else "o_sec_row_count_15",
                     "sample_id"]).to_pandas()
# 补 order_count / txn_count 直接算
ord_ = pl.scan_ipc(r"D:\mscapital-forecasting\data\raw\train\order.feather", memory_map=False)\
    .filter(pl.col("seconds_before_predict") <= 60.0)\
    .group_by("sample_id").agg(pl.len().alias("order_count")).collect()
tx_ = pl.scan_ipc(r"D:\mscapital-forecasting\data\raw\train\transaction.feather", memory_map=False)\
    .filter(pl.col("seconds_before_predict") <= 60.0)\
    .group_by("sample_id").agg(pl.len().alias("txn_count")).collect()

m_df = lab.join(blsm, on="sample_id", how="left").join(ord_, on="sample_id", how="left").join(tx_, on="sample_id", how="left")
m_df = m_df.join(f726.select(["sample_id","m_mid_std","m_rv"]), on="sample_id", how="left")
m_df = m_df.to_pandas()

# ---------- 行为特征标准化 + PCA ----------
feat_cols = [c for c in blsm.columns if c.startswith(("b5_","b6_","m_depth_mean","m_spread_mean","m_shock_txvol","m_txvol_std","m_bidvol_std","m_askvol_std")) or c in ("t_signed_imb","o_ofi_norm","o_add_imb","o_cancel_imb")]
feat_cols = [c for c in feat_cols if c in blsm.columns]
X = blsm.select(feat_cols).to_pandas().fillna(0.0)
# 处理 inf
X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_std = StandardScaler().fit_transform(X)

pca = PCA(n_components=8, random_state=2026).fit(X_std)
print("=== PCA 解释方差 (行为特征 8PC) ===")
print("累计解释比:", np.round(np.cumsum(pca.explained_variance_ratio_), 4))
print("PC1 载荷 top5:", [feat_cols[i] for i in np.argsort(-np.abs(pca.components_[0]))[:5]])

# ---------- 问题1-6: PC 是否被现有代理/月/活动度预测? ----------
Z = pca.transform(X_std)  # (N,8)
pidx = m_df["sample_id"].to_numpy()
# 对齐 Z 行与 m_df 行顺序 (blsm 按 sample_id 全量, m_df 也是全量同序)
z_df = pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(8)])
z_df.insert(0, "sample_id", blsm["sample_id"].to_numpy())

# 从 152 提取活动度/波动代理列
proxies = m_df[["order_count","txn_count","m_mid_std","m_rv"]].fillna(m_df[["order_count","txn_count","m_mid_std","m_rv"]].median())
pm = m_df[["order_count","txn_count","m_mid_std","m_rv","month"]].fillna(0)

def redundancy_r2(pidx_arr, target_col, feats, pm_arr):
    """Ridge: 现有代理能解释 PC 多少 (pidx 对齐)."""
    from sklearn.model_selection import cross_val_score
    y = target_col
    Xp = pm_arr
    mask = np.isfinite(y)
    if mask.sum() < 1000: return float('nan')
    sc = cross_val_score(Ridge(alpha=1.0), Xp[mask], y[mask], cv=3, scoring="r2")
    return float(np.mean(sc))

# 对齐 m_df 与 Z (都按 blsm 的 sample_id 全量同序)
z_dft = pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(8)])
# m_df 行序 = lab(全量) join, blsm 也是全量同 sample_id, 直接 merge
m_df = m_df.merge(z_dft, how="left", left_index=True, right_index=True)
pm = m_df[["order_count","txn_count","m_mid_std","m_rv"]].fillna(m_df[["order_count","txn_count","m_mid_std","m_rv"]].median())
Xp = StandardScaler().fit_transform(pm.to_numpy())

print("\n=== R2: 现有代理(order_count/txn_count/vol)解释 PC 的能力 ===")
for k in range(4):
    r2 = redundancy_r2(None, m_df[f"PC{k+1}"].to_numpy(), None, Xp)
    print(f"  PC{k+1} 被 4代理解释 R2 = {r2:+.4f}" + ("  [高=冗余, FAIL信号]" if r2 > 0.3 else "  [低=独立]"))

# 与 month 的关联 (month 能预测 PC 吗) — 用 m_df (已含 PC, 与 Z 对齐)
from sklearn.linear_model import LogisticRegression
print("\n=== month 预测 PC: 若 PC 被 month 主导则 AUC 高 ===")
ym = (m_df["month"] >= 35).astype(int).to_numpy()
for k in range(3):
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    pcx = m_df[f"PC{k+1}"].to_numpy()
    mask = np.isfinite(pcx)
    yp = cross_val_predict(LogisticRegression(max_iter=500), pcx[mask].reshape(-1,1), ym[mask], cv=3, method="predict_proba")[:,1]
    print(f"  PC{k+1} 预测 month≥35 AUC = {roc_auc_score(ym[mask], yp):.3f}")

# ---------- 行为特征与 target 直接 IC (方向快照, 不训练) ----------
print("\n=== 行为特征 与 target 的 rank IC (未调权, 仅方向快照) ===")
from scipy.stats import spearmanr
# 严格对齐: 用 blsm 的 sample_id 顺序取 m_df 的 PC
blsm_ids = blsm["sample_id"].to_numpy()
tgt_map = dict(zip(lab["sample_id"].to_numpy(), lab["target"].to_numpy()))
yf = np.array([tgt_map[i] for i in blsm_ids], dtype=np.float64)
# m_df 的 PC 按 sample_id 建 map
pc_maps = {}
for k in range(4):
    pc_maps[k] = dict(zip(m_df["sample_id"].to_numpy(), m_df[f"PC{k+1}"].to_numpy()))
for k in range(4):
    pc = np.array([pc_maps[k].get(i, np.nan) for i in blsm_ids], dtype=np.float64)
    mask = np.isfinite(pc) & np.isfinite(yf)
    rho, p = spearmanr(pc[mask], yf[mask])
    print(f"  PC{k+1} rankIC = {rho:+.4f} (p={p:.2e})")

print("\n=== 结论判据 (per brief) ===")
print("若 PC 被 4 代理 R2>0.3 或 month AUC>0.6 且与 target IC 全 ~0 → FAIL (重新编码)")
print("若 PC 有独立方差 (代理 R2 低) 且个别 IC≠0 → EXIST (行为状态可能存在)")
