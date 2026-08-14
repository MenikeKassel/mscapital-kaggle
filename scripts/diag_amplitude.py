# -*- coding: utf-8 -*-
"""Zero-cost diagnostics: does v7's magnitude carry |y| information?

D1: corr(|v7|, |y|) — is v7 magnitude already calibrated to target scale?
D2: v7 direction accuracy by |y| quintile — do large-|y| samples have >= average
    direction skill (the premise of magnitude modulation on uncentered cosine)?
D3: per-quintile corr(v7, y) — where does v7's cosine contribution concentrate?
"""
import numpy as np
import polars as pl

lab = pl.read_ipc(r"D:\mscapital-forecasting\data\raw\train\label.feather")
m_all = lab["month"].to_numpy()
sid_all = lab["sample_id"].to_numpy()
y_all = lab["target"].to_numpy()

# v7_like = 0.8*v5 + 0.2*RL (same as P5-01/P5-02I)
rl = np.load(r"D:\mscapital-kaggle\output\rlps_final\realmlp_pseudo_pred.npz")["pred"]
v5 = np.load(r"D:\mscapital-kaggle\output\rlps_v12\v5_table_pseudo_pred.npz")["pred"]
fe_all = pl.read_parquet(r"D:\mscapital-forecasting\data\processed\f0726_train_f32.parquet").sort("sample_id")
m3340 = (m_all >= 33) & (m_all <= 70)
rl_ids = fe_all["sample_id"].to_numpy()[m3340]
rp = np.searchsorted(rl_ids, sid_all[m3340])
v7 = np.full(len(y_all), np.nan)
v7[m3340] = 0.8 * v5[rp] + 0.2 * rl[rp]

mask = (m_all >= 41) & (m_all <= 70) & np.isfinite(v7)
y, p = y_all[mask], v7[mask]
n = len(y)
print(f"n={n:,}  months 41-70")

print("\n[D1] magnitude calibration")
print(f"  corr(|v7|, |y|) = {np.corrcoef(np.abs(p), np.abs(y))[0,1]:+.4f}")
print(f"  corr(v7, y)     = {np.corrcoef(p, y)[0,1]:+.4f}  (reference)")
print(f"  corr(|v7|, |y|^2)= {np.corrcoef(np.abs(p), y**2)[0,1]:+.4f}")
print(f"  std(|y|)/std(y) = {np.std(np.abs(y))/np.std(y):.4f}  (scale ratio)")

print("\n[D2] v7 direction accuracy by |y| quintile")
q = np.quantile(np.abs(y), [0.2, 0.4, 0.6, 0.8])
lbl = ["q1(lo)", "q2", "q3", "q4", "q5(hi)"]
acc = []
for i, lo_ in enumerate([0.0, *q]):
    hi_ = q[i] if i < 4 else np.inf
    m = (np.abs(y) >= lo_) & (np.abs(y) <= hi_)
    acc.append((m.sum(), float(((p[m] * y[m]) > 0).mean())))
for (cnt, a), name in zip(acc, lbl):
    print(f"  {name:<8} n={cnt:>7,}  sign-acc={a:.4f}")
print(f"  overall sign-acc = {((p*y)>0).mean():.4f}")

print("\n[D3] per-quintile corr(v7, y) — where cosine contribution concentrates")
for i, lo_ in enumerate([0.0, *q]):
    hi_ = q[i] if i < 4 else np.inf
    m = (np.abs(y) >= lo_) & (np.abs(y) <= hi_)
    c = np.corrcoef(p[m], y[m])[0, 1]
    # share of total <p,y> inner product per quintile
    share = float((p[m] * y[m]).sum()) / float((p * y).sum())
    print(f"  {lbl[i]:<8} n={m.sum():>7,}  corr={c:+.4f}  inner-prod share={share:.2%}")

print("\n[D2b] extreme samples (|y| > p90)")
thr = np.quantile(np.abs(y), 0.90)
me = np.abs(y) > thr
print(f"  n={me.sum():,}  sign-acc={((p[me]*y[me])>0).mean():.4f}  corr={np.corrcoef(p[me],y[me])[0,1]:+.4f}")
print(f"  inner-prod share={float((p[me]*y[me]).sum())/float((p*y).sum()):.2%}")
