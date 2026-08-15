# -*- coding: utf-8 -*-
"""P7-AMP 快速版 (2026-08-15): Market Amplitude Gate Probe
冻结 baseline (canonical OOF 0.1351) → shallow LGB amplitude gate → α calibration (嵌套) → frozen 评估

臂:
  A: baseline 原样 (frozen 基准)
  C: baseline × gate(不含 mid_range)   — 控制单纯 amplitude calibration
  D: baseline × gate(含 mid_range)     — 测试 mid_range 独立价值 (主结果: D−C)

嵌套纪律: gate 拟合 + α 选择全部只在 calibration (m21-50), frozen 评估 m51-70
门禁: GREEN ≥ +0.0015 / YELLOW +0.0005~0.0015 / RED < +0.0005 (ΔD vs A, frozen)
"""
import polars as pl
import numpy as np
import lightgbm as lgb
import json, time

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)

df = pl.read_parquet(r"D:\mscapital-kaggle\output\p7amp_audit.parquet")
log(f"loaded: {len(df)} rows, months {df['month'].min()}-{df['month'].max()}")

def cosine(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# 切分: calibration m21-50 / frozen eval m51-70
cal = df.filter(pl.col("month") <= 50)
evl = df.filter(pl.col("month") > 50)
y_c, p_c = cal["y"].to_numpy(), cal["p"].to_numpy()
y_e, p_e = evl["y"].to_numpy(), evl["p"].to_numpy()
log(f"cal: {len(cal)} rows (m21-50) | eval: {len(evl)} rows (m51-70)")

cos_A_cal = cosine(p_c, y_c)
cos_A_eval = cosine(p_e, y_e)
log(f"arm A baseline: cal={cos_A_cal:.5f}  frozen={cos_A_eval:.5f}")

# amplitude target: z = log(|y| + eps)
eps = 1e-6
z_c = np.log(np.abs(y_c) + eps)
z_e = np.log(np.abs(y_e) + eps)

FEATS_C = ["mid_std", "n_order", "n_tx", "depth1"]           # 不含 mid_range
FEATS_D = ["mid_range", "mid_std", "n_order", "n_tx", "depth1"]  # 含 mid_range
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

def fit_gate(feats, X_cal, X_eval):
    m = lgb.LGBMRegressor(
        max_depth=3, num_leaves=8, n_estimators=200,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1)
    m.fit(X_cal, z_c)
    g_c = np.exp(m.predict(X_cal))
    g_e = np.exp(m.predict(X_eval))
    return m, g_c, g_e

results = {}
for arm, feats in [("C", FEATS_C), ("D", FEATS_D)]:
    X_cal = cal.select(feats).to_numpy()
    X_eval = evl.select(feats).to_numpy()
    m, g_c, g_e = fit_gate(feats, X_cal, X_eval)
    # α 在 calibration 选一次 (嵌套)
    best_a, best_cos = None, -1
    for a in ALPHAS:
        p_new = p_c * (g_c / np.median(g_c)) ** a
        c = cosine(p_new, y_c)
        if c > best_cos:
            best_cos, best_a = c, a
    # frozen 评估 (选定的 α 固定)
    p_new_e = p_e * (g_e / np.median(g_e)) ** best_a
    cos_eval = cosine(p_new_e, y_e)
    # 诊断: gate 对幅度校准的改善
    corr_g_y = float(np.corrcoef(g_e, np.abs(y_e))[0, 1])
    corr_p_y = float(np.corrcoef(np.abs(p_e), np.abs(y_e))[0, 1])
    dA = cos_eval - cos_A_eval
    results[arm] = {"alpha": best_a, "cos_cal": best_cos, "cos_frozen": cos_eval,
                    "dA_frozen": dA, "corr_g_absy": corr_g_y, "corr_abs_p_absy": corr_p_y,
                    "gate_median": float(np.median(g_e))}
    log(f"arm {arm}: α={best_a} cal_cos={best_cos:.5f} frozen={cos_eval:.5f} "
        f"ΔvsA={dA:+.5f} corr(g,|y|)={corr_g_y:+.3f}")

dC = results["C"]["dA_frozen"]
dD = results["D"]["dA_frozen"]
dDC = results["D"]["cos_frozen"] - results["C"]["cos_frozen"]
log("="*70)
log(f"主结果: ΔD(vs A) = {dD:+.5f}   ΔC(vs A) = {dC:+.5f}   mid_range 独立价值 Δ(D−C) = {dDC:+.5f}")
if dD >= 0.0015:
    verdict = "GREEN (≥+0.0015): Market 幅度条件化有效, 进入生产候选"
elif dD >= 0.0005:
    verdict = "YELLOW (+0.0005~0.0015): 保留为 ensemble/calibration component, 不深挖"
else:
    verdict = "RED (<+0.0005): baseline 已隐式吸收 heteroskedasticity, 关闭 market amplitude 路线"
log(f"门禁判定: {verdict}")

out = {
    "cos_A_frozen": cos_A_eval, "cos_A_cal": cos_A_cal,
    "arms": results, "dD_vs_A": dD, "dC_vs_A": dC, "d_D_minus_C": dDC,
    "verdict": verdict, "cal_months": "21-50", "eval_months": "51-70",
}
with open(r"D:\mscapital-kaggle\output\p7amp_quick.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
log("saved -> output/p7amp_quick.json")
