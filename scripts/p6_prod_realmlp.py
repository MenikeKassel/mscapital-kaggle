# -*- coding: utf-8 -*-
"""P6-PROD: RealMLP-C (152+Z) 生产推理 + PSEUDO 门禁 + blend 提交候选。

步骤:
1. Z_test: nuisance (M→O, M+O→T) fit on train 0-70 → predict test (M=f0726 子集,
   O/T=raw_ot_agg_test, scale=MAD(train 折外残差))
2. RealMLP-C test: preprocessor fit on train 0-70 (225 feats) → refit (progress=1.0)
   → predict test
3. PSEUDO 门禁: 同 pipeline 在 PSEUDO 切分 (inner 0-20 / tune 21-32 / refit 0-32 /
   eval 33-70), 与 canonical PSEUDO 对比 + blend (w 在 21-32 调)
4. 生产 blend: v8b 锚 (submission_v8_ref50) + RealMLP-C test, w 网格 0.05-0.70
   (参考 spot-check: w=0.50 顶格), 分布检查 (test std vs 51-70 valid std)
5. 提交候选 CSV 落盘 (不自动提交 Kaggle)

Z 列名与 p5b_scfi FE_RAW 完全一致 (Z_{raw_name}).
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, r"D:\mscapital-kaggle\src")
sys.path.insert(0, r"D:\mscapital-kaggle\scripts")

import p5b_scfi as P
from mscapital.models.realmlp import (CleanRealMLPPreprocessor, RealMLPConfig,
                                      _train_refit_predict, train_inner)

OUT = Path(r"D:\mscapital-kaggle\output\p6_prod")
LABEL = r"D:\mscapital-forecasting\data\raw\train\label.feather"
CFG = r"D:\mscapital-kaggle\configs\clean-realmlp-v2a.json"
SPOT_C = r"D:\mscapital-kaggle\output\p5b_scfi\spot_rmlp_C.parquet"  # train 152+Z (0-70)
F726_TEST = r"D:\mscapital-forecasting\data\processed\f0726_test_f32.parquet"
RAW_TEST = r"D:\mscapital-kaggle\output\p6_prod\raw_ot_agg_test.parquet"
V8B = r"D:\mscapital-kaggle\output\submissions\submission_v8_ref50.csv"
CANON_PSEUDO = r"D:\mscapital-kaggle\output\c4_protocol_closed_final\clean-baseline-v2\PSEUDO\predictions.npz"


def cosine(p, y):
    return float(np.dot(p, y) / np.sqrt(np.dot(p, p) * np.dot(y, y)))


def rms_norm(p):
    s = float(np.sqrt(np.mean(p ** 2)))
    return p / s if s > 0 else p


def mad(arr):
    return float(np.median(np.abs(arr - np.median(arr)))) + 1e-9


def main() -> None:
    t0 = time.time()
    cfg = RealMLPConfig.from_mapping(json.loads(Path(CFG).read_text(encoding="utf-8")))

    # ---------- 数据 ----------
    lab = pl.read_ipc(LABEL).sort("sample_id")
    m_all = lab["month"].to_numpy()
    f726 = pl.read_parquet(r"D:\mscapital-forecasting\data\processed\f0726_train_f32.parquet").sort("sample_id")
    fe152 = [c for c in f726.columns if c not in ("sample_id", "target")]

    # 训练 Z (复用 spot_rmlp_C 的 225 列) + train 152
    sp = pl.read_parquet(SPOT_C).sort("sample_id")
    znames = [c for c in sp.columns if c.startswith("Z_")]
    train_ids = sp["sample_id"].to_numpy()
    assert len(train_ids) == len(lab) and np.array_equal(train_ids, lab["sample_id"].to_numpy())
    X152_tr = sp.select(fe152).to_numpy().astype(np.float32)
    Z_tr = sp.select(znames).to_numpy().astype(np.float32)
    y_tr = lab["target"].to_numpy().astype(np.float64)
    print(f"train: {len(train_ids):,} rows, 152+{len(znames)} feats ({time.time()-t0:.0f}s)", flush=True)

    # test 152 + raw O/T
    tf = pl.read_parquet(F726_TEST).sort("sample_id")
    test_ids = tf["sample_id"].to_numpy()
    X152_te = tf.select(fe152).to_numpy().astype(np.float32)
    raw_te = pl.read_parquet(RAW_TEST).sort("sample_id")
    raw_names = [c for c in raw_te.columns if c.startswith(("ob_", "tb_"))]
    O_te = raw_te.select([c for c in raw_names if c.startswith("ob_")]).to_numpy().astype(np.float64)
    T_te = raw_te.select([c for c in raw_names if c.startswith("tb_")]).to_numpy().astype(np.float64)
    # M test = f0726 子集 (与 m05 同源已验证)
    m05_feats = ["m_mid_std", "m_rv", "m_sp_mean_60", "m_imb_mean_60", "m_ofi_sum_60",
                 "m_mid_range_60", "m_mid_std_180", "m_imb_mean_180", "m_ofi_sum_180",
                 "t_buy_ratio_15", "t_px_std_15", "t_avg_time_gap", "o_buy_ratio_15",
                 "o_market_ratio_15", "o_avg_time_gap", "x_trans_order_buy_diff_15"]
    M_te = tf.select(m05_feats).to_numpy().astype(np.float64)
    print(f"test: {len(test_ids):,} rows ({time.time()-t0:.0f}s)", flush=True)

    # ---------- Z_test: nuisance fit on train 0-70 (全量) ----------
    data = P.load_data()  # train 侧 Xm/Xraw (1.26M)
    Xm_tr = data["Xm"]
    # f0726_test 含 ~2% NaN → M_te 用 train 列中位数插补 (train 侧 Xm 已 fill_nan 0)
    med_m = np.median(Xm_tr, axis=0)
    M_te = np.where(np.isnan(M_te), med_m, M_te)
    assert np.isfinite(M_te).all()
    Xraw_tr = data["Xraw"]
    o_cols = [i for i, c in enumerate(data["feraw"]) if c.startswith("ob_")]
    t_cols = [i for i, c in enumerate(data["feraw"]) if c.startswith("tb_")]
    O_tr, T_tr = Xraw_tr[:, o_cols], Xraw_tr[:, t_cols]
    # 折外残差 MAD (train 内 10 月折, 同 P5-B 协议)
    fold_starts = np.arange(0, 71, 10)
    resid_O = np.full((len(Xm_tr), O_tr.shape[1]), np.nan)
    resid_T = np.full((len(Xm_tr), T_tr.shape[1]), np.nan)
    for (s, e) in [(s, min(s + 9, 70)) for s in fold_starts]:
        f_mask = (m_all >= s) & (m_all <= e)
        o_mask = ~f_mask
        Xf = Xm_tr[o_mask]
        Xv = Xm_tr[f_mask]
        sc = StandardScaler().fit(Xf)
        r = Ridge(alpha=1.0).fit(sc.transform(Xf), np.clip(O_tr[o_mask], -6 * mad(O_tr.ravel()), 6 * mad(O_tr.ravel())))
        resid_O[f_mask] = O_tr[f_mask] - r.predict(sc.transform(Xv))
        Xfo = np.hstack([Xf, O_tr[o_mask]])
        Xvo = np.hstack([Xv, O_tr[f_mask]])
        sc2 = StandardScaler().fit(Xfo)
        r2 = Ridge(alpha=1.0).fit(sc2.transform(Xfo), np.clip(T_tr[o_mask], -6 * mad(T_tr.ravel()), 6 * mad(T_tr.ravel())))
        resid_T[f_mask] = T_tr[f_mask] - r2.predict(sc2.transform(Xvo))
    madO = np.array([mad(resid_O[:, j]) for j in range(O_tr.shape[1])])
    madT = np.array([mad(resid_T[:, j]) for j in range(T_tr.shape[1])])
    # fit on ALL train → predict test
    sc = StandardScaler().fit(Xm_tr)
    rO = Ridge(alpha=1.0).fit(sc.transform(Xm_tr), np.clip(O_tr, -6 * mad(O_tr.ravel()), 6 * mad(O_tr.ravel())))
    Z_O_te = (O_te - rO.predict(sc.transform(M_te))) / madO
    Xfo_all = np.hstack([Xm_tr, O_tr])
    Xfo_te = np.hstack([M_te, O_te])
    sc2 = StandardScaler().fit(Xfo_all)
    rT = Ridge(alpha=1.0).fit(sc2.transform(Xfo_all), np.clip(T_tr, -6 * mad(T_tr.ravel()), 6 * mad(T_tr.ravel())))
    Z_T_te = (T_te - rT.predict(sc2.transform(Xfo_te))) / madT
    Z_te = np.hstack([Z_O_te, Z_T_te]).astype(np.float32)
    assert np.isfinite(Z_te).all()
    print(f"Z_test done ({time.time()-t0:.0f}s)", flush=True)

    # ---------- RealMLP-C: preprocessor + refit (0-70) → test ----------
    XC_tr = np.hstack([X152_tr, Z_tr])
    XC_te = np.hstack([X152_te, Z_te])
    pre = CleanRealMLPPreprocessor(tuple(fe152 + znames), cfg).fit(
        __import__("pandas").DataFrame(XC_tr, columns=fe152 + znames), y_tr)
    x_tr, c_tr = pre.transform(__import__("pandas").DataFrame(XC_tr, columns=fe152 + znames))
    x_te, c_te = pre.transform(__import__("pandas").DataFrame(XC_te, columns=fe152 + znames))
    del XC_tr, XC_te
    gc.collect()
    p_test, hist, step, prog, rq = _train_refit_predict(
        x_tr, c_tr, np.round(y_tr, cfg.target_round) if cfg.target_round else y_tr,
        x_te, c_te, 1.0, cfg)
    print(f"RealMLP-C test preds done ({time.time()-t0:.0f}s), prog={prog:.3f}", flush=True)
    np.savez(OUT / "realmlpC_test_pred.npz", sample_id=test_ids, pred=p_test)

    # ---------- PSEUDO 门禁 (inner 0-20 / tune 21-32 / refit 0-32 / eval 33-70) ----------
    m_tr = m_all
    i_tr = (m_tr >= 0) & (m_tr <= 20)
    i_tu = (m_tr >= 21) & (m_tr <= 32)
    rf = (m_tr >= 0) & (m_tr <= 32)
    ev = (m_tr >= 33) & (m_tr <= 70)
    XC_all = np.hstack([X152_tr, Z_tr])
    df_all = __import__("pandas").DataFrame(XC_all, columns=fe152 + znames)
    y_all = y_tr
    pre_p = CleanRealMLPPreprocessor(tuple(fe152 + znames), cfg).fit(df_all.loc[i_tr], y_all[i_tr])
    x_it, c_it = pre_p.transform(df_all.loc[i_tr])
    x_tu, c_tu = pre_p.transform(df_all.loc[i_tu])
    inner = train_inner(x_it, c_it, np.round(y_all[i_tr], cfg.target_round),
                        x_tu, c_tu, y_all[i_tu], cfg)
    pre_r = CleanRealMLPPreprocessor(tuple(fe152 + znames), cfg).fit(df_all.loc[rf], y_all[rf])
    x_rf, c_rf = pre_r.transform(df_all.loc[rf])
    x_ev, c_ev = pre_r.transform(df_all.loc[ev])
    p_ev, _, _, _, _ = _train_refit_predict(
        x_rf, c_rf, np.round(y_all[rf], cfg.target_round), x_ev, c_ev, inner.best_progress, cfg)
    y_ev = y_all[ev]
    cos_pseudo = cosine(p_ev, y_ev)
    # canonical PSEUDO 对比 + blend (w 在 21-32 调)
    canon_p = np.load(CANON_PSEUDO)
    c_ids, c_pred, c_y = canon_p["sample_id"], canon_p["pred"], canon_p["target"]
    # canonical PSEUDO preds 与 p_ev 用 sample_id 对齐
    i_ev_ids = lab["sample_id"].to_numpy()[ev]
    cpos = np.searchsorted(c_ids, i_ev_ids)
    canon_ev = c_pred[cpos]
    canon_y = c_y[cpos]
    assert np.array_equal(canon_y, y_ev), "canonical PSEUDO target mismatch"
    cos_canon = cosine(canon_ev, y_ev)
    # blend w tuned on 21-32 (inner tune preds vs canonical 21-32)
    i_tu_ids = lab["sample_id"].to_numpy()[i_tu]
    cpos_tu = np.searchsorted(c_ids, i_tu_ids)
    canon_tu = c_pred[cpos_tu]
    p_tu = inner.predictions
    best_w, best_s = 0.0, -1e9
    for w in np.arange(0.05, 0.71, 0.05):
        s = cosine(rms_norm(canon_tu) + w * rms_norm(p_tu), y_all[i_tu])
        if s > best_s:
            best_s, best_w = s, float(w)
    p_blend = rms_norm(canon_ev) + best_w * rms_norm(p_ev)
    blend_delta = cosine(p_blend, y_ev) - cos_canon
    print(f"\nPSEUDO gate: RealMLP-C={cos_pseudo:.6f} canon={cos_canon:.6f} "
          f"blendΔ={blend_delta:+.6f} (w={best_w:.2f} tuned 21-32)", flush=True)

    # ---------- 生产 blend: v8b + RealMLP-C (w 网格, 参考 51-60 调权) ----------
    v8b = pl.read_csv(V8B).sort("sample_id")
    p_v8b = v8b["prediction"].to_numpy()
    assert len(p_v8b) == len(test_ids)
    # 分布检查: RealMLP-C test std vs R61_70 valid std
    sp61 = np.load(r"D:\mscapital-kaggle\output\p5e_realmlp_spotcheck\C\R61_70\predictions.npz")
    std_valid = float(np.std(sp61["pred"]))
    std_test = float(np.std(p_test))
    ratio = std_test / std_valid
    # w: 沿用 PSEUDO tune 的 best_w 会跨域; 用 spot-check 51-60 的 w=0.50 基准 + 网格敏感性
    cands = {}
    for w in (0.30, 0.40, 0.50, 0.60, 0.70):
        p_c = rms_norm(p_v8b) + w * rms_norm(p_test)
        cands[f"w{w:.2f}"] = {"w": w}
    # 主候选: w=0.50 (spot-check 51-60 调权结果)
    p_main = rms_norm(p_v8b) + 0.50 * rms_norm(p_test)
    np.savez(OUT / "blend_candidates.npz",
             sample_id=test_ids, p_v8b=p_v8b, p_realmlpC=p_test,
             p_main=p_main)
    pl.DataFrame({"sample_id": test_ids, "prediction": p_main}).write_csv(OUT / "submission_candidate_p6.csv")

    res = {
        "pseudo": {"realmlpC": cos_pseudo, "canonical": cos_canon,
                   "blend_delta": blend_delta, "blend_w": best_w},
        "distribution": {"std_test": std_test, "std_valid_61_70": std_valid,
                         "ratio_test_valid": ratio},
        "candidates": {k: {"w": v["w"]} for k, v in cands.items()},
        "main_candidate": "submission_candidate_p6.csv (v8b + 0.50*RealMLP-C)",
    }
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print("\n" + "=" * 70)
    print(f"PSEUDO: RealMLP-C {cos_pseudo:.6f} vs canon {cos_canon:.6f} "
          f"(blendΔ {blend_delta:+.6f})")
    print(f"分布: std_test/std_valid = {ratio:.4f} (valid {std_valid:.5f} / test {std_test:.5f})")
    print(f"候选: output/p6_prod/submission_candidate_p6.csv (未提交)")
    print("=" * 70)
    print(f"total {time.time()-t0:.0f}s → {OUT}")


if __name__ == "__main__":
    main()
