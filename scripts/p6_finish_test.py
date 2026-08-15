# -*- coding: utf-8 -*-
"""P6-FINISH: 完成 test 预测剩余步骤 (复用 realmlpC_test_pred.npz).

1. 分布检查: test std vs R61_70 valid std (spot-check outer preds)
2. 生产 blend 权重: 用 spot-check C 的 inner_tune (51-60) vs canonical 51-60 调权
   (网格 0.05-0.75, 弥补 P5-E 的 0.50 顶格问题)
3. PSEUDO 门禁: RealMLP-C 在 PSEUDO 切分 (inner 0-20/tune 21-32/refit 0-32/eval 33-70)
   → cosine vs canonical PSEUDO + blendΔ (w 调于 21-32)
4. 生产 blend: v8b + RealMLP-C test → 候选 CSV (w 网格 0.3-0.7) + 主候选
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd

sys.path.insert(0, r"D:\mscapital-kaggle\src")
from mscapital.models.realmlp import (CleanRealMLPPreprocessor, RealMLPConfig,
                                      _train_refit_predict, train_inner)

OUT = Path(r"D:\mscapital-kaggle\output\p6_prod")
LABEL = r"D:\mscapital-forecasting\data\raw\train\label.feather"
CFG = r"D:\mscapital-kaggle\configs\clean-realmlp-v2a.json"
SPOT_C = r"D:\mscapital-kaggle\output\p5b_scfi\spot_rmlp_C.parquet"
SPOT_INNER = r"D:\mscapital-kaggle\output\p5e_realmlp_spotcheck\C\R61_70\inner_predictions.npz"
SPOT_OUTER = r"D:\mscapital-kaggle\output\p5e_realmlp_spotcheck\C\R61_70\predictions.npz"
CANON_OOF = r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz"
CANON_PSEUDO = r"D:\mscapital-kaggle\output\c4_protocol_closed_final\clean-baseline-v2\PSEUDO\predictions.npz"
V8B = r"D:\mscapital-kaggle\output\submissions\submission_v8_ref50.csv"
TEST_PRED = r"D:\mscapital-kaggle\output\p6_prod\realmlpC_test_pred.npz"


def cosine(p, y):
    return float(np.dot(p, y) / np.sqrt(np.dot(p, p) * np.dot(y, y)))


def rms_norm(p):
    s = float(np.sqrt(np.mean(p ** 2)))
    return p / s if s > 0 else p


def main() -> None:
    t0 = time.time()
    cfg = RealMLPConfig.from_mapping(json.loads(Path(CFG).read_text(encoding="utf-8")))
    res = {}

    # ---------- test preds + 分布检查 ----------
    tp = np.load(TEST_PRED)
    test_ids, p_test = tp["sample_id"], tp["pred"]
    sp_o = np.load(SPOT_OUTER)
    std_valid = float(np.std(sp_o["pred"]))
    std_test = float(np.std(p_test))
    res["distribution"] = {"std_test": std_test, "std_valid_61_70": std_valid,
                           "ratio": std_test / std_valid}
    print(f"分布: std_test={std_test:.5f} std_valid={std_valid:.5f} ratio={std_test/std_valid:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---------- 生产 blend 权重: spot-check 51-60 调权 (网格扩到 0.75) ----------
    sp_i = np.load(SPOT_INNER)
    co = np.load(CANON_OOF)
    c_ids, c_month, c_base = co["sample_id"], co["month"].astype(int), co["baseline_oof"]
    i51 = (c_month >= 51) & (c_month <= 60)
    c_ids51, c_base51 = c_ids[i51], c_base[i51]
    # 对齐 spot inner preds (sample_id 同序)
    p51 = sp_i["pred"]
    p51_ids = sp_i["sample_id"]
    pos51 = np.searchsorted(c_ids51, p51_ids)
    canon51 = c_base51[pos51]
    y51 = sp_i["target"]
    best_w, best_s = 0.0, -1e9
    for w in np.arange(0.05, 0.76, 0.05):
        s = cosine(rms_norm(canon51) + w * rms_norm(p51), y51)
        if s > best_s:
            best_s, best_w = s, float(w)
    res["blend"] = {"w_prod_51_60": best_w, "score_51_60": best_s}
    print(f"生产 blend w (51-60 调): {best_w:.2f} (score {best_s:.6f})", flush=True)

    # ---------- PSEUDO 门禁 (inner 0-20/tune 21-32/refit 0-32/eval 33-70) ----------
    lab = pl.read_ipc(LABEL).sort("sample_id")
    m_all = lab["month"].to_numpy()
    sp = pl.read_parquet(SPOT_C).sort("sample_id")
    znames = [c for c in sp.columns if c.startswith("Z_")]
    fe152 = [c for c in sp.columns if c not in ("sample_id",) + tuple(znames)]
    XC_all = sp.select(fe152 + znames).to_numpy().astype(np.float32)
    y_all = lab["target"].to_numpy().astype(np.float64)
    df_all = pd.DataFrame(XC_all, columns=fe152 + znames)
    print(f"PSEUDO 数据就绪 ({time.time()-t0:.0f}s)", flush=True)

    i_tr = (m_all >= 0) & (m_all <= 20)
    i_tu = (m_all >= 21) & (m_all <= 32)
    rf = (m_all >= 0) & (m_all <= 32)
    ev = (m_all >= 33) & (m_all <= 70)

    pre_p = CleanRealMLPPreprocessor(tuple(fe152 + znames), cfg).fit(df_all.loc[i_tr], y_all[i_tr])
    x_it, c_it = pre_p.transform(df_all.loc[i_tr])
    x_tu, c_tu = pre_p.transform(df_all.loc[i_tu])
    inner = train_inner(x_it, c_it, np.round(y_all[i_tr], cfg.target_round),
                        x_tu, c_tu, y_all[i_tu], cfg)
    print(f"PSEUDO inner done (best_progress={inner.best_progress:.3f}, {time.time()-t0:.0f}s)", flush=True)
    del x_it, c_it, x_tu, c_tu
    gc.collect()

    pre_r = CleanRealMLPPreprocessor(tuple(fe152 + znames), cfg).fit(df_all.loc[rf], y_all[rf])
    x_rf, c_rf = pre_r.transform(df_all.loc[rf])
    x_ev, c_ev = pre_r.transform(df_all.loc[ev])
    p_ev, _, _, _, _ = _train_refit_predict(
        x_rf, c_rf, np.round(y_all[rf], cfg.target_round), x_ev, c_ev, inner.best_progress, cfg)
    y_ev = y_all[ev]
    cos_pseudo = cosine(p_ev, y_ev)
    np.savez(OUT / "realmlpC_pseudo_pred.npz",
             sample_id=lab["sample_id"].to_numpy()[ev], pred=p_ev, target=y_ev)

    canon_p = np.load(CANON_PSEUDO)
    c_ids2, c_pred2, c_y2 = canon_p["sample_id"], canon_p["pred"], canon_p["target"]
    i_ev_ids = lab["sample_id"].to_numpy()[ev]
    cpos = np.searchsorted(c_ids2, i_ev_ids)
    canon_ev = c_pred2[cpos]
    canon_y = c_y2[cpos]
    assert np.array_equal(canon_y, y_ev), "canonical PSEUDO target mismatch"
    cos_canon = cosine(canon_ev, y_ev)
    i_tu_ids = lab["sample_id"].to_numpy()[i_tu]
    cpos_tu = np.searchsorted(c_ids2, i_tu_ids)
    canon_tu = c_pred2[cpos_tu]
    p_tu = inner.predictions
    best_wp, best_sp = 0.0, -1e9
    for w in np.arange(0.05, 0.76, 0.05):
        s = cosine(rms_norm(canon_tu) + w * rms_norm(p_tu), y_all[i_tu])
        if s > best_sp:
            best_sp, best_wp = s, float(w)
    p_blend = rms_norm(canon_ev) + best_wp * rms_norm(p_ev)
    blend_delta = cosine(p_blend, y_ev) - cos_canon
    res["pseudo"] = {"realmlpC": cos_pseudo, "canonical": cos_canon,
                     "blend_delta": blend_delta, "blend_w": best_wp}
    print(f"\nPSEUDO gate: RealMLP-C={cos_pseudo:.6f} canon={cos_canon:.6f} "
          f"Δ={cos_pseudo - cos_canon:+.6f} blendΔ={blend_delta:+.6f} (w={best_wp:.2f}) "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---------- 生产 blend: v8b + RealMLP-C test ----------
    v8b = pl.read_csv(V8B).sort("sample_id")
    p_v8b = v8b["prediction"].to_numpy()
    assert len(p_v8b) == len(test_ids)
    w_prod = best_w  # 51-60 调的 w
    cands = {}
    for w in (0.30, 0.40, 0.50, 0.60, 0.70, w_prod):
        p_c = rms_norm(p_v8b) + w * rms_norm(p_test)
        cands[f"w{w:.2f}"] = {"w": float(w)}
    p_main = rms_norm(p_v8b) + w_prod * rms_norm(p_test)
    np.savez(OUT / "blend_candidates.npz",
             sample_id=test_ids, p_v8b=p_v8b, p_realmlpC=p_test, p_main=p_main)
    pl.DataFrame({"sample_id": test_ids, "prediction": p_main}).write_csv(
        OUT / "submission_candidate_p6.csv")
    res["candidates"] = {k: v for k, v in cands.items()}
    res["main_candidate"] = f"submission_candidate_p6.csv (v8b + {w_prod:.2f}*RealMLP-C)"

    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print("\n" + "=" * 70)
    print(f"PSEUDO: RealMLP-C {cos_pseudo:.6f} vs canon {cos_canon:.6f} (blendΔ {blend_delta:+.6f})")
    print(f"分布: std_test/std_valid = {std_test/std_valid:.4f}")
    print(f"提交候选: submission_candidate_p6.csv (v8b + {w_prod:.2f}*RealMLP-C) — 未提交")
    print("=" * 70)
    print(f"total {time.time()-t0:.0f}s → {OUT}")


if __name__ == "__main__":
    main()
