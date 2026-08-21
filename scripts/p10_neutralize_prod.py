# -*- coding: utf-8 -*-
"""P10-NEUT-PROD: neutralization 生产应用 (P9-NEUT 的 RQ 版).

1. 用 RQ C 臂 best.pt (152+73Z) 推理 PSEUDO eval 33-70 → 定标 γ (calibration 33-50 fit / 51-70 验证)
2. 加载 rq_scfi_test_pred.npz (test 预测) + test Z 特征 → 应用 γ → final submission 预测
Z 集 (与 P9-NEUT 一致): m_mid_std/m_mid_std_180/m_rv/m_rv_60/m_rv_180/m_sp_mean_60/o_vol_sum + |p|
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, r"D:\mscapital-kaggle\src")
sys.path.insert(0, r"D:\mscapital-kaggle\scripts")
from mscapital.models.realmlp import (
    RealMLPConfig, load_frame, CleanRealMLPPreprocessor,
    _build_torch_classes, RQKMeansEncoder, _predict, _EMA,
)
from c05_recipe_e0 import cosine_uncentered

RAW = Path(r"D:\mscapital-forecasting\data\raw\train")
DATA = Path(r"D:\mscapital-forecasting\data\processed")
OUT = Path(r"D:\mscapital-kaggle\output\p10_rq_prod_scfi")
# RQ C 臂模型在 p9_scfi_rq (PSEUDO m0-32 训练 / m33-70 eval 协议); p10 训练脚本不保存模型
MODEL_CKPT = Path(r"D:\mscapital-kaggle\output\p9_scfi_rq") / "arm_C" / "best.pt"

NUISANCE = ["m_mid_std", "m_mid_std_180", "m_rv", "m_rv_60", "m_rv_180",
            "m_sp_mean_60", "o_vol_sum", "t_vol_sum", "t_transaction_count", "o_order_count"]


def load_model_and_preds():
    """加载 C 臂 best.pt → 推理 PSEUDO eval 33-70 (定标用)."""
    cfg = RealMLPConfig(epochs=30)
    frame = load_frame(DATA / "f0726_train_z_f32.parquet", RAW / "label.feather")
    lab = pl.read_ipc(RAW / "label.feather").select(["sample_id", "month", "target"]).sort("sample_id")
    sid = np.asarray(frame.sample_id)
    lab_map = dict(zip(lab["sample_id"].to_numpy(), lab["month"].to_numpy()))
    y_map = dict(zip(lab["sample_id"].to_numpy(), lab["target"].to_numpy()))
    m = np.asarray([lab_map[s] for s in sid])
    y = np.asarray([y_map[s] for s in sid], dtype=np.float64)
    ev = m > 32

    feats = frame.features
    y_all = y
    pre = CleanRealMLPPreprocessor(feature_names=tuple(feats.columns), config=cfg)
    pre.fit(feats.iloc[~ev], y_all[~ev])
    x_ev, c_ev = pre.transform(feats.iloc[ev])
    x_tr, c_tr = pre.transform(feats.iloc[~ev])  # cat_dims 必须与训练协议一致(train 计算)

    torch, _, F, RealMLPRQ = _build_torch_classes()
    ck = torch.load(MODEL_CKPT, map_location="cpu")
    cat_dims = [int(np.max(c_tr[:, j]) + 2) if c_tr.shape[1] else 1 for j in range(c_tr.shape[1])]
    model = RealMLPRQ(x_ev.shape[1], cat_dims, cfg)
    model.load_state_dict(ck)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        p_ev = _predict(model, x_ev, c_ev, cfg, str(device), torch)
    return p_ev, y[ev], m[ev], None


def main():
    t0 = time.time()
    p_ev, y_ev, m_ev, _ = load_model_and_preds()
    print(f"PSEUDO eval preds ready (n={len(p_ev)}) ({time.time()-t0:.0f}s)")

    # Z 特征 (eval 33-70, 与 P9-NEUT 同集)
    df = pl.read_parquet(DATA / "f0726_train_z_f32.parquet")
    lab = pl.read_ipc(RAW / "label.feather").select(["sample_id", "month"]).sort("sample_id")
    df = df.join(lab, on="sample_id", how="left")
    feat_cols = [c for c in df.columns if c not in ("sample_id", "month", "target")]
    ev = df["month"].to_numpy() > 32
    X_ev = df.filter(pl.col("month") > 32).select(feat_cols).to_numpy().astype(np.float64)
    m_ev_df = df.filter(pl.col("month") > 32)["month"].to_numpy()
    z_cols = [c for c in NUISANCE if c in feat_cols]
    Z = np.nan_to_num(np.column_stack([X_ev[:, feat_cols.index(c)] for c in z_cols] + [np.abs(p_ev)]),
                      nan=0.0, posinf=0.0, neginf=0.0)

    # 定标: calibration 33-50 fit β, frozen 51-70 验证 γ
    cal = m_ev_df <= 50
    fro = m_ev_df > 50
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(cal.sum()), Z[cal]]), p_ev[cal], rcond=None)
    yhat = Z @ beta[1:] + beta[0]
    base_frozen = cosine_uncentered(p_ev[fro], y_ev[fro])
    results = {"baseline_frozen_51_70": base_frozen, "z_cols": z_cols + ["|p|"]}
    best = {"gamma": 0.0, "cos": base_frozen}
    for g in [0.0, 0.25, 0.5, 0.75, 1.0]:
        pn = p_ev - g * yhat
        c_fro = cosine_uncentered(pn[fro], y_ev[fro])
        results[f"gamma_{g}"] = c_fro
        if c_fro > best["cos"]:
            best = {"gamma": g, "cos": c_fro}
    print(f"gamma 扫描: {results}")
    print(f"选定 γ={best['gamma']} (frozen cos={best['cos']:.6f}, 基线 {base_frozen:.6f})")

    # test 应用
    te = np.load(OUT / "rq_scfi_test_pred.npz")
    p_te = te["pred"].astype(np.float64).ravel()
    test_ids = te["test_ids"]
    f_test = pl.read_parquet(DATA / "f0726_test_z_f32.parquet")
    X_te = f_test.select(feat_cols).to_numpy().astype(np.float64)
    Z_te = np.nan_to_num(np.column_stack([X_te[:, feat_cols.index(c)] for c in z_cols] + [np.abs(p_te)]),
                         nan=0.0, posinf=0.0, neginf=0.0)
    p_final = p_te - best["gamma"] * (Z_te @ beta[1:] + beta[0])
    np.savez(OUT / "rq_scfi_neut_test_pred.npz", pred=p_final, test_ids=test_ids)

    # submission 文件
    sub = pl.DataFrame({"sample_id": test_ids, "target": p_final}).sort("sample_id")
    sub.write_csv(OUT / "submission_rq_scfi_neut.csv")
    print(f"final test pred: mean={p_final.mean():.2e} std={p_final.std():.2e}")
    print(f"submission saved: {OUT / 'submission_rq_scfi_neut.csv'} ({len(p_final):,})")
    (OUT / "neutralize_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
