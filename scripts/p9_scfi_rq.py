# -*- coding: utf-8 -*-
"""H-SCFI-RQ: SCFI 特征 → RealMLP RQ 生产验证 (PSEUDO m0-32/m33-70).

A 臂: 152 特征 (f0726_train_f32.parquet)
C 臂: 152 + 73 Z 特征 (f0726_train_z_f32.parquet)
骨架: 生产 RealMLP RQ (RealMLPConfig n_ens=16, 30 epochs, flat_anneal, EMA),
      fold-local 预处理 (fit 只 train), best-epoch global cosine 选择.
对照: P5-07 spot-check (+0.0040, 51-70) 升级为全协议 PSEUDO.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\mscapital-kaggle\src")
sys.path.insert(0, r"D:\mscapital-kaggle\scripts")

import numpy as np
import polars as pl
import torch

from mscapital.models.realmlp import (
    RealMLPConfig, load_frame, CleanRealMLPPreprocessor,
    _build_torch_classes, RQKMeansEncoder, _make_optimizer, _EMA,
    _loss, _predict, _set_seed, flat_anneal,
)

RAW = Path(r"D:\mscapital-forecasting\data\raw\train")
DATA = Path(r"D:\mscapital-forecasting\data\processed")
OUT = Path(r"D:\mscapital-kaggle\output\p9_scfi_rq")

FEATS = {
    "A": DATA / "f0726_train_f32.parquet",
    "C": DATA / "f0726_train_z_f32.parquet",
}


def cosine_uncentered(p, y):
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    return float(p @ y / (np.sqrt(p @ p) * np.sqrt(y @ y) + 1e-30))


def run_arm(arm: str, epochs: int = 30):
    t0 = time.time()
    out = OUT / f"arm_{arm}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = RealMLPConfig(epochs=epochs)
    print(f"[arm {arm}] cfg epochs={cfg.epochs} n_ens={cfg.n_ens} lr={cfg.learning_rate}")

    frame = load_frame(FEATS[arm], RAW / "label.feather")
    lab = pl.read_ipc(RAW / "label.feather").select(["sample_id", "month"]).sort("sample_id")
    sid = np.asarray(frame.sample_id)
    months = lab["month"].to_numpy()
    # align months to frame order
    lab_map = dict(zip(lab["sample_id"].to_numpy(), months))
    m = np.asarray([lab_map[s] for s in sid])
    tr = m <= 32
    print(f"[arm {arm}] train {tr.sum():,} / eval {(~tr).sum():,} ({time.time()-t0:.0f}s)")

    feats = frame.features
    y_all = np.asarray(frame.target, dtype=np.float64)
    X_tr, X_ev = feats.iloc[tr], feats.iloc[~tr]
    y_tr, y_ev = y_all[tr], y_all[~tr]

    pre = CleanRealMLPPreprocessor(feature_names=tuple(feats.columns), config=cfg)
    pre.fit(X_tr, y_tr)
    x_tr, c_tr = pre.transform(X_tr)
    x_ev, c_ev = pre.transform(X_ev)
    print(f"[arm {arm}] preprocessed num={x_tr.shape} cats={c_tr.shape[1]} ({time.time()-t0:.0f}s)")

    torch, _, F, RealMLPRQ = _build_torch_classes()
    _set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rq = RQKMeansEncoder(cfg.rq_encoder_layers, cfg.rq_vocab_size).fit(y_tr)
    codes_np = rq.encode(y_tr)
    cat_dims = [int(np.max(c_tr[:, j]) + 2) if c_tr.shape[1] else 1 for j in range(c_tr.shape[1])]
    model = RealMLPRQ(x_tr.shape[1], cat_dims, cfg).to(device)
    optimizer = _make_optimizer(model, cfg, torch)
    ema = _EMA(model, cfg.ema_decay, torch)

    xt = torch.from_numpy(x_tr).to(device)
    ct = torch.from_numpy(c_tr).to(device)
    yt = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    codes = torch.from_numpy(codes_np).to(device)
    n = len(y_tr)
    total_batches = (n + cfg.train_batch_size - 1) // cfg.train_batch_size
    total_steps = max(total_batches * cfg.epochs, 1)

    step = 0
    best_cos, best_ep = -9, 0
    for ep in range(1, cfg.epochs + 1):
        ep_t0 = time.time()
        permutation = torch.randperm(n, device=device)
        model.train()
        for start in range(0, n, cfg.train_batch_size):
            progress = min(step / total_steps, 1.0)
            for group, base in zip(optimizer.param_groups, (20.0, 0.093, 1.0, 1.0, 0.1)):
                group["lr"] = flat_anneal(cfg.learning_rate * base, progress)
            indices = permutation[start:start + cfg.train_batch_size]
            target = yt[indices]
            noisy = target + torch.randn_like(target) * (cfg.label_noise_std * (1.0 - progress))
            optimizer.zero_grad(set_to_none=True)
            logits, pred = model(xt[indices], ct[indices], return_codes=True)
            loss, cos, mse, rq_loss = _loss(pred, noisy, logits, codes[indices], progress, cfg, torch, F)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
            ema.update()
            step += 1
        # eval with EMA weights
        original = ema.apply()
        with torch.no_grad():
            p_ev = _predict(model, x_ev, c_ev, cfg, str(device), torch)
        cos_ev = cosine_uncentered(p_ev, y_ev)
        if cos_ev > best_cos:
            best_cos, best_cos_ep = cos_ev, ep
            torch.save(model.state_dict(), out / "best.pt")  # EMA weights (apply 后)
        ema.restore(original)

        ep_dt = time.time() - ep_t0
        print(f"[arm {arm}] ep {ep:02d}/{cfg.epochs} cos={cos_ev:.6f} | {ep_dt:.0f}s | ETA {(ep_dt*(cfg.epochs-ep))/60:.0f}min", flush=True)

    # final: load best EMA weights
    model.load_state_dict(torch.load(out / "best.pt"))
    with torch.no_grad():
        p_final = _predict(model, x_ev, c_ev, cfg, str(device), torch)

    m_ev = m[~tr]
    monthly = {}
    for mm in np.unique(m_ev):
        mask = m_ev == mm
        monthly[int(mm)] = cosine_uncentered(p_final[mask], y_ev[mask])
    res = {"arm": arm, "n_feat": int(feats.shape[1]),
           "best_cosine_epoch": best_cos_ep, "best_cosine": best_cos,
           "final_cosine": cosine_uncentered(p_final, y_ev),
           "monthly": monthly, "runtime_s": round(time.time() - t0, 1)}
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[arm {arm}] DONE best_ep={best_cos_ep} cos={best_cos:.6f} ({time.time()-t0:.0f}s)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "C", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=30, help="training epochs (smoke test: 2)")
    args = ap.parse_args()
    results = {}
    if args.arm in ("A", "both"):
        results["A"] = run_arm("A", args.epochs)
    if args.arm in ("C", "both"):
        results["C"] = run_arm("C", args.epochs)
    if "A" in results and "C" in results:
        dA, dC = results["A"]["best_cosine"], results["C"]["best_cosine"]
        npos = sum(1 for mm in results["C"]["monthly"]
                   if results["C"]["monthly"][mm] > results["A"]["monthly"].get(mm, 0))
        nmon = len(results["C"]["monthly"])
        summary = {"delta_C_minus_A": dC - dA, "monthly_positive": f"{npos}/{nmon}"}
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n=== RQ Δ(C−A) = {dC-dA:+.6f} (A={dA:.6f} → C={dC:.6f}), 月度正 {npos}/{nmon} ===")


if __name__ == "__main__":
    main()
