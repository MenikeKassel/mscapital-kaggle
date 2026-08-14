# -*- coding: utf-8 -*-
"""P5-B follow-up: learner-robustness spot-check (NN vs LGB).

问题: SCFI 的 Z 特征增益 (+0.0075 LGB) 是否 learner 特异?
用 F1 风格 SmallMLP [256×2] + cosine loss, 3 seeds, 同 split (B51_60/B61_70),
对比 arm A (152) vs arm C (152+Z) — 与 P5-B 完全同协议, 只换 learner。

若 ΔC 在 NN 上存活 → 结论 learner 家族无关; 否则 → LGB 特异, 降级标注。
"""
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OUT = Path(r"D:\mscapital-kaggle\output\p5b_scfi")
EPOCHS = 15
BATCH = 1024
LR = 1e-3
WD = 1e-4
SEEDS = (7, 42, 2026)


def cosine(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(p, y) / np.sqrt(np.dot(p, p) * np.dot(y, y)))


def rms_norm(p: np.ndarray) -> np.ndarray:
    s = float(np.sqrt(np.mean(p ** 2)))
    return p / s if s > 0 else p


class SmallMLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.SiLU(),
                                 nn.Linear(256, 256), nn.SiLU(),
                                 nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).reshape(-1)


def train(Xtr, ytr, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SmallMLP(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    Xt = torch.from_numpy(Xtr).float().to(device)
    yt = torch.from_numpy(ytr.astype(np.float32)).to(device)
    n = len(Xtr)
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            p = model(Xt[idx])
            loss = 1.0 - torch.nn.functional.cosine_similarity(
                p.reshape(1, -1), yt[idx].reshape(1, -1), dim=1).squeeze()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


@torch.no_grad()
def predict(model, X, device):
    model.eval()
    outs = []
    for i in range(0, len(X), BATCH * 4):
        outs.append(model(torch.from_numpy(X[i:i + BATCH * 4]).float().to(device)).cpu().numpy())
    return np.concatenate(outs).reshape(-1)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    t0 = time.time()
    results = {}
    for blk in ("B51_60", "B61_70"):
        d = np.load(OUT / f"spotcheck_{blk}.npz")
        XA_tr, XA_ho, XA_ev = d["X152_tr"], d["X152_ho"], d["X152_ev"]
        Z_tr, Z_ho, Z_ev = d["Z_tr"], d["Z_ho"], d["Z_ev"]
        XC_tr = np.hstack([XA_tr, Z_tr]).astype(np.float32)
        XC_ho = np.hstack([XA_ho, Z_ho]).astype(np.float32)
        XC_ev = np.hstack([XA_ev, Z_ev]).astype(np.float32)
        ytr, yho, yev = d["y_tr"], d["y_ho"], d["y_ev"]
        canon_ev, canon_ho = d["canon_ev"], d["canon_ho"]
        month_ev = d["month_ev"]
        print(f"\n=== {blk}: A={XA_tr.shape[1]}d C={XC_tr.shape[1]}d "
              f"train={len(ytr):,} eval={len(yev):,} ===", flush=True)

        blk_res = {}
        for arm, (Xtr, Xho, Xev) in {"A": (XA_tr, XA_ho, XA_ev),
                                     "C": (XC_tr, XC_ho, XC_ev)}.items():
            # median impute (train stats) + standardize — f0726 含 ~2% NaN,
            # MLP 不能像 LGB 原生处理 NaN
            med = np.nanmedian(Xtr, axis=0)
            Xtr_i = np.where(np.isnan(Xtr), med, Xtr)
            Xho_i = np.where(np.isnan(Xho), med, Xho)
            Xev_i = np.where(np.isnan(Xev), med, Xev)
            mu = Xtr_i.mean(axis=0, keepdims=True)
            sd = Xtr_i.std(axis=0, keepdims=True) + 1e-6
            Xtr_s = np.clip((Xtr_i - mu) / sd, -10, 10).astype(np.float32)
            Xho_s = np.clip((Xho_i - mu) / sd, -10, 10).astype(np.float32)
            Xev_s = np.clip((Xev_i - mu) / sd, -10, 10).astype(np.float32)
            assert np.isfinite(Xtr_s).all() and np.isfinite(Xev_s).all(), "non-finite after impute"
            preds_ev, preds_ho = [], []
            for seed in SEEDS:
                model = train(Xtr_s, ytr, device, seed)
                preds_ev.append(predict(model, Xev_s, device))
                preds_ho.append(predict(model, Xho_s, device))
                del model
                gc.collect()
            p_ev = np.mean(preds_ev, axis=0)
            p_ho = np.mean(preds_ho, axis=0)
            best_w, best_s = 0.0, -1e9
            for w in np.arange(0.05, 0.51, 0.05):
                s = cosine(rms_norm(canon_ho) + w * rms_norm(p_ho), yho)
                if s > best_s:
                    best_s, best_w = s, float(w)
            p_blend = rms_norm(canon_ev) + best_w * rms_norm(p_ev)
            blk_res[arm] = {
                "cos": cosine(p_ev, yev),
                "corr_canon": float(np.corrcoef(p_ev, canon_ev)[0, 1]),
                "blend_w": float(best_w),
                "blend_delta": cosine(p_blend, yev) - cosine(canon_ev, yev),
                "monthly": {},
            }
            for mm in np.unique(month_ev):
                mk = month_ev == mm
                blk_res[arm]["monthly"][int(mm)] = cosine(p_ev[mk], yev[mk])
            print(f"  {arm}: cos={blk_res[arm]['cos']:.6f} "
                  f"corrCanon={blk_res[arm]['corr_canon']:.3f} "
                  f"blendΔ={blk_res[arm]['blend_delta']:+.6f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        blk_res["delta_C_minus_A"] = blk_res["C"]["cos"] - blk_res["A"]["cos"]
        blk_res["monthly_delta_C_minus_A"] = {
            mm: blk_res["C"]["monthly"][mm] - blk_res["A"]["monthly"][mm]
            for mm in blk_res["A"]["monthly"]}
        results[blk] = blk_res

    dA = (results["B51_60"]["delta_C_minus_A"] + results["B61_70"]["delta_C_minus_A"]) / 2
    dA1, dA2 = results["B51_60"]["delta_C_minus_A"], results["B61_70"]["delta_C_minus_A"]
    md = {**results["B51_60"]["monthly_delta_C_minus_A"],
          **results["B61_70"]["monthly_delta_C_minus_A"]}
    pos = sum(1 for v in md.values() if v > 0)
    results["consolidated"] = {
        "delta_C_minus_A_avg": dA, "B51_60": dA1, "B61_70": dA2,
        "monthly_pos": pos, "monthly_n": len(md),
        "verdict": ("NN learner: Z gain survives" if dA >= 0.0005 and pos >= 13
                    else "NN learner: Z gain does NOT survive → LGB-specific"),
    }
    (OUT / "spotcheck_results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print("\n" + "=" * 70)
    print(f"NN spot-check: ΔC(A→C) B1={dA1:+.6f} B2={dA2:+.6f} avg={dA:+.6f} "
          f"months={pos}/{len(md)}")
    print(f"verdict: {results['consolidated']['verdict']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
