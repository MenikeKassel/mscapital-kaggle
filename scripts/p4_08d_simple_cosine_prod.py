# -*- coding: utf-8 -*-
"""P4-08D: simple-MLP + cosine 全量生产 (模型 = P4-08A-unc 验证模型).

唯一变量 vs 验证: 全量 train 0-70 + test 预测. 无 RQ/EMA/成员平均,
cosine 方向信息保留 (验证 corr 0.58-0.70).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn

sys.path.insert(0, r"D:\mscapital-kaggle\src")

OUT = r"D:\mscapital-forecasting\data\processed\p12_out"
EPOCHS = 20
BATCH = 8192
LR = 1e-3
SEED = 42


class MLP(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.SiLU(), nn.Dropout(0.01),
            nn.Linear(256, 256), nn.SiLU(), nn.Dropout(0.01),
            nn.Linear(256, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1)


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tr = pl.read_parquet(r"D:\mscapital-forecasting\data\processed\f0726_train_f32.parquet").sort("sample_id")
    lab = pl.read_ipc(r"D:\mscapital-forecasting\data\raw\train\label.feather").sort("sample_id")
    names = [c for c in tr.columns if c not in ("sample_id", "target")]
    X = np.nan_to_num(tr.select(names).to_numpy().astype(np.float32), nan=0.0)
    y = lab["target"].to_numpy().astype(np.float32)
    assert len(X) == len(y) == 1257637, len(X)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    Xs = (X - mu) / sd
    print(f"train: {Xs.shape} ({time.time()-t0:.0f}s)")

    te = pl.read_parquet(r"D:\mscapital-forecasting\data\processed\f0726_test_f32.parquet").sort("sample_id")
    Xte = np.nan_to_num(te.select(names).to_numpy().astype(np.float32), nan=0.0)
    Xte = (Xte - mu) / sd
    print(f"test: {Xte.shape} ({time.time()-t0:.0f}s)")

    model = MLP(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    Xt = torch.from_numpy(Xs).to(device)
    yt = torch.from_numpy(y).to(device)
    n = len(Xs)
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        t_ep = time.time()
        nb = 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            p = model(Xt[idx])
            # uncentered cosine (生产形态, LB 指标精确形式)
            loss = 1.0 - torch.nn.functional.cosine_similarity(
                p.reshape(1, -1), yt[idx].reshape(1, -1), dim=1).squeeze()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        print(f"[ep {ep:02d}/{EPOCHS}] loss={ep_loss/nb:.5f} | {time.time()-t_ep:.0f}s | {time.time()-t0:.0f}s", flush=True)

    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xte), BATCH * 4):
            xb = torch.from_numpy(Xte[i:i + BATCH * 4]).to(device)
            outs.append(model(xb).cpu().numpy())
    pred = np.concatenate(outs)
    np.savez(f"{OUT}/realmlp_simple_cosine_test_pred.npz",
             pred=pred, test_ids=te["sample_id"].to_numpy())
    print(f"saved realmlp_simple_cosine_test_pred.npz ({len(pred):,})", flush=True)
    print(f"mean={pred.mean():.2e} std={pred.std():.2e} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
