# -*- coding: utf-8 -*-
"""C 系列 RealMLP recipe 变体训练器 (论文 Better by Default v2 顺序, 单变量消融).

用法: python scripts/c_recipe.py <variant>
  baseline    C-05: robust+clip + β2=0.999 + constant LR (复现锚)
  no_clip     C-06: StandardScaler (无 robust+clip 对照)
  beta2_095   C-07: β2=0.95
  c08_cosine  C-08: constant LR → cosine decay
  c09_paramish C-09: ReLU → Parametric Mish (α init=1, α lr×0.1)
  c10_pl      C-10: 无嵌入 → PL 数值嵌入 (4 维/特征, lr×0.1)
  c11_pbld    C-11: 无嵌入 → PBLD 数值嵌入 (4 维/特征 densenet, lr×0.1)
  c12_scaling C-12: 无 → learnable scaling layer (s init=1, lr×6)
  c13_schedreg C-13: dropout 0.15 + wd 0.02 (flat_cos 调度)
  c14_coslog4 C-14: cosine → coslog4 (4 cycles)
  c15_ntp_init C-15: NT parametrization + 数据驱动 init + he+5 bias

协议: PSEUDO fold (train m0-32 / eval m33-70), 严格 temporal.
纪律: 训练用 MSE, 选 checkpoint 用 GLOBAL cosine; fold-local 预处理 fit.
实现忠实于 pytabkit 源码 (research/paper-reading-2026-08/pytabkit_code/).
"""
import json
import math
import os
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn

RAW = r"D:\mscapital-forecasting\data\raw\train"
FEAT = r"D:\mscapital-forecasting\data\processed\f0726_train.parquet"
OUT = r"D:\mscapital-kaggle\output\c_recipe"
os.makedirs(OUT, exist_ok=True)

SEED = 2026
EPOCHS = 30
BATCH = 512
LR = 1e-3
HIDDEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VARIANTS = ("baseline", "no_clip", "beta2_095", "c08_cosine", "c09_paramish",
            "c10_pl", "c11_pbld", "c12_scaling", "c13_schedreg", "c14_coslog4",
            "c15_ntp_init")


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def cosine_uncentered(p, y):
    p = p.reshape(-1).astype(np.float64)
    y = y.reshape(-1).astype(np.float64)
    return float(p @ y / (np.sqrt(p @ p) * np.sqrt(y @ y) + 1e-30))


# ---------- 预处理 ----------
class RobustScaleSmoothClip:
    """median_center + robust_scale + smooth_clip (pytabkit tfms)."""
    def fit(self, X):
        self.median = np.median(X, axis=0)
        q75, q25 = np.quantile(X, 0.75, axis=0), np.quantile(X, 0.25, axis=0)
        qd = q75 - q25
        self.factors = np.zeros_like(qd)
        ok = qd != 0
        self.factors[ok] = 1.0 / qd[ok]
        zero = qd == 0
        rng = np.max(X, axis=0) - np.min(X, axis=0)
        self.factors[zero] = np.where(rng[zero] != 0, 2.0 / rng[zero], 0.0)
        return self

    def transform(self, X):
        x = self.factors[None, :] * (X - self.median[None, :])
        return x / np.sqrt(1 + (x / 3.0) ** 2)


class StandardScale:
    """对照: mean/std 标准化, 无 smooth clip (C-06)."""
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean[None, :]) / self.std[None, :]


# ---------- 组件 (忠实 pytabkit) ----------
def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))


class ParametricMish(nn.Module):
    """ParametricActivationLayer: x + (mish(x) - x) * alpha, alpha init=1 (lr×0.1)."""
    def __init__(self, n_features):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, n_features))

    def forward(self, x):
        return x + (mish(x) - x) * self.alpha


class PLEmbeddings(nn.Module):
    """PL: x_i → (x_i, W·cos(2π·w·x_i + b)) ∈ R^4, w,b∈R^16, W∈R^{16×3}. lr×0.1."""
    def __init__(self, n_features, n_freq=16, d_out=3, sigma=0.1):
        super().__init__()
        self.w = nn.Parameter(sigma * torch.randn(n_features, 1, n_freq))
        self.b = nn.Parameter(np.pi * (-1 + 2 * torch.rand(n_features, 1, n_freq)))
        self.W = nn.Parameter((-1 + 2 * torch.rand(n_features, n_freq, d_out)) / np.sqrt(n_freq))
        self.d_out = d_out + 1  # +1 原值 (densenet)

    def forward(self, x):
        x_orig = x
        x = x.transpose(-1, -2).unsqueeze(-1)  # (B, n_feat) -> (n_feat, B, 1)
        h = torch.cos(2 * math.pi * x.matmul(self.w) + self.b)  # (n_feat, B, n_freq)
        h = h.matmul(self.W)  # (n_feat, B, d_out)
        h = h.transpose(-2, -3)  # (B, n_feat, d_out)
        return torch.cat([h, x_orig.unsqueeze(-1)], dim=-1).reshape(x_orig.shape[0], -1)


class PBLDEmbeddings(nn.Module):
    """PBLD (pytabkit tabr_lib): cos(2π·x·w1+b1) @ w2 + b2, densenet cat 原值 → 4 维/特征. lr×0.1."""
    def __init__(self, n_features, n_freq=16, d_embedding=4, sigma=0.1):
        super().__init__()
        hidden_2 = d_embedding - 1  # densenet 保留 1 维给原值
        self.weight_1 = nn.Parameter(sigma * torch.randn(n_features, 1, n_freq))
        self.weight_2 = nn.Parameter((-1 + 2 * torch.rand(n_features, n_freq, hidden_2)) / np.sqrt(n_freq))
        self.bias_1 = nn.Parameter(np.pi * (-1 + 2 * torch.rand(n_features, 1, n_freq)))
        self.bias_2 = nn.Parameter((-1 + 2 * torch.rand(n_features, 1, hidden_2)) / np.sqrt(n_freq))

    def forward(self, x):
        x_orig = x
        x = x.transpose(-1, -2).unsqueeze(-1)  # (B, n_feat) -> (n_feat, B, 1)
        x = torch.cos(2 * math.pi * x.matmul(self.weight_1) + self.bias_1)  # (n_feat, B, freq)
        x = x.matmul(self.weight_2) + self.bias_2  # (n_feat, B, d-1), bias 广播
        x = x.transpose(-2, -3)  # (B, n_feat, d-1)
        return torch.cat([x, x_orig.unsqueeze(-1)], dim=-1).reshape(x_orig.shape[0], -1)


class ScalingLayer(nn.Module):
    """Learnable scaling: x' = s_i·x_i, s init=1 (lr×6)."""
    def __init__(self, n_features):
        super().__init__()
        self.s = nn.Parameter(torch.ones(n_features))

    def forward(self, x):
        return x * self.s


class NTLinear(nn.Module):
    """NT parametrization: z = d^{-1/2} W x + b (C-15)."""
    def __init__(self, n_in, n_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_in, n_out))
        self.bias = nn.Parameter(torch.zeros(n_out))
        self.n_in = n_in

    def forward(self, x):
        return x @ (self.weight / math.sqrt(self.n_in)) + self.bias


class MLP(nn.Module):
    def __init__(self, n_in, hidden=256, variant="baseline"):
        super().__init__()
        self.variant = variant
        lin = NTLinear if variant == "c15_ntp_init" else nn.Linear
        act = ParametricMish if variant == "c09_paramish" else nn.ReLU
        self.net = nn.Sequential(
            lin(n_in, hidden), act(hidden),
            lin(hidden, hidden), act(hidden),
            lin(hidden, hidden), act(hidden),
            lin(hidden, 1),
        )
        if variant == "c15_ntp_init":
            self._data_driven_init = True  # 训练前用一批数据初始化

    def forward(self, x):
        return self.net(x).squeeze(-1)


def data_driven_init(model, X_batch, variant):
    """C-15: 首前向数据依赖初始化 (权重行重缩放至输出方差 1) + he+5 bias.

    he+5: 对第一层输出 h (256 维) 随机抽 5 样本, 指数加权求和取负 (pytabkit heplus_bias).
    """
    if variant != "c15_ntp_init":
        return
    first = model.net[0]
    with torch.no_grad():
        h = X_batch[:8192] @ (first.weight.detach() / math.sqrt(first.n_in))  # (8192, 256)
        hn = h.cpu().numpy()
        rng = np.random.default_rng(0)
        n_out = hn.shape[1]
        idxs = rng.integers(0, hn.shape[0], size=(n_out, 5))
        sw = rng.exponential(1.0, size=(n_out, 5))
        sw /= sw.sum(axis=1, keepdims=True)
        out = np.stack([hn[idxs[:, i], np.arange(n_out)] for i in range(5)], axis=1)
        b = -(out * sw).sum(axis=1)
        first.bias.data = torch.from_numpy(b.astype(np.float32)).to(DEVICE)
        # 数据驱动权重: 输出 std → 1 (跨 batch)
        std = h.std(dim=0, unbiased=False).clamp(min=1e-8)
        first.weight.data = first.weight.data / std[None, :].to(DEVICE)
        print("[c15_ntp_init] data-driven init done "
              f"(bias mean={b.mean():+.3f}, std range {std.min():.3f}-{std.max():.3f})", flush=True)


def get_lr_schedule(variant):
    """返回 lr(t) 乘数, t = epoch/EPOCHS ∈ [0,1)."""
    if variant == "c08_cosine":
        # cosine decay: 1 → 0 (pytabkit cos_sched.scaled(1., 0.))
        return lambda t: 0.5 * (1.0 + math.cos(math.pi * t))
    if variant == "c14_coslog4":
        return lambda t: 0.5 * (1.0 - math.cos(2 * math.pi * math.log2(1 + 15 * t)))
    return None  # constant


def flat_cos(t):
    """flat_cos: 前 50% 保持 1.0, 后 50% cosine → 0."""
    if t < 0.5:
        return 1.0
    u = 2.0 * (t - 0.5)
    return 0.5 * (1.0 + math.cos(math.pi * u))  # 1→0


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    assert variant in VARIANTS, f"{variant} not in {VARIANTS}"
    t0 = time.time()

    df = pl.read_parquet(FEAT)
    lab = pl.read_ipc(f"{RAW}/label.feather")
    df = df.join(lab.select(["sample_id", "month"]), on="sample_id", how="left")
    feat_cols = [c for c in df.columns if c not in ("sample_id", "target", "month")]
    X = df.select(feat_cols).to_numpy().astype(np.float32)
    y = df["target"].to_numpy().astype(np.float64)
    m = df["month"].to_numpy()

    tr = m <= 32
    X_tr, y_tr, X_ev, y_ev = X[tr], y[tr], X[~tr], y[~tr]
    n_tr = X_tr.shape[0]
    print(f"[{variant}] train={n_tr} eval={X_ev.shape[0]} feat={X_tr.shape[1]}", flush=True)

    pp = RobustScaleSmoothClip() if variant != "no_clip" else StandardScale()
    pp.fit(X_tr)
    X_tr = np.nan_to_num(pp.transform(X_tr).astype(np.float32), nan=0.0)
    X_ev = np.nan_to_num(pp.transform(X_ev).astype(np.float32), nan=0.0)

    betas = (0.9, 0.999) if variant != "beta2_095" else (0.9, 0.95)
    lr_sched = get_lr_schedule(variant)

    # 嵌入 / scaling 前置模块
    n_in = X_tr.shape[1]
    front = None
    if variant == "c10_pl":
        front = PLEmbeddings(n_in).to(DEVICE)
        n_in = n_in * 4
    elif variant == "c11_pbld":
        front = PBLDEmbeddings(n_in).to(DEVICE)
        n_in = n_in * 4
    elif variant == "c12_scaling":
        front = ScalingLayer(n_in).to(DEVICE)

    set_seed(SEED)
    model = MLP(n_in, HIDDEN, variant).to(DEVICE)

    # 参数组: front 模块单独 lr 因子; parametric act 的 alpha 单独 lr×0.1
    if variant == "c09_paramish":
        model_params = [p for n, p in model.named_parameters() if "alpha" not in n]
        param_groups = [{"params": model_params}]
    else:
        param_groups = [{"params": model.parameters()}]
    if front is not None:
        f = front
        if variant in ("c10_pl", "c11_pbld"):
            lr_f = 0.1  # plr_lr_factor
        else:
            lr_f = 6.0  # scale_lr_factor
        param_groups.append({"params": f.parameters(), "lr": LR * lr_f})
    if variant == "c09_paramish":
        param_groups.append({"params": [p for n, p in model.named_parameters() if "alpha" in n], "lr": LR * 0.1})
    opt = torch.optim.AdamW(param_groups, lr=LR, betas=betas, weight_decay=0.0)
    lossf = nn.MSELoss()

    Xt = torch.from_numpy(X_tr).to(DEVICE)
    yt = torch.from_numpy(y_tr.astype(np.float32)).to(DEVICE)
    Xe = torch.from_numpy(X_ev).to(DEVICE)

    # C-15: 数据驱动 init (首前向)
    if variant == "c15_ntp_init":
        data_driven_init(model, Xt, variant)

    # C-13: dropout + wd flat_cos
    use_schedreg = variant == "c13_schedreg"
    drop = nn.Dropout(0.15) if use_schedreg else None
    wd0 = 0.02 if use_schedreg else 0.0

    n = len(X_tr)
    hist = []
    best_cos, best_cos_ep = -9, 0
    for ep in range(EPOCHS):
        t = ep / EPOCHS
        # 调度
        if lr_sched is not None:
            for g in opt.param_groups:
                g["lr"] = g.get("base_lr", LR) * lr_sched(t)
        else:
            for g in opt.param_groups:
                if g.get("base_lr", None) is None:
                    g["base_lr"] = g["lr"]
        if use_schedreg:
            p = 0.15 * flat_cos(t)
            wd = wd0 * flat_cos(t)
            drop.p = p
            for g in opt.param_groups:
                g["weight_decay"] = wd if "alpha" not in str(g["params"][0].shape) else 0.0
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = Xt[idx]
            if front is not None:
                xb = front(xb)
            if drop is not None:
                xb = drop(xb)
            opt.zero_grad()
            loss = lossf(model(xb), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = []
            for i in range(0, len(Xe), 4096):
                xb = Xe[i:i + 4096]
                if front is not None:
                    xb = front(xb)
                pv.append(model(xb).cpu().numpy())
        p_ev = np.concatenate(pv)
        cos = cosine_uncentered(p_ev, y_ev)
        mse = float(np.mean((p_ev - y_ev) ** 2))
        hist.append({"epoch": ep + 1, "cosine": cos, "mse": mse})
        if cos > best_cos:
            best_cos, best_cos_ep = cos, ep + 1
            torch.save(model.state_dict(), f"{OUT}/{variant}_best.pt")
        print(f"[{variant}] ep {ep+1:02d} cos={cos:.6f} lr={opt.param_groups[0]['lr']:.2e}", flush=True)

    model.load_state_dict(torch.load(f"{OUT}/{variant}_best.pt"))
    model.eval()
    with torch.no_grad():
        p_final = []
        for i in range(0, len(Xe), 4096):
            xb = Xe[i:i + 4096]
            if front is not None:
                xb = front(xb)
            p_final.append(model(xb).cpu().numpy())
    final_cos = cosine_uncentered(np.concatenate(p_final), y_ev)

    results = {
        "variant": variant, "betas": list(betas),
        "preprocess": "robust+clip" if variant != "no_clip" else "standard",
        "pseudo_eval": {"train_months": "0-32", "eval_months": "33-70"},
        "best_cosine_epoch": best_cos_ep, "best_cosine": best_cos,
        "final_cosine": final_cos, "history": hist,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(f"{OUT}/{variant}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: results[k] for k in ("variant", "best_cosine_epoch", "best_cosine", "runtime_s")}, indent=2))


if __name__ == "__main__":
    main()
