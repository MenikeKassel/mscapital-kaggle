# -*- coding: utf-8 -*-
"""P9-Lite 结果汇总与判定 (读 output/p9_lite/pkg{ABC}/{base,feat}/results.json).

输出: 每包 base vs feat 在 eval 33-70 与 frozen 51-70 的 cosine、Δ、月度正数,
      base-feat 预测相关 (增量独立性粗判), 以及 GREEN/YELLOW/RED 判定草稿.
"""
from __future__ import annotations

import json
import os

import numpy as np

OUT = r"D:\mscapital-kaggle\output\p9_lite"
PKGS = ["A", "B", "C"]


def load(pkg, arm):
    with open(f"{OUT}/pkg{pkg}/{arm}/results.json") as f:
        return json.load(f)


def main():
    rows = []
    for pkg in PKGS:
        base = load(pkg, "base")
        feat = load(pkg, "feat")
        p_b = np.load(f"{OUT}/pkg{pkg}/base/preds.npz")
        p_f = np.load(f"{OUT}/pkg{pkg}/feat/preds.npz")
        assert np.array_equal(p_b["sample_id"], p_f["sample_id"])
        pred_b, pred_f, yy, mm = p_b["pred"], p_f["pred"], p_b["y"], p_b["month"]

        def cos(p, sel):
            p = p[sel].astype(np.float64); y = yy[sel].astype(np.float64)
            return float(p @ y / (np.sqrt(p @ p) * np.sqrt(y @ y) + 1e-30))

        fr = (mm >= 51) & (mm <= 70)
        d = {k: float(v) for k, v in feat["monthly_cosine"].items()}
        db = {k: float(v) for k, v in base["monthly_cosine"].items()}
        # 月度 Δ (frozen)
        mdel = {int(k): d.get(k, 0.0) - db.get(k, 0.0) for k in db if 51 <= int(k) <= 70}
        pos_delta = sum(1 for v in mdel.values() if v > 0)
        # 相关
        corr = float(np.corrcoef(pred_b, pred_f)[0, 1])

        rows.append({
            "pkg": pkg,
            "n_feat_base": base["n_feat"], "n_feat_feat": feat["n_feat"],
            "eval33_70": (base["best_cosine_eval33_70"], feat["best_cosine_eval33_70"]),
            "frozen51_70": (base["frozen51_70_cosine"], feat["frozen51_70_cosine"]),
            "d_frozen": feat["frozen51_70_cosine"] - base["frozen51_70_cosine"],
            "pos_base": base["frozen51_70_pos_months"], "pos_feat": feat["frozen51_70_pos_months"],
            "delta_pos_months": pos_delta, "pred_corr": corr,
        })

    print(f"{'pkg':>4} {'eval Δ':>9} {'frozen Δ':>10} {'pos 51-70 b→f':>16} {'Δ月正':>6} {'corr':>7}  判定草稿")
    for r in rows:
        de = r["eval33_70"][1] - r["eval33_70"][0]
        df_ = r["d_frozen"]
        jud = "GREEN" if df_ >= 0.001 and r["delta_pos_months"] >= 12 else (
            "YELLOW" if 0.0005 <= df_ < 0.001 else ("RED" if df_ < 0.0005 else "YELLOW"))
        print(f"{r['pkg']:>4} {de:+9.6f} {df_:+10.6f} {r['pos_base']:>4}→{r['pos_feat']:<7} "
              f"{r['delta_pos_months']:>6} {r['pred_corr']:>7.4f}  {jud}")

    # 明细: 每包 frozen 月度 Δ 排序 + 累积
    print("\n--- frozen 51-70 月度 Δ 明细 (feat−base) ---")
    for pkg in PKGS:
        base = load(pkg, "base"); feat = load(pkg, "feat")
        db = {int(k): float(v) for k, v in base["monthly_cosine"].items()}
        d = {int(k): float(v) for k, v in feat["monthly_cosine"].items()}
        mdel = sorted(((int(k), d.get(k, 0.0) - db.get(k, 0.0)) for k in db if 51 <= int(k) <= 70),
                      key=lambda x: -x[1])
        top3 = ", ".join(f"{m}:{v:+.5f}" for m, v in mdel[:3])
        bot3 = ", ".join(f"{m}:{v:+.5f}" for m, v in mdel[-3:])
        print(f"  P9-{pkg}: top3[{top3}]  bot3[{bot3}]  pos={sum(1 for _,v in mdel if v>0)}/20")


if __name__ == "__main__":
    main()
