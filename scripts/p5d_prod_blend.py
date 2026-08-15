# -*- coding: utf-8 -*-
"""P5-D: 生产级融合验证 — LGB 3-seed 集成 + CatBoost, 152+73raw / 152+Z 进 blend。

问题: P5-B 的 LGB 单 seed 证据是否在生产级配置 (多 seed 集成 + 第二 learner)
下稳定? 目标: 给出可进 blend 的最终候选 + 权重。

协议 (与 P5-B 完全相同的 temporal blocks):
  B51_60: train 0-48, holdout 49-50 (早停 + blend 调权), eval 51-60
  B61_70: train 0-58, holdout 59-60, eval 61-70
Learners: LGB (B1 参数, 3 seeds 平均) + CatBoost (项目 H1 参数, 早停 200)
Arms:    A=152, B=+raw(73), C=+Z(73), D=+raw+Z(146)
输出:    output/p5d_prod_blend/results.json + preds.npz
"""
import gc
import json
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor

OUT = Path(r"D:\mscapital-kaggle\output\p5d_prod_blend")
OUT.mkdir(parents=True, exist_ok=True)
N_THREADS = 10

LGB_PARAMS = dict(
    objective="regression", metric="rmse",
    learning_rate=0.02, num_leaves=32, min_data_in_leaf=300,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    lambda_l2=5.0, max_bin=255, verbose=-1, num_threads=N_THREADS)
LGB_SEEDS = (0, 7, 2026)

CAT_PARAMS = dict(
    iterations=10000, learning_rate=0.02, depth=6, l2_leaf_reg=5.0,
    loss_function="RMSE", random_seed=2026, thread_count=N_THREADS, verbose=False)


def cosine(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(p, y) / np.sqrt(np.dot(p, p) * np.dot(y, y)))


def rms_norm(p: np.ndarray) -> np.ndarray:
    s = float(np.sqrt(np.mean(p ** 2)))
    return p / s if s > 0 else p


def fit_lgb(Xtr, ytr, Xho, yho, Xev, seed):
    params = {**LGB_PARAMS, "seed": seed}
    dtr = lgb.Dataset(Xtr, ytr)
    dho = lgb.Dataset(Xho, yho, reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dho],
                      callbacks=[lgb.early_stopping(200)])
    return (model.predict(Xev, num_iteration=model.best_iteration),
            model.predict(Xho, num_iteration=model.best_iteration))


def fit_cat(Xtr, ytr, Xho, yho, Xev):
    model = CatBoostRegressor(**CAT_PARAMS)
    model.fit(Xtr, ytr, eval_set=(Xho, yho), early_stopping_rounds=200, verbose=False)
    return model.predict(Xev), model.predict(Xho)


def main() -> None:
    t0 = time.time()
    results = {"arms": {}, "monthly": {}}
    all_preds = {}

    # raw aggregates (months 0-70), for train/hold/eval slicing
    import polars as pl
    rawdf = pl.read_parquet(r"D:\mscapital-kaggle\output\p5b_scfi\raw_ot_agg.parquet")
    rawdf = rawdf.with_columns(pl.col("sample_id").cast(pl.Int32))
    rawdf = rawdf.sort("sample_id")
    raw_names = [c for c in rawdf.columns if c.startswith(("ob_", "tb_"))]
    raw_np = rawdf.select(raw_names).to_numpy()

    lab = pl.read_ipc(r"D:\mscapital-forecasting\data\raw\train\label.feather").sort("sample_id")
    m_all = lab["month"].to_numpy()

    blocks = [
        dict(name="B51_60", tr=np.arange(0, 49), ho=np.arange(49, 51), ev=np.arange(51, 61)),
        dict(name="B61_70", tr=np.arange(0, 59), ho=np.arange(59, 61), ev=np.arange(61, 71)),
    ]

    for blk in blocks:
        print(f"\n===== block {blk['name']} =====", flush=True)
        d = np.load(OUT.parent / "p5b_scfi" / f"spotcheck_{blk['name']}.npz")
        XA_tr, XA_ho, XA_ev = d["X152_tr"], d["X152_ho"], d["X152_ev"]
        Z_tr, Z_ho, Z_ev = d["Z_tr"], d["Z_ho"], d["Z_ev"]
        ytr, yho, yev = d["y_tr"], d["y_ho"], d["y_ev"]
        canon_ev, canon_ho = d["canon_ev"], d["canon_ho"]
        month_ev = d["month_ev"]

        tr_mask = np.isin(m_all, blk["tr"])
        ho_mask = np.isin(m_all, blk["ho"])
        ev_mask = np.isin(m_all, blk["ev"])
        R_tr = raw_np[tr_mask].astype(np.float32)
        R_ho = raw_np[ho_mask].astype(np.float32)
        R_ev = raw_np[ev_mask].astype(np.float32)

        feat_sets = {
            "A": [XA_tr, XA_ho, XA_ev],
            "B": [np.hstack([XA_tr, R_tr]), np.hstack([XA_ho, R_ho]), np.hstack([XA_ev, R_ev])],
            "C": [np.hstack([XA_tr, Z_tr]), np.hstack([XA_ho, Z_ho]), np.hstack([XA_ev, Z_ev])],
            "D": [np.hstack([XA_tr, R_tr, Z_tr]), np.hstack([XA_ho, R_ho, Z_ho]),
                  np.hstack([XA_ev, R_ev, Z_ev])],
        }

        for arm, (Xtr, Xho, Xev) in feat_sets.items():
            Xtr = Xtr.astype(np.float32)
            Xho = Xho.astype(np.float32)
            Xev = Xev.astype(np.float32)
            # LGB 3-seed
            p_lgb_ev, p_lgb_ho = zip(*[fit_lgb(Xtr, ytr, Xho, yho, Xev, s) for s in LGB_SEEDS])
            p_lgb = np.mean(p_lgb_ev, axis=0)
            p_lgb_ho = np.mean(p_lgb_ho, axis=0)
            # CatBoost
            p_cat, p_cat_ho = fit_cat(Xtr, ytr, Xho, yho, Xev)
            for ln, (p, p_ho) in (("lgb", (p_lgb, p_lgb_ho)), ("cat", (p_cat, p_cat_ho))):
                # blend weight on holdout
                best_w, best_s = 0.0, -1e9
                for w in np.arange(0.05, 0.51, 0.05):
                    s = cosine(rms_norm(canon_ho) + w * rms_norm(p_ho), yho)
                    if s > best_s:
                        best_s, best_w = s, float(w)
                p_blend = rms_norm(canon_ev) + best_w * rms_norm(p)
                key = f"{arm}_{ln}"
                results["arms"].setdefault(key, {})[blk["name"]] = {
                    "cos": cosine(p, yev),
                    "blend_w": float(best_w),
                    "blend_delta": cosine(p_blend, yev) - cosine(canon_ev, yev),
                    "corr_canon": float(np.corrcoef(p, canon_ev)[0, 1]),
                }
                all_preds.setdefault(key, {})[blk["name"]] = p
                print(f"  {key}: cos={cosine(p, yev):.6f} blendΔ="
                      f"{results['arms'][key][blk['name']]['blend_delta']:+.6f} "
                      f"corr={results['arms'][key][blk['name']]['corr_canon']:.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            del Xtr, Xho, Xev
            gc.collect()

    # ---- consolidated + monthly ----
    cons = {}
    for key, blkres in results["arms"].items():
        b1, b2 = blkres["B51_60"], blkres["B61_70"]
        bd = (b1["blend_delta"] + b2["blend_delta"]) / 2
        p_all = np.concatenate([all_preds[key]["B51_60"], all_preds[key]["B61_70"]])
        ev_canon = np.concatenate([
            np.load(OUT.parent / "p5b_scfi" / "spotcheck_B51_60.npz")["canon_ev"],
            np.load(OUT.parent / "p5b_scfi" / "spotcheck_B61_70.npz")["canon_ev"]])
        y_all = np.concatenate([
            np.load(OUT.parent / "p5b_scfi" / "spotcheck_B51_60.npz")["y_ev"],
            np.load(OUT.parent / "p5b_scfi" / "spotcheck_B61_70.npz")["y_ev"]])
        m_all_ev = np.concatenate([
            np.load(OUT.parent / "p5b_scfi" / "spotcheck_B51_60.npz")["month_ev"],
            np.load(OUT.parent / "p5b_scfi" / "spotcheck_B61_70.npz")["month_ev"]])
        # monthly delta vs canonical
        md = {}
        for mm in np.unique(m_all_ev):
            mk = m_all_ev == mm
            md[int(mm)] = cosine(p_all[mk], y_all[mk]) - cosine(ev_canon[mk], y_all[mk])
        pos = sum(1 for v in md.values() if v > 0)
        cons[key] = {
            "blend_delta_avg": bd, "B51_60": b1["blend_delta"], "B61_70": b2["blend_delta"],
            "cos_B51_60": b1["cos"], "cos_B61_70": b2["cos"],
            "corr_canon_avg": (b1["corr_canon"] + b2["corr_canon"]) / 2,
            "monthly_pos_vs_canon": pos, "monthly_n": len(md),
        }
        print(f"\n{key}: blendΔ={bd:+.6f} (B1 {b1['blend_delta']:+.6f}/B2 {b2['blend_delta']:+.6f}) "
              f"months>canon {pos}/{len(md)} corr={cons[key]['corr_canon_avg']:.3f}", flush=True)

    results["consolidated"] = cons
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    np.savez(OUT / "p5d_preds.npz",
             **{f"p_{k.replace('_', '')}": np.concatenate([all_preds[k]["B51_60"], all_preds[k]["B61_70"]])
                for k in all_preds})
    print(f"\ntotal {time.time()-t0:.0f}s → {OUT}")


if __name__ == "__main__":
    main()
