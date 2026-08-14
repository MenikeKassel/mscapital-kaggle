# -*- coding: utf-8 -*-
"""P5-B — SCFI Tabular Conditional Innovation Probe (任务书 2026-08-14, Step 4).

Hypothesis (§5.1): 事件流的价值在"给定市场状态下的 surprise"
  Z = (observed − E[observed|state]) / robust_scale,
  而非 absolute level。状态条件化应带来更稳定、更可用的表示。

协议:
  - 下游模型: LightGBM (B1 官方参数), 每 arm 同架构/同种子/同早停,
    只改变 feature representation (§5.10)
  - temporal blocks (完全相同的 split, §5.8):
      block B1: train 0-48, holdout 49-50 (早停+blend 调权), eval 51-60
      block B2: train 0-58, holdout 59-60, eval 61-70
  - nuisance (§5.4-5.5): Ridge (multi-output), M→O 与 (M,O)→T, 全部 cross-fit:
      train 内: 10 个月 fold-block (fit on other blocks, predict block)
      eval:     fit on ALL train months → predict eval (严格 OOF)
    Z 的 scale = train 折外残差的 MAD (绝不使用 eval 残差 → 无目标泄漏)
  - M = m05 ReVol-lite market_state 16 特征 (E02 已验证的 state 集)

五臂 (§5.8-5.9, 维度匹配):
  A: f0726 152
  B: A + raw O/T aggregates (~57)
  C: A + Z_O + Z_T (~57)
  D: A + raw + Z (~114 增量)
  E: A + raw + (z-score)² (~114 增量, capacity control, 与 D 增量数匹配)

判定 (§5.12-5.13): 见文末 decision 逻辑
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

RAW = Path(r"D:\mscapital-forecasting\data\raw\train")
PROC = Path(r"D:\mscapital-forecasting\data\processed")
OUT = Path(r"D:\mscapital-kaggle\output\p5b_scfi")
OUT.mkdir(parents=True, exist_ok=True)
N_THREADS = 8

LGB_PARAMS = dict(
    objective="regression", metric="rmse",
    learning_rate=0.02, num_leaves=32, min_data_in_leaf=300,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    lambda_l2=5.0, max_bin=255, verbose=-1, num_threads=N_THREADS, seed=0)


def cosine(p: np.ndarray, y: np.ndarray) -> float:
    num = float(np.dot(p, y))
    den = float(np.sqrt(np.dot(p, p) * np.dot(y, y)))
    return num / den if den > 0 else 0.0


def rms_norm(p: np.ndarray) -> np.ndarray:
    s = float(np.sqrt(np.mean(p ** 2)))
    return p / s if s > 0 else p


def mad(arr: np.ndarray) -> float:
    return float(np.median(np.abs(arr - np.median(arr)))) + 1e-9


def load_data():
    lab = pl.read_ipc(RAW / "label.feather").sort("sample_id")
    f726 = pl.read_parquet(PROC / "f0726_train_f32.parquet").sort("sample_id")
    rawagg = (pl.read_parquet(OUT / "raw_ot_agg.parquet").sort("sample_id")
              .with_columns(pl.all().fill_null(0.0))  # 单事件样本的 std/quantile null → 0
              .with_columns(pl.col("sample_id").cast(pl.Int32)))
    mstate0 = pl.read_parquet(r"D:\mscapital-kaggle\output\m05_state\market_state_train.parquet").sort("sample_id")
    # mstate 特征名与 f0726 重复 → 加 st_ 前缀, 避免 join 后缀混乱且选错列;
    # mstate 含 NaN (15s 窗无数据, polars null_count 不统计 NaN) → fill_nan(0)
    mstate = (mstate0
              .rename({c: f"st_{c}" for c in mstate0.columns
                       if c not in ("sample_id", "month", "target")})
              .with_columns(pl.col("sample_id").cast(pl.Int32))
              .with_columns([pl.col(f"st_{c}").fill_nan(0.0) for c in mstate0.columns
                             if c not in ("sample_id", "month", "target")]))

    df = lab.join(f726, on="sample_id", how="left").join(rawagg, on="sample_id", how="left") \
            .join(mstate.drop(["month", "target"]), on="sample_id", how="left")
    assert df.height == lab.height, "join broken"
    # f0726 含 ~2% NaN (项目已知, P6R 同款); LightGBM 原生处理 NaN 分支 →
    # X152 允许 NaN, 所有 arm 相同处理 (公平比较)。raw/Z/M 必须有限。
    fe_raw_cols = [c for c in rawagg.columns if c.startswith(("ob_", "tb_"))]
    fe_m_cols = [f"st_{c}" for c in mstate0.columns if c not in ("sample_id", "month", "target")]
    Xraw_df = df.select(fe_raw_cols)
    assert Xraw_df.null_count().sum_horizontal().item() == 0, "rawagg nulls remain"
    Xm_df = df.select(fe_m_cols)
    assert Xm_df.null_count().sum_horizontal().item() == 0, "mstate nulls"
    assert np.isfinite(Xraw_df.to_numpy()).all() and np.isfinite(Xm_df.to_numpy()).all(), "non-finite raw/M"

    FE_152 = [c for c in f726.columns if c != "sample_id" and c != "target"]
    FE_RAW = [c for c in rawagg.columns if c.startswith(("ob_", "tb_"))]
    FE_M = fe_m_cols

    return {
        "month": df["month"].to_numpy(),
        "sample_id": df["sample_id"].to_numpy(),
        "target": df["target"].to_numpy().astype(np.float64),
        "X152": df.select(FE_152).to_numpy().astype(np.float32),
        "Xraw": df.select(FE_RAW).to_numpy().astype(np.float64),
        "Xm": df.select(FE_M).to_numpy().astype(np.float64),
        "fe152": FE_152, "feraw": FE_RAW, "fem": FE_M,
    }


def ridge_fit_predict(Xtr, Ytr, Xva, alpha=1.0):
    """Multi-output Ridge on standardized X; y clipped ±6·MAD (train only)."""
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xva_s = sc.transform(Xva)
    Ytr_c = np.clip(Ytr, -6 * mad(Ytr.ravel()), 6 * mad(Ytr.ravel()))
    r = Ridge(alpha=alpha)
    r.fit(Xtr_s, Ytr_c)
    return r.predict(Xva_s)


def nuisance_crossfit(data, months_train, months_eval, fold=10):
    """Ridge cross-fit: M→O then (M,O)→T.
    Returns Z (standardized surprise) for train (fold-OOF), holdout-ready scale
    info, and eval (full-train fit). Z scale = MAD of train fold-OOF residuals."""
    m = data["month"]
    Xm_all = data["Xm"]
    Xraw = data["Xraw"]
    o_cols = [i for i, c in enumerate(data["feraw"]) if c.startswith("ob_")]
    t_cols = [i for i, c in enumerate(data["feraw"]) if c.startswith("tb_")]
    O = Xraw[:, o_cols]
    T = Xraw[:, t_cols]
    nO, nT = O.shape[1], T.shape[1]

    tr_mask = np.isin(m, months_train)
    ev_mask = np.isin(m, months_eval)
    mtr = m[tr_mask]
    n_tr = int(tr_mask.sum())

    fold_starts = np.arange(months_train[0], months_train[-1] + 1, fold)
    folds = [(s, min(s + fold - 1, months_train[-1])) for s in fold_starts]

    resid_tr_O = np.full((n_tr, nO), np.nan)
    resid_tr_T = np.full((n_tr, nT), np.nan)
    for (s, e) in folds:
        f_mask = (mtr >= s) & (mtr <= e)
        o_mask = ~f_mask
        Xf = Xm_all[tr_mask][o_mask]
        Xv = Xm_all[tr_mask][f_mask]
        pred = ridge_fit_predict(Xf, O[tr_mask][o_mask], Xv)
        resid_tr_O[f_mask] = O[tr_mask][f_mask] - pred
        Xfo = np.hstack([Xf, O[tr_mask][o_mask]])
        Xvo = np.hstack([Xv, O[tr_mask][f_mask]])
        pred = ridge_fit_predict(Xfo, T[tr_mask][o_mask], Xvo)
        resid_tr_T[f_mask] = T[tr_mask][f_mask] - pred

    madO = np.array([mad(resid_tr_O[:, j]) for j in range(nO)])
    madT = np.array([mad(resid_tr_T[:, j]) for j in range(nT)])
    Ztr = np.hstack([resid_tr_O / madO, resid_tr_T / madT])

    # R2 (fold-OOF, train scale)
    R2 = {}
    for j in range(nO):
        v0 = float(np.var(O[tr_mask, j]))
        R2[f"O_{data['feraw'][o_cols[j]]}"] = 1.0 - float(np.var(resid_tr_O[:, j])) / v0 if v0 > 0 else 0.0
    for j in range(nT):
        v0 = float(np.var(T[tr_mask, j]))
        R2[f"T_{data['feraw'][t_cols[j]]}"] = 1.0 - float(np.var(resid_tr_T[:, j])) / v0 if v0 > 0 else 0.0

    # eval: fit on all train months
    Xf = Xm_all[tr_mask]
    Xe = Xm_all[ev_mask]
    pred_e_O = ridge_fit_predict(Xf, O[tr_mask], Xe)
    Ze_O = (O[ev_mask] - pred_e_O) / madO
    Xfo = np.hstack([Xf, O[tr_mask]])
    Xeo = np.hstack([Xe, O[ev_mask]])
    pred_e_T = ridge_fit_predict(Xfo, T[tr_mask], Xeo)
    Ze_T = (T[ev_mask] - pred_e_T) / madT
    Ze = np.hstack([Ze_O, Ze_T])

    return Ztr, Ze, R2, (o_cols, t_cols), (madO, madT)


def fit_lgb(Xtr, ytr, Xho, yho, Xev, Xho_out):
    dtr = lgb.Dataset(Xtr, ytr)
    dho = lgb.Dataset(Xho, yho, reference=dtr)
    model = lgb.train(LGB_PARAMS, dtr, num_boost_round=2000,
                      valid_sets=[dho],
                      callbacks=[lgb.early_stopping(200)])
    return model.predict(Xev, num_iteration=model.best_iteration), \
           model.predict(Xho_out, num_iteration=model.best_iteration), model


def main() -> None:
    t0 = time.time()
    data = load_data()
    print(f"loaded: months {data['month'].min()}-{data['month'].max()} "
          f"n={len(data['month']):,} 152f={len(data['fe152'])} rawf={len(data['feraw'])} "
          f"Mf={len(data['fem'])} ({time.time()-t0:.0f}s)", flush=True)

    canon = np.load(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
    cids = canon["sample_id"]
    cpos = np.searchsorted(cids, data["sample_id"])
    ok = np.searchsorted(cids, data["sample_id"], side="right") > cpos
    canon_pred = np.full(len(data["sample_id"]), np.nan)
    canon_pred[ok] = canon["baseline_oof"][cpos[ok]]
    data["canon"] = canon_pred

    blocks = [
        dict(name="B51_60", tr=np.arange(0, 49), ho=np.arange(49, 51), ev=np.arange(51, 61)),
        dict(name="B61_70", tr=np.arange(0, 59), ho=np.arange(59, 61), ev=np.arange(61, 71)),
    ]

    results = {"arms": {a: {} for a in "ABCDE"}, "blocks": {}, "nuisance": {}}
    all_monthly = {a: [] for a in "ABCDE"}
    all_preds = {}
    top_importances = {}

    for blk in blocks:
        print(f"\n===== block {blk['name']} =====", flush=True)
        m = data["month"]
        tr_mask = np.isin(m, blk["tr"])
        ho_mask = np.isin(m, blk["ho"])
        ev_mask = np.isin(m, blk["ev"])

        Ztr, Ze, R2, (o_cols, t_cols), (madO, madT) = nuisance_crossfit(data, blk["tr"], blk["ev"])
        r2o = [v for k, v in R2.items() if k.startswith("O_")]
        r2t = [v for k, v in R2.items() if k.startswith("T_")]
        results["nuisance"][blk["name"]] = {
            "R2_mean_O": float(np.mean(r2o)), "R2_mean_T": float(np.mean(r2t)),
            "R2_max": float(max(R2.values())),
            "R2_frac_gt_0.05": float(np.mean([v for v in R2.values() if v > 0.05])),
            "per_feature": R2,
        }
        print(f"  nuisance R2: O={results['nuisance'][blk['name']]['R2_mean_O']:.4f} "
              f"T={results['nuisance'][blk['name']]['R2_mean_T']:.4f} "
              f"max={results['nuisance'][blk['name']]['R2_max']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

        # holdout Z: fit on tr only, scale by train MADs
        Xf = data["Xm"][tr_mask]
        Xh = data["Xm"][ho_mask]
        Otr, Thr = data["Xraw"][tr_mask][:, o_cols], data["Xraw"][tr_mask][:, t_cols]
        Oho, Tho = data["Xraw"][ho_mask][:, o_cols], data["Xraw"][ho_mask][:, t_cols]
        pred_ho_O = ridge_fit_predict(Xf, Otr, Xh)
        Zh_O = (Oho - pred_ho_O) / madO
        Xfo = np.hstack([Xf, Otr])
        Xho2 = np.hstack([Xh, Oho])
        pred_ho_T = ridge_fit_predict(Xfo, Thr, Xho2)
        Zh_T = (Tho - pred_ho_T) / madT
        Zh = np.hstack([Zh_O, Zh_T])

        # assemble Z over all three windows
        nO = len(o_cols)
        Z = np.full((len(m), nO + len(t_cols)), np.nan)
        Z[tr_mask] = Ztr
        Z[ho_mask] = Zh
        Z[ev_mask] = Ze

        # capacity control: standardized square of raw (train stats)
        mu_tr = data["Xraw"][tr_mask].mean(axis=0)
        sd_tr = data["Xraw"][tr_mask].std(axis=0) + 1e-9
        RAW2 = ((data["Xraw"] - mu_tr) / sd_tr) ** 2

        feat_sets = {
            "A": [data["X152"]],
            "B": [data["X152"], data["Xraw"]],
            "C": [data["X152"], Z],
            "D": [data["X152"], data["Xraw"], Z],
            "E": [data["X152"], data["Xraw"], RAW2],
        }
        ytr = data["target"][tr_mask]
        yho = data["target"][ho_mask]
        yev = data["target"][ev_mask]
        p_canon = canon_pred[ev_mask]
        p_canon_ho = canon_pred[ho_mask]

        blk_imp = {}
        for arm, parts in feat_sets.items():
            Xtr = np.hstack(parts)[tr_mask].astype(np.float32)
            Xho = np.hstack(parts)[ho_mask].astype(np.float32)
            Xev = np.hstack(parts)[ev_mask].astype(np.float32)
            p_ev, p_ho, model = fit_lgb(Xtr, ytr, Xho, yho, Xev, Xho)
            all_preds.setdefault(arm, {})[blk["name"]] = p_ev
            # blend weight tuned on holdout ONLY
            best_w, best_s = 0.0, -1e9
            for w in np.arange(0.05, 0.51, 0.05):
                s = cosine(rms_norm(p_canon_ho) + w * rms_norm(p_ho), yho)
                if s > best_s:
                    best_s, best_w = s, float(w)
            p_blend = rms_norm(p_canon) + best_w * rms_norm(p_ev)
            results["arms"][arm][blk["name"]] = {
                "cos": cosine(p_ev, yev),
                "cos_canon": cosine(p_canon, yev),
                "blend_w": best_w,
                "blend_delta": cosine(p_blend, yev) - cosine(p_canon, yev),
            }
            print(f"  arm {arm}: cos={results['arms'][arm][blk['name']]['cos']:.6f} "
                  f"canon={cosine(p_canon, yev):.6f} "
                  f"blend_w={best_w:.2f} blendΔ={results['arms'][arm][blk['name']]['blend_delta']:+.6f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if arm == "C":
                imp = sorted(zip([*data["fe152"], *[f"Z_{c}" for c in data["feraw"]]],
                                 model.feature_importance("gain")),
                             key=lambda x: -x[1])
                blk_imp["top20"] = [(n, float(v)) for n, v in imp[:20]]
                blk_imp["z_in_top20"] = [n for n, _ in imp[:20] if n.startswith("Z_")]
                blk_imp["z_rank_best"] = next((i + 1 for i, (n, _) in enumerate(imp) if n.startswith("Z_")), None)
            del Xtr, Xho, Xev, model
            gc.collect()
        top_importances[blk["name"]] = blk_imp

        for arm in "BCDE":
            results["arms"][arm][blk["name"]]["delta_vs_A"] = (
                results["arms"][arm][blk["name"]]["cos"] - results["arms"]["A"][blk["name"]]["cos"])

        ev_m = m[ev_mask]
        yev_m = data["target"][ev_mask]
        for arm in "ABCDE":
            p = all_preds[arm][blk["name"]]
            for mm in blk["ev"]:
                mmask = ev_m == mm
                if mmask.sum() < 50:
                    continue
                all_monthly[arm].append({
                    "month": int(mm),
                    "cos": cosine(p[mmask], yev_m[mmask]),
                    "delta_vs_canon": cosine(p[mmask], yev_m[mmask]) - cosine(p_canon[mmask], yev_m[mmask]),
                })

    # ---- consolidated ----
    cons = {}
    for arm in "ABCDE":
        d1 = results["arms"][arm]["B51_60"].get("delta_vs_A", 0.0)
        d2 = results["arms"][arm]["B61_70"].get("delta_vs_A", 0.0)
        bd = (results["arms"][arm]["B51_60"]["blend_delta"] + results["arms"][arm]["B61_70"]["blend_delta"]) / 2
        mt = all_monthly[arm]
        pos = sum(1 for r in mt if r["delta_vs_canon"] > 0)
        pA = np.concatenate([all_preds[arm]["B51_60"], all_preds[arm]["B61_70"]])
        pA_arm = np.concatenate([all_preds["A"]["B51_60"], all_preds["A"]["B61_70"]])
        ev_canon = np.concatenate([canon_pred[np.isin(m, blocks[0]["ev"])], canon_pred[np.isin(m, blocks[1]["ev"])]])
        cons[arm] = {
            "delta_vs_A_avg": (d1 + d2) / 2 if arm != "A" else 0.0,
            "delta_vs_A_B51_60": d1, "delta_vs_A_B61_70": d2,
            "blend_delta_avg": bd,
            "monthly_pos": pos, "monthly_n": len(mt),
            "corr_with_A": float(np.corrcoef(pA, pA_arm)[0, 1]),
            "corr_with_canon": float(np.corrcoef(pA, ev_canon)[0, 1]),
        }
        print(f"\narm {arm}: ΔvsA={cons[arm]['delta_vs_A_avg']:+.6f} "
              f"(B1 {d1:+.6f}/B2 {d2:+.6f}) blendΔ={bd:+.6f} "
              f"months {pos}/{len(mt)} corrA={cons[arm]['corr_with_A']:.3f} "
              f"corrCanon={cons[arm]['corr_with_canon']:.3f}", flush=True)

    pc = np.concatenate([all_preds["C"]["B51_60"], all_preds["C"]["B61_70"]])
    pb = np.concatenate([all_preds["B"]["B51_60"], all_preds["B"]["B61_70"]])
    cons["C"]["corr_C_B"] = float(np.corrcoef(pc, pb)[0, 1])
    cons["D"]["D_minus_E"] = cons["D"]["delta_vs_A_avg"] - cons["E"]["delta_vs_A_avg"]
    cons["D"]["D_minus_E_B1"] = cons["D"]["delta_vs_A_B51_60"] - cons["E"]["delta_vs_A_B51_60"]
    cons["D"]["D_minus_E_B2"] = cons["D"]["delta_vs_A_B61_70"] - cons["E"]["delta_vs_A_B61_70"]

    # ---- decision ----
    dC = cons["C"]["delta_vs_A_avg"]
    dDE = cons["D"]["D_minus_E"]
    late_C = cons["C"]["delta_vs_A_B61_70"]
    pos_frac = cons["C"]["monthly_pos"] / max(cons["C"]["monthly_n"], 1)
    corr_CB = cons["C"]["corr_C_B"]
    corr_Ccanon = cons["C"]["corr_with_canon"]
    blend_C = cons["C"]["blend_delta_avg"]

    crit = {
        "ΔC ≥ +0.0005 或 Δ(D−E) ≥ +0.0005": (dC >= 0.0005) or (dDE >= 0.0005),
        "late (61-70) > 0": late_C > 0,
        "≥65% months positive": pos_frac >= 0.65,
        "innovation ≠ raw (corr(C,B) < 0.95)": corr_CB < 0.95,
        "D 增益不被 capacity-control 解释 (D−E ≥ +0.0005)": dDE >= 0.0005,
    }
    if dC >= 0.0005 or dDE >= 0.0005:
        if corr_Ccanon < 0.80 and blend_C >= 0.0007:
            decision = "SCFI STRONG LIVE" if blend_C >= 0.0015 else "SCFI LIVE"
        else:
            decision = "SCFI CONTINUE (arm delta ok, blend/orthogonality below LIVE bar)"
    elif dC >= 0.0002 or dDE >= 0.0002:
        decision = "INCONCLUSIVE"
    else:
        decision = "KILL"
    cons["decision"] = decision
    cons["criteria"] = crit
    cons["importance_C"] = top_importances

    results["consolidated"] = cons
    results["monthly"] = all_monthly
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    ev_all = np.concatenate([m[np.isin(m, blocks[0]["ev"])], m[np.isin(m, blocks[1]["ev"])]])
    np.savez(OUT / "p5b_preds.npz",
             **{f"p_{a}": np.concatenate([all_preds[a]["B51_60"], all_preds[a]["B61_70"]]) for a in "ABCDE"},
             month=ev_all,
             y=np.concatenate([data["target"][np.isin(m, blocks[0]["ev"])], data["target"][np.isin(m, blocks[1]["ev"])]]),
             canon=np.concatenate([canon_pred[np.isin(m, blocks[0]["ev"])], canon_pred[np.isin(m, blocks[1]["ev"])]]))

    print("\n" + "=" * 78)
    print(f"P5-B SCFI: ΔC={dC:+.6f} Δ(D−E)={dDE:+.6f} late_C={late_C:+.6f} "
          f"months={cons['C']['monthly_pos']}/{cons['C']['monthly_n']} "
          f"corr(C,B)={corr_CB:.3f} corr(C,canon)={corr_Ccanon:.3f} blend_C={blend_C:+.6f}")
    print(f"  decision: {decision}")
    print("=" * 78)
    print(f"\ntotal {time.time()-t0:.0f}s → {OUT}")


if __name__ == "__main__":
    main()
