# -*- coding: utf-8 -*-
"""P5-E: RealMLP 精确 spot-check — 生产 learner 上 152 vs 152+Z (R61_70 outer).

决定性证据: SCFI 的 Z 增益是否在 canonical 生产 learner (RealMLP, clean-realmlp-v2a
配置) 上存活。若 C > A → SCFI 升级证据成立; 否则 SCFI 止步于表格族。

协议: run_outer(frame, "R61_70", cfg) — inner_train 0-50, inner_tune 51-60,
refit 0-60, outer_valid 61-70 (与 canonical R61_70 块完全同构)。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\mscapital-kaggle\src")

from mscapital.models.realmlp import RealMLPConfig, load_frame, run_outer

OUT = Path(r"D:\mscapital-kaggle\output\p5e_realmlp_spotcheck")
OUT.mkdir(parents=True, exist_ok=True)
LABELS = r"D:\mscapital-forecasting\data\raw\train\label.feather"
CFG = r"D:\mscapital-kaggle\configs\clean-realmlp-v2a.json"

t0 = time.time()
cfg = RealMLPConfig.from_mapping(json.loads(Path(CFG).read_text(encoding="utf-8")))
results = {}

for arm, feat_path in (("A", r"D:\mscapital-kaggle\output\p5b_scfi\spot_rmlp_A.parquet"),
                       ("C", r"D:\mscapital-kaggle\output\p5b_scfi\spot_rmlp_C.parquet")):
    print(f"\n===== RealMLP arm {arm} =====", flush=True)
    frame = load_frame(feat_path, LABELS)
    print(f"frame: {len(frame.sample_id):,} rows, {frame.features.shape[1]} features "
          f"({time.time()-t0:.0f}s)", flush=True)
    result = run_outer(frame, "R61_70", cfg, OUT / arm, experiment_id=f"spot-rmlp-{arm}")
    results[arm] = {
        "outer_cosine": result["diagnostics"]["cosine_uncentered"],
        "pearson": result["diagnostics"].get("pearson"),
        "best_progress": result["best_progress"],
        "refit_progress": result["refit_progress"],
    }
    print(f"arm {arm}: outer cosine={results[arm]['outer_cosine']:.6f} "
          f"({time.time()-t0:.0f}s)", flush=True)

(OUT / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
print("\n" + "=" * 70)
print(f"RealMLP spot-check: A={results['A']['outer_cosine']:.6f} "
      f"C={results['C']['outer_cosine']:.6f} "
      f"Δ(C-A)={results['C']['outer_cosine'] - results['A']['outer_cosine']:+.6f}")
print("=" * 70)
print(f"total {time.time()-t0:.0f}s → {OUT}")
