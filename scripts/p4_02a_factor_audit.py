# -*- coding: utf-8 -*-
"""P4-02A: LB142 v10 factors reverse engineering - name-level audit.

Extract the 217 v10 feature names from fulldata.pt, classify by origin
(raw state / 60s geometry / activity / normalized / grid / cross-scale /
auto-generated interaction / suspicious), and audit overlap with our
152 dynamics features (0726).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

CKPT = Path(r"D:\mscapital-forecasting\reference\lb142\weights\v10\fulldata.pt")
OUT = Path(r"D:\mscapital-kaggle\output\p4_02_factors")
OUT.mkdir(parents=True, exist_ok=True)

# our 152 features (0726)
from mscapital.features.revol_lite import revol_lite_feature_names  # noqa: E402
F0726_FEATS = sorted({f[0] for f in [()]}) if False else None  # placeholder

import polars as pl  # noqa: E402
fe = pl.read_parquet(r"D:\mscapital-kaggle\scripts\kaggle_0726ds\f0726_train_f32.parquet")
OURS = [c for c in fe.columns if c not in ("sample_id", "target")]
print(f"our 152-feature stack: {len(OURS)} features")

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
feats = list(ck.get("feats") or [])
print(f"v10 feats: {len(feats)}\n")

RECIPE = re.compile(r"^(c2|r2|r9|r63|r106)_(mul|sub|abs_diff|max|mean|div|add|min|std|ewm)_?")
PREFIX = re.compile(r"^(t|o|m|x|c2|r2|r9|r63|r106)_")


def classify(name: str) -> str:
    if name.startswith(("c2_", "r2_", "r9_", "r63_", "r106_")):
        return "G_auto_interaction"
    if "__" in name:
        return "G_auto_interaction"
    if name.startswith("m_"):
        return "A_market_state"
    if name.startswith("t_") or name.startswith("o_"):
        base = name[2:]
        if any(k in base for k in ("ratio", "signed", "imb", "sd", "ewm", "mean", "std", "median")):
            return "D_normalized"
        if any(k in base for k in ("slope", "range", "high", "low", "reversal", "first", "last", "max", "min")):
            return "B_path_geometry"
        return "A_raw_state"
    if name.startswith("x_"):
        return "F_cross_scale"
    return "H_other"


cls = {c: 0 for c in ["A_market_state", "A_raw_state", "B_path_geometry", "D_normalized",
                       "F_cross_scale", "G_auto_interaction", "H_other"]}
for f in feats:
    cls[classify(f)] += 1
print("=== v10 217 feats by class ===")
for c, n in cls.items():
    print(f"  {c:24s} {n}")

# overlap with ours: exact name match + normalized-name match
ours_set = set(OURS)
exact = [f for f in feats if f in ours_set]
print(f"\nexact name overlap: {len(exact)}/217")
for f in exact[:10]:
    print("   ", f)

# semantic overlap: strip windows & interaction part, compare base tokens
def base_tokens(name: str) -> set[str]:
    name = re.sub(r"_\d+$", "", name)
    name = re.sub(r"_\d+_\d+$", "", name)
    parts = re.split(r"__(?!_)", name)
    toks = set()
    for p in parts:
        p = re.sub(r"^(c2|r2|r9|r63|r106)_(mul|sub|abs_diff|max|mean|div|add|min|std|ewm|sum)_?", "", p)
        toks.add(p)
    return toks

our_tokens = set()
for f in OURS:
    our_tokens |= base_tokens(f)
sem = []
for f in feats:
    toks = base_tokens(f)
    hit = toks & our_tokens
    sem.append((f, len(hit), hit))
no_overlap = [s for s in sem if s[1] == 0]
print(f"\nsemantic overlap: {len(feats) - len(no_overlap)}/217 share >=1 base token with ours")
print(f"no-overlap factors (candidates for new info): {len(no_overlap)}")
for f, _, _ in no_overlap[:25]:
    print("   ", f)

# window audit
wins = sorted({int(m.group(1)) for f in feats for m in [re.search(r"_(\d+)$", f)] if m})
print(f"\nwindow sizes present: {wins}")
# suspicious
susp = [f for f in feats if re.search(r"(month|sample|id|group|day|hour|seq|rank|index)", f, re.I)]
print(f"suspicious-looking: {len(susp)}")

# dump
with open(OUT / "v10_feats.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(feats))
print("\ndumped to", OUT / "v10_feats.txt")
