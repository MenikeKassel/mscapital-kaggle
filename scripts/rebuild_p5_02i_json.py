# -*- coding: utf-8 -*-
"""Rebuild arms results from run.log (M0-M6 done, M7 crashed before json dump)."""
import json
import re
from pathlib import Path

log = Path(r"D:\mscapital-kaggle\output\p5_02i_info_audit\run.log").read_text(encoding="utf-8", errors="replace")
arms = {}
pat = re.compile(r"^  \[(\w+)\] corr_y=([+-][\d.]+) corr_v7=([+-][\d.]+) alpha=([\d.]+) "
                 r"frozen_delta=([+-][\d.]+) monthly=(\d+)/(\d+) lo=([+-][\d.]+) hi=([+-][\d.]+)")
for line in log.splitlines():
    m = pat.match(line)
    if m:
        name, cy, cv, a, fd, mp, mn, lo, hi = m.groups()
        arms[name] = {"alpha": float(a), "corr_y": float(cy), "corr_v7": float(cv),
                      "delta_frozen": float(fd), "monthly_pos": int(mp), "monthly_n": int(mn),
                      "monthly_mean": float(fd), "delta_lo": float(lo), "delta_mid": 0.0,
                      "delta_hi": float(hi)}
print("parsed arms:", list(arms.keys()))
assert set(arms) == {"raw", "shuffle", "reverse", "block10", "block20", "block50", "desync"}, set(arms) - {"raw", "shuffle", "reverse", "block10", "block20", "block50", "desync"}
json.dump({"arms": arms, "probes": {}}, open(r"D:\mscapital-kaggle\output\p5_02i_info_audit\results.json", "w", encoding="utf-8"), indent=2)
print("results.json written:", len(arms), "arms")
