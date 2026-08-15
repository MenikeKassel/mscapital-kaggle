# -*- coding: utf-8 -*-
"""一次性修正 experiment_data.py 的实验 ID (统一编号规则, 2026-08-15).
规则:
- P{n}-{NN}{suffix}: 正式实验 (P0-01, P1-01a, P5-02I, P6R-00, P7-AMP)
- 早期表格实验保留字母序号 A1..H1, 重名/易混者加消歧后缀 (C1-FE, E1-TW, B1-LGO)
- 提交事件独立命名空间: SUB-v4/SUB-v5/SUB-v7/SUB-v8 (与 submissions/README.md 对应)
- 历史编号写入 alias 字段保留可追溯
"""
import re, os

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_data.py")
src = open(P, encoding="utf-8").read()

# (旧ID, 新ID, alias)
MAP = [
    ("B1_ab", "B1-LGO", "B1 特征组消融 (RESULTS.md 历史名)"),
    ("C1_feat", "C1-FE", "C1 增强窗口特征 (RESULTS.md 历史名; 与 C1 Clean-RealMLP 区分)"),
    ("D1_hp", "D1", "D1 超参单变量"),
    ("E1_tw", "E1-TW", "E1 时间衰减 (与 E01 ReVol-lite 区分)"),
    ("F1_mlp", "F1", "F1 轻量 MLP"),
    ("F2_ens", "F2", "F2 MLP 集成"),
    ("H1_5m", "H1", "H1 五模型 (与 P4-H1H2 假设区分)"),
    ("V4", "SUB-v4", "提交 v4 (RESULTS.md)"),
    ("V5", "SUB-v5", "提交 v5 (RESULTS.md)"),
    ("V7", "SUB-v7", "提交 v7 (RESULTS.md)"),
    ("V8", "SUB-v8", "提交 v8/v8b (RESULTS.md)"),
    ("P2-cal", "P2", "P2 校准 (Codex 阶段)"),
]

for old, new, alias in MAP:
    # 只替换 id="..." 字段 (不碰 name/next 等文字)
    pat = f'id="{old}"'
    assert pat in src, f"NOT FOUND: {pat}"
    src = src.replace(pat, f'id="{new}"')
    # 加 alias 字段: 插到该 dict 的 "date" 之后
    if alias:
        src = src.replace(f'id="{new}", phase=', f'id="{new}", alias="{alias}", phase=', 1)

# P0-01/P1-01a 历史简写 alias (无重名, 但标注历史名)
for new, old in [("P0-01", "P0-1"), ("P0-02", "P0-2"), ("P0-03", "P0-3"),
                 ("P1-01a", "P1-1a"), ("P1-01b", "P1-1b"), ("P1-01c", "P1-1c"), ("P1-01e", "P1-1e")]:
    pat = f'id="{new}", phase='
    assert pat in src, f"NOT FOUND: {pat}"
    src = src.replace(pat, f'id="{new}", alias="{old} (历史简写)", phase=', 1)

open(P, "w", encoding="utf-8").write(src)
print("experiment_data.py updated:", len(MAP) + 7, "ID changes")
