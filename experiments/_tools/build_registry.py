# -*- coding: utf-8 -*-
"""生成 experiments/registry.csv + 每个实验的 README.md (Phase C/D/G)."""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_data import EXPERIMENTS

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # experiments/
CSV_COLS = ["experiment_id","phase","name","date","status","data_market","data_order",
            "data_transaction","base_model","objective","validation","baseline_score",
            "score","delta","public_lb","conclusion","failure_reason","successor",
            "script_path","report_path","artifact_path"]

# phase → 目录名
PHASE_DIR = {
    "baseline": "baseline", "P0_protocol": "P0_protocol", "P1_representation": "P1_representation",
    "P2_calibration": "P2_calibration", "C_clean": "C_clean_baseline", "P3_nextgen": "P3_nextgen",
    "P4_hidden": "P4_hidden_info", "M_representation": "M_representation",
    "E_conditional": "E_conditional", "P5_market": "P5_market",
    "P6_production": "P6_production", "P7_amplitude": "P7_amplitude",
}
PHASE_TITLE = {
    "baseline": "Baseline 表格阶梯 (2026-08-10/11)",
    "P0_protocol": "P0 Protocol 验证 (2026-08-11)",
    "P1_representation": "P1 表示与序列 (2026-08-11/12)",
    "P2_calibration": "P2 校准 (2026-08-12)",
    "C_clean": "C 系列 Clean Baseline v2 (2026-08-13)",
    "P3_nextgen": "P3 下一代方法 (2026-08-14)",
    "P4_hidden": "P4 隐藏信息调查 (2026-08-13/14)",
    "M_representation": "M 系列残差表示 (2026-08-13)",
    "E_conditional": "E 系列状态条件化 (2026-08-13)",
    "P5_market": "P5 市场探针 (2026-08-14/15)",
    "P6_production": "P6/P6R 生产与检索 (2026-08-14/15)",
    "P7_amplitude": "P7 幅度门控终裁 (2026-08-15)",
}

README_TEMPLATE = """# {id} — {name}

> 阶段: {phase_title} | 日期: {date} | 状态: **{decision}**
> 生成: 2026-08-15 仓库工程化整理 (Phase G), 数据来源 RESULTS.md / 各阶段报告

## 研究问题 (可证伪命题)
{question}

## Hypothesis
{hypothesis}

## Motivation
{motivation}

## Data
{data}

## Validation Protocol
{protocol}

## Method
{method}

## Baseline / Result / Delta
| | 值 |
|---|---|
| Baseline | {baseline} |
| 实验分数 | {score} |
| Delta | {delta} |
| Public LB | {lb} |

## Decision
**{decision}**

## Failure Analysis
{failure}

## Do Not Repeat
{do_not_repeat}

## Conclusion / Next
- Conclusion: 见阶段报告
- Next: {next}

## 复现入口
- Scripts: `{scripts}`
- Outputs: `{outputs}`
- Reports: `{reports}`
"""

def main():
    os.makedirs(os.path.join(BASE, "_unclassified"), exist_ok=True)
    rows = []
    for e in EXPERIMENTS:
        eid = e["id"]
        ph = e["phase"]
        d = e["decision"]
        rows.append({
            "experiment_id": eid, "phase": ph, "name": e["name"], "date": e["date"],
            "status": d,
            "data_market": "1" if "market" in e["data"] or "LOB" in e["data"] else "",
            "data_order": "1" if "order" in e["data"] else "",
            "data_transaction": "1" if "transaction" in e["data"] else "",
            "base_model": e["method"].split("(")[0][:60],
            "objective": "cosine (全局)",
            "validation": e["protocol"].replace(",", ";"),
            "baseline_score": e["baseline"], "score": e["score"], "delta": e["delta"],
            "public_lb": e["lb"], "conclusion": e["do_not_repeat"][:80],
            "failure_reason": e["failure"].replace(",", ";")[:120],
            "successor": e["next"], "script_path": e["scripts"], "report_path": e["reports"],
            "artifact_path": e["outputs"],
        })
    with open(os.path.join(BASE, "registry.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(rows)
    # README per experiment
    for e in EXPERIMENTS:
        ph = e["phase"]
        d = os.path.join(BASE, PHASE_DIR[ph], f'{e["id"]}_{e["name"].split(" ")[0]}_')
        # sanitize dir name
        name = e["name"].replace("/", "-").replace(":", "").replace("?", "").replace("(", "").replace(")", "").replace(" ", "_")[:40]
        d = os.path.join(BASE, PHASE_DIR[ph], f'{e["id"]}_{name}')
        os.makedirs(d, exist_ok=True)
        content = README_TEMPLATE.format(
            id=e["id"], name=e["name"], phase_title=PHASE_TITLE[ph], date=e["date"],
            decision=e["decision"], question=e["question"], hypothesis=e["hypothesis"],
            motivation=e["motivation"], data=e["data"], protocol=e["protocol"],
            method=e["method"], baseline=e["baseline"], score=e["score"], delta=e["delta"],
            lb=e["lb"], failure=e["failure"] or "(无 — 实验通过/非失败)",
            do_not_repeat=e["do_not_repeat"] or "(无特别禁止项)",
            next=e["next"] or "(无直接后继 — 见 experiment-index.md)",
            scripts=e["scripts"] or "(见阶段报告)", outputs=e["outputs"] or "-",
            reports=e["reports"] or "RESULTS.md")
        with open(os.path.join(d, "README.md"), "w", encoding="utf-8") as f:
            f.write(content)
    # experiments/README.md
    with open(os.path.join(BASE, "README.md"), "w", encoding="utf-8") as f:
        f.write("# experiments/ — 实验逻辑层\n\n"
                "> 本目录是 MSCapital 实验的 **canonical 逻辑索引**: 每个正式实验一个目录 + README,\n"
                "> 总台账见 `registry.csv`。物理资产 (scripts/output/docs) 保持原位以维护可运行性,\n"
                "> README 内以相对路径指向实际位置。\n\n"
                "## 阶段目录\n\n")
        for ph, title in PHASE_TITLE.items():
            f.write(f"- `{PHASE_DIR[ph]}/` — {title}\n")
        f.write("- `_unclassified/` — 未归属文件暂存\n")
        f.write("- `_tools/` — 生成工具 (experiment_data.py / build_registry.py)\n\n")
        f.write("## 用法\n\n- 阅读: 从 `../docs/experiment-index.md` 进入\n"
                "- 机器读取: `pd.read_csv('registry.csv')`\n"
                "- 新增实验: 在 `_tools/experiment_data.py` 加条目 → 重跑 `python _tools/build_registry.py`\n")
    print(f"registry.csv: {len(rows)} rows")
    print(f"READMEs: {len(EXPERIMENTS)}")

if __name__ == "__main__":
    main()
