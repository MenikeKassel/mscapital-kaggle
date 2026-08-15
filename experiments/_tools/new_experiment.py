# -*- coding: utf-8 -*-
"""MSCapital 新实验分配器 (Experiment ID v1.0 唯一正式入口).

用法:
  python experiments/_tools/new_experiment.py P5 market-volatility-residual [--parent P5-01]
  python experiments/_tools/new_experiment.py --arm P5-09 cosine

规则 (任务书 §15/§16/§17/§37):
- monotonic: max(ever)+1, 不回填空洞
- ID 永不回收/复用
- arm 只能在已存在 canonical parent 下分配
- 注册后 status=planned, decision=NA
"""
import argparse, csv, json, os, re, sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根 (experiments/_tools → ×3)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src"))
from mscapital.experiment_registry import (
    load_registry, load_meta, allocate_next_id, allocate_next_arm,
    resolve_row, REGISTRY_PATH, META_PATH,
)

COLS = ["experiment_id","id_status","title","name","phase","created_at","status","decision",
        "parent","successor","aliases","tags","data_market","data_order","data_transaction",
        "base_model","objective","validation","baseline_score","score","delta","public_lb",
        "conclusion","failure_reason","do_not_repeat","script_path","report_path","artifact_path"]

PHASE_DIRS = {
    "P0": "P0_protocol", "P1": "P1_representation", "P2": "P2_calibration",
    "P3": "P3_nextgen", "P4": "P4_hidden_info", "P5": "P5_market",
    "P6": "P6_production", "P6R": "P6_production", "P7": "P7_amplitude",
    "C": "C_clean_baseline", "E": "E_conditional", "M": "M_representation", "S": "S_submissions",
}

def main():
    ap = argparse.ArgumentParser(description="MSCapital 新实验分配器 (No ID, No Experiment)")
    ap.add_argument("series_or_arm", nargs="?", help="series (P5) 或 --arm 的 parent")
    ap.add_argument("name", nargs="?", help="实验 slug 名 (ASCII)")
    ap.add_argument("--arm", action="store_true", help="分配 arm (parent 的变体)")
    ap.add_argument("--parent", default="", help="parent canonical ID")
    ap.add_argument("--title", default="", help="中文标题 (可选)")
    ap.add_argument("--tags", default="", help="tags (| 分隔, 可选)")
    ap.add_argument("--registry", default=REGISTRY_PATH, help="registry 路径 (测试用)")
    args = ap.parse_args()

    reg = load_registry(args.registry)
    if args.arm:
        if not args.series_or_arm:
            ap.error("--arm 需要 parent ID")
        new_id = allocate_next_arm(reg, args.series_or_arm)
        parent = resolve_row(args.series_or_arm, reg)["experiment_id"]
        series = parent.split("-")[0]
        slug = args.name or "arm"
    else:
        series = args.series_or_arm
        if series not in ("P0","P1","P2","P3","P4","P5","P6","P6R","P7","C","E","M","S"):
            ap.error(f"invalid series: {series}")
        if not args.name or not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,59}", args.name):
            ap.error("name 必须为 ASCII slug (小写字母/数字/连字符)")
        new_id = allocate_next_id(reg, series)
        parent = args.parent
        slug = args.name
        if parent:
            resolve_row(parent, reg)  # parent 必须存在

    if args.registry != REGISTRY_PATH:
        print(f"[TEST MODE] allocated: {new_id} (registry={args.registry})")
        return

    # 写 registry (追加)
    row = {c: "" for c in COLS}
    row.update({
        "experiment_id": new_id, "id_status": "canonical",
        "title": args.title or slug.replace("-", " ").title(),
        "name": slug, "phase": {"P6R": "P6_production"}.get(series, PHASE_DIRS.get(series, series)),
        "created_at": "2026-08-15", "status": "planned", "decision": "NA",
        "parent": parent, "aliases": "", "tags": args.tags,
    })
    with open(args.registry, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writerow(row)

    # 建目录 + README 骨架
    d = os.path.join(BASE, "experiments", PHASE_DIRS.get(series, series), f"{new_id}_{slug}")
    os.makedirs(os.path.join(d, "results"), exist_ok=True)
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    readme = f"""# {new_id} — {row['title']}

## Metadata

- ID: `{new_id}` (canonical)
- Phase: {row['phase']} | Created: {row['created_at']}
- Status: `planned` | Decision: NA
- Parent: {parent or '-'} | Successor: -
- Tags: {args.tags or '-'}

## Research Question

(待填写 — 必须是可证伪命题)

## Hypothesis

(待填写)

## Motivation

(来自: 上一个实验结论 / EDA / 外部方法 / leaderboard 逆向 / 模型诊断 / 理论推导)

## Data

(待填写)

## Protocol / Validation

(待填写 — 必须严格 temporal + 嵌套)

## Arms

(如无 arm 删除本节)

## Results

| | 值 |
|---|---|
| Baseline |  |
| Score |  |
| Delta |  |
| Public LB |  |

## Decision

NA (尚未执行)

## Failure Mechanism / Do Not Repeat

(完成后填写)

## Successor

(完成后填写)

## Scripts / Reports / Artifacts

(执行时填写)
"""
    with open(os.path.join(d, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"Allocated: {new_id}")
    print(f"Created: {os.path.relpath(d, BASE)}/ (README.md, results/, logs/)")
    print(f"Registry: {new_id} added as planned")
    print(f"Parent: {parent or '-'}")
    print("Validation: PASS")

if __name__ == "__main__":
    main()
