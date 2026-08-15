# -*- coding: utf-8 -*-
"""Experiment ID v1.0 构建器: v1 数据 + MIGRATE 增量 → schema v2 registry + READMEs + migration map.
用法: python experiments/_tools/build_registry.py [--dry-run] [--execute]
"""
import csv, json, os, re, shutil, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # experiments/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_data import EXPERIMENTS as V1
from migrate_v2 import MIGRATE

# v1 无独立条目的迁移源补充 (P4-08B~E 拆分自原 P4-08A 合并条目)
EXTRA_V1 = {
    "P4-08B": dict(phase="P4_hidden", scripts="scripts/p4_08a_unc.py scripts/48b_realmlp_cosine_prod.py",
                   outputs="output/p4_08a_unc_ablation", reports="docs/p4-hidden-information-report.md §13"),
    "P4-08A": dict(phase="P4_hidden", scripts="scripts/p4_08a_loss_ablation.py",
                   outputs="output/p4_08a_loss_ablation", reports="docs/p4-hidden-information-report.md §12"),
    "P4-08C": dict(phase="P4_hidden", scripts="scripts/p4_08c_blend.py",
                   outputs="output/submissions/submission_v9_cos_*.csv", reports="docs/p4-hidden-information-report.md §13"),
    "P4-08D": dict(phase="P4_hidden", scripts="scripts/p4_08d_simple_cosine_prod.py",
                   outputs="output/p4_08a_unc_ablation", reports="docs/p4-hidden-information-report.md §13"),
    "P4-08E": dict(phase="P4_hidden", scripts="scripts/p4_08e_v7like_check.py",
                   outputs="", reports="docs/p4-hidden-information-report.md §14"),
    "P4-02a": dict(phase="P4_hidden", scripts="scripts/p4_02a_factor_audit.py",
                   outputs="output/p4_02_factors", reports="docs/p4-hidden-information-report.md"),
    "P4-02b": dict(phase="P4_hidden", scripts="scripts/p4_02b_market_forms.py",
                   outputs="output/p4_02b_market_forms", reports="docs/p4-hidden-information-report.md"),
    "P4-02c": dict(phase="P4_hidden", scripts="scripts/p4_02c_ofi_protocol.py",
                   outputs="output/p4_02c_features output/p4_02c_ofi_protocol", reports="docs/p4-hidden-information-report.md"),
    "M02-T": dict(phase="M_representation", scripts="(src/mscapital CLI run-m02t)",
                  outputs="output/m02t_*", reports="docs/m02-t-results.md",
                  failure="几何 temporal 变体无残差增量 (与 M-02 同判)", do_not_repeat="几何特征线整体关闭"),
    "S-01": dict(phase="P1_representation", scripts="scripts/16_final_submission.py",
                 outputs="output/submissions/submission_blend_v1.csv", reports="RESULTS.md"),
    "S-02": dict(phase="P1_representation", scripts="scripts/18_final_submission_v2.py",
                 outputs="output/submissions/submission_blend_v2.csv", reports="RESULTS.md"),
    "S-03": dict(phase="P0_protocol", scripts="scripts/22_final_submission_v3.py",
                 outputs="output/submissions/submission_blend_v3.csv", reports="RESULTS.md"),
    "S-06": dict(phase="P1_representation", scripts="scripts/37_final_v6.py",
                 outputs="output/submissions/submission_blend_v6.csv", reports="RESULTS.md (N005)",
                 failure="N005: TCN test 分布外严重退化 (PSEUDO +0.004 → LB 0.082), test corr(tab,tcn)=0.03 致命预警",
                 do_not_repeat="序列模型不通过 test 侧 corr 结构验证不得进生产融合"),
}

PHASE_DIR = {
    "baseline": "baseline", "P0_protocol": "P0_protocol", "P1_representation": "P1_representation",
    "P2_calibration": "P2_calibration", "C_clean": "C_clean_baseline", "P3_nextgen": "P3_nextgen",
    "P4_hidden": "P4_hidden_info", "M_representation": "M_representation",
    "E_conditional": "E_conditional", "P5_market": "P5_market",
    "P6_production": "P6_production", "P7_amplitude": "P7_amplitude",
    "S_submissions": "S_submissions",
}
PHASE_TITLE = {
    "baseline": "Baseline 表格阶梯 (Legacy 冻结区)",
    "P0_protocol": "P0 Protocol 验证", "P1_representation": "P1 表示与序列",
    "P2_calibration": "P2 校准", "C_clean": "C 系列 Clean Baseline v2",
    "P3_nextgen": "P3 下一代方法", "P4_hidden": "P4 隐藏信息调查",
    "M_representation": "M 系列残差表示", "E_conditional": "E 系列状态条件化",
    "P5_market": "P5 市场探针", "P6_production": "P6/P6R 生产与检索",
    "P7_amplitude": "P7 幅度终裁",
}

COLS = ["experiment_id","id_status","title","name","phase","created_at","status","decision",
        "parent","successor","aliases","tags","data_market","data_order","data_transaction",
        "base_model","objective","validation","baseline_score","score","delta","public_lb",
        "conclusion","failure_reason","do_not_repeat","script_path","report_path","artifact_path"]

def main():
    dry = "--dry-run" in sys.argv
    v1_by_id = {e["id"]: e for e in V1}
    assert len(v1_by_id) == len(V1), "v1 duplicate ids"

    # 1. 合并: MIGRATE 顺序即 canonical 分配顺序 (allocator: max+1 语义由表顺序保证)
    rows = []
    mig_map = []  # Phase D: migration map rows
    allocated = {}  # old -> new
    for old, new, id_status, title, slug, parent, aliases, tags, decision, status in MIGRATE:
        v1 = dict(v1_by_id.get(old) or {})
        v1.update(EXTRA_V1.get(old, {}))
        if id_status == "canonical":
            if not re.fullmatch(r"(P[0-7]|P6R|C|E|M|S)-[0-9]{2}[a-z]?", new):
                raise SystemExit(f"CANONICAL regex violation: {new}")
        row = {
            "experiment_id": new, "id_status": id_status, "title": title,
            "name": slug,
            "phase": "S_submissions" if new.startswith("S-") else (v1.get("phase") if v1 else ""),
            "created_at": v1.get("date", "2026-08-15") if v1 else "2026-08-15",
            "status": status, "decision": decision,
            "parent": parent, "successor": "", "aliases": aliases, "tags": tags,
            "data_market": v1.get("data_market", ""), "data_order": v1.get("data_order", ""),
            "data_transaction": v1.get("data_transaction", ""),
            "base_model": v1.get("base_model", ""), "objective": v1.get("objective", "cosine (全局)"),
            "validation": v1.get("validation", ""), "baseline_score": v1.get("baseline", ""),
            "score": v1.get("score", ""), "delta": v1.get("delta", ""),
            "public_lb": v1.get("lb", ""),
            "conclusion": (v1.get("do_not_repeat") or v1.get("failure") or
                           f"Δ={v1.get('delta','?')} → {decision}")[:120],
            "failure_reason": (v1.get("failure") or "")[:150],
            "do_not_repeat": v1.get("do_not_repeat", ""),
            "script_path": v1.get("scripts", ""), "report_path": v1.get("reports", ""),
            "artifact_path": v1.get("outputs", ""),
        }
        rows.append(row)
        mig_map.append({
            "old_id": old, "new_id": new, "id_status": id_status,
            "old_aliases": aliases, "phase": row["phase"], "parent": parent,
            "migration_reason": "legacy-frozen" if id_status == "legacy" else
                               ("canonical-migrate" if old != new else "canonical-keep"),
            "title": title, "decision": decision, "status": status,
        })
        if old != new:
            allocated[old] = new

    # 2. successor 回填 (parent 反向)
    parent_of = {r["experiment_id"]: r["parent"] for r in rows}
    children = {}
    for cid, p in parent_of.items():
        if p:
            children.setdefault(p, []).append(cid)
    for r in rows:
        r["successor"] = "|".join(sorted(children.get(r["experiment_id"], [])))

    # 3. 唯一性 + alias 冲突检查
    ids = [r["experiment_id"] for r in rows]
    assert len(ids) == len(set(ids)), f"dup ids: {[i for i in set(ids) if ids.count(i)>1]}"
    alias_idx = {}
    for r in rows:
        for a in [x.strip() for x in r["aliases"].split("|") if x.strip()]:
            assert a not in alias_idx and a not in ids, f"alias collision: {a}"
            alias_idx[a] = r["experiment_id"]

    # 4. 写 registry + meta
    if not dry:
        with open(os.path.join(BASE, "registry.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows(rows)
        meta = {"schema_version": 2, "experiment_id_spec_version": "1.0",
                "generated_at": "2026-08-15", "entries": len(rows),
                "series": ["P0","P1","P2","P3","P4","P5","P6","P6R","P7","C","E","M","S"]}
        with open(os.path.join(BASE, "registry.meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        # migration map (Phase D 产物)
        with open(os.path.join(BASE, "..", "docs", "experiment-id-migration-map-2026-08-15.csv"),
                  "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(mig_map[0].keys()))
            w.writeheader()
            w.writerows(mig_map)

    # 5. READMEs (新模板, 目录 <ID>_<slug>)
    if not dry:
        for r, m in zip(rows, mig_map):
            phdir = os.path.join(BASE, PHASE_DIR.get(r["phase"], r["phase"]))
            d = os.path.join(phdir, f'{r["experiment_id"]}_{r["name"]}')
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "README.md"), "w", encoding="utf-8") as f:
                f.write(readme(r))
    print(f"rows: {len(rows)} (v1={len(V1)}, +{len(rows)-len(V1)} 补录/拆分)")
    print(f"canonical: {sum(1 for r in rows if r['id_status']=='canonical')}, "
          f"legacy: {sum(1 for r in rows if r['id_status']=='legacy')}")
    print(f"migration map: {len(mig_map)} rows -> docs/experiment-id-migration-map-2026-08-15.csv")
    print("DRY RUN (未写盘)" if dry else "WRITTEN")

def readme(r):
    ph_title = PHASE_TITLE.get(r["phase"], r["phase"])
    return f"""# {r['experiment_id']} — {r['title']}

## Metadata

- ID: `{r['experiment_id']}` ({r['id_status']})
- Phase: {r['phase']} | Created: {r['created_at']}
- Status: `{r['status']}` | Decision: **{r['decision']}**
- Parent: {r['parent'] or '-'} | Successor: {r['successor'] or '-'}
- Aliases: {r['aliases'] or '-'}
- Tags: {r['tags'] or '-'}

> 生成: 2026-08-15 Experiment ID v1.0 迁移 (registry 为 SSOT, 本文件自动生成)

## Research Question

见阶段报告 (report_path)。

## Hypothesis / Motivation / Data / Protocol / Method

见阶段报告 + registry 字段 (validation/base_model/objective/data_*)。

## Results

| | 值 |
|---|---|
| Baseline | {r['baseline_score']} |
| Score | {r['score']} |
| Delta | {r['delta']} |
| Public LB | {r['public_lb'] or '-'} |

## Decision

**{r['decision']}** ({r['status']})

## Failure Analysis

{r['failure_reason'] or '(无)'}

## Do Not Repeat

{r['do_not_repeat'] or '(无特别禁止项)'}

## 复现入口

- Scripts: `{r['script_path'] or '-'}`
- Reports: `{r['report_path'] or '-'}`
- Artifacts: `{r['artifact_path'] or '-'}`
"""

if __name__ == "__main__":
    main()
