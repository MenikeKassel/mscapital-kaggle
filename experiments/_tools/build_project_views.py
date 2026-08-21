#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deterministic generated views for the experiment SSOT (schema v3)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

def rows():
    with (ROOT / "experiments" / "registry.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def esc(s):
    return str(s or "").replace("|", "\\|").replace("\n", " ")

def table(headers, data):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(esc(x) for x in row) + " |" for row in data]
    return "\n".join(out)

def render(rows):
    routes = json.loads((ROOT / "experiments" / "routes.yaml").read_text(encoding="utf-8"))
    submissions = list(csv.DictReader((ROOT / "submissions" / "registry.csv").open(encoding="utf-8-sig", newline="")))
    files = {}
    files["all-experiments.md"] = """# 全部实验（SSOT 生成视图）\n\n> 本页由 `experiments/_tools/build_project_views.py` 生成。事实唯一来源是 `experiments/registry.csv`；`RESULTS.md` 仅为 append-only 历史日志。\n\n""" + table(["ID", "类型", "路线", "状态", "证据", "决策", "分数", "Δ", "结论"], [(r["experiment_id"], r["record_kind"], r["route_id"], r["status"], r["evidence_state"], r["decision"], r["score"], r["delta"], r["conclusion"]) for r in rows]) + "\n"
    timeline = sorted(rows, key=lambda r: (r.get("created_at", ""), r["experiment_id"]))
    files["experiment-index.md"] = "# 实验时间线\n\n> 每条谱系只列一次；详细字段见 `all-experiments.md`。\n\n" + table(["日期", "ID", "路线", "证据", "结论"], [(r.get("created_at", ""), r["experiment_id"], r["route_id"], r["evidence_state"], r["conclusion"]) for r in timeline]) + "\n"
    ids = {r["experiment_id"] for r in rows}
    edges = []
    for r in rows:
        for nxt in (r.get("successor", "") or "").split("|"):
            if nxt in ids: edges.append((r["experiment_id"], nxt))
    files["experiment-lineage.md"] = "# 实验血缘 DAG\n\n唯一边集（只保留 registry 中存在的 ID，避免重复谱系视图）：\n\n" + table(["父实验", "子实验"], sorted(set(edges))) + "\n"
    files["method-map.md"] = "# 方法族地图\n\n" + table(["路线", "状态", "证据数", "结果", "终止/下一门"], [(x["route_id"], x["state"], len(x["evidence_ids"]), x["outcome"], x["terminal_reason"] if x["state"] in {"closed", "blocked"} else x["next_gate"]) for x in routes]) + "\n"
    files["methods-tried-zh.md"] = """# 试过的方法（大白话版）\n\n- 先做了表格树模型、R2 漂移归一化、22 个微观 primitive 和多模型融合，形成 152+73Z 与 Clean Baseline 两个冻结资产。\n- RealMLP 在严格时序协议下成立；TCN、无监督 latent、幅度门、残差检索和 M01–M05 没有达到稳定晋级线。\n- SCFI/Z 是当前唯一被生产吸收的条件创新；Cancel 与 Z 重叠，因此被吸收而不是叠加。\n- E01 ReVol-lite 四折为正但 PSEUDO 不够；E02/E03 是诊断，不是预测模型。\n- M06 没有可复现的 train/test 资产时间键，正式冻结为不可识别。\n- 外部 lb142 与纯原创 P6-ORIG 分栏记录，不混入本地训练。\n\n具体数值和证据索引见 registry 与项目状态报告。\n"""
    failed = [r for r in rows if r["evidence_state"] in {"negative", "insufficient", "invalid", "not_identifiable"}]
    groups = {}
    for r in failed: groups.setdefault(r["failure_category"], []).append(r)
    body = "# 失败与不足结果库\n\n失败不等于删除；每条记录都保留原因和 do-not-repeat。\n\n"
    for cat in sorted(groups):
        body += f"## {cat}\n\n" + table(["ID", "路线", "证据", "结论", "原因"], [(r["experiment_id"], r["route_id"], r["evidence_state"], r["conclusion"], r["failure_reason"]) for r in groups[cat]]) + "\n\n"
    files["failed-experiments.md"] = body
    active = [x for x in routes if x["state"] in {"active", "candidate"}]
    files["current-research-queue.md"] = "# 当前研究队列\n\n本页只展示 active/candidate 路线；本阶段不自动运行实验。\n\n" + table(["路线", "状态", "已知结果", "下一门"], [(x["route_id"], x["state"], x["outcome"], x["next_gate"]) for x in active]) + "\n"
    sub_body = "# 提交登记\n\n提交成绩与实验事实分栏，LB142 仅作 external-assisted 取证。数据和预测文件不入库。\n\n" + table(["提交", "实验", "Kaggle ref", "日期", "LB", "归属", "状态"], [(s["submission_id"], s["experiment_id"], s["kaggle_ref"], s["submitted_at"], s["public_lb"], s["ownership"], s["status"]) for s in submissions]) + "\n"
    files["../submissions/README.md"] = sub_body
    status = """# MSCapital 项目状态\n\n更新时间：2026-08-21。当前没有运行中的训练或 Kaggle 提交。\n\n## 两条成绩轨\n\n- external-assisted：v8b/lb142，Public LB **0.142**，只作为外部取证。\n- self-owned：P6-ORIG，Public LB **0.132**，作为纯原创锚点。\n\n## 冻结资产\n\n- Clean Baseline v2：C4 严格嵌套协议，四折约 0.142649 / 0.141762 / 0.143515 / 0.156924，RMS 生产规则。\n- 152+73Z：SCFI/Z 条件线，当前生产特征资产。\n\n## 已关闭或不足\n\nTCN、无监督 latent、幅度 gate、残差检索、NCL/V-REx、M01–M05 均未达到稳定晋级门槛；Cancel 被证明是 Z 的重叠信息。M06 因缺少 train/test 共有的资产/时间键而 `not_identifiable`。E01 四折均正但 PSEUDO 未达 +0.0015，E02/E03 仅诊断。P10 RQ 生产运行因跨历史范围定标不一致而 `protocol_invalid`。\n\n## 当前推荐\n\n科学探索优先 BLSM-G1；工程提分其次是 RealMLP recipe 组合；H7 refit ensemble 只用于降方差。三者在本阶段均不启动，预注册见 `docs/next-direction-2026-08-21.md`。\n\n## 权威链\n\n`CONTEXT.md` → `experiments/registry.csv` → `experiments/routes.yaml` → `submissions/registry.csv` → 本页 → 生成视图 → 单实验报告 → `RESULTS.md`。\n"""
    files["project-status.md"] = status
    return {k: v.rstrip() + "\n" for k, v in files.items()}

def write_files(files):
    for rel, content in files.items():
        p = (DOCS / rel).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); args = ap.parse_args()
    expected = render(rows())
    if args.check:
        for k in expected:
            p = (DOCS / k).resolve()
            if not p.exists() or p.read_text(encoding="utf-8") != expected[k]: raise SystemExit(f"view out of date: {k}")
        print("views: deterministic")
    else:
        write_files(expected)
        print(f"generated {len(expected)} views")

if __name__ == "__main__": main()
