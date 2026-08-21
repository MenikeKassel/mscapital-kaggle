#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One-way migration of the research ledger to SSOT schema v3.

This is deliberately a data migration, not a model runner.  It is safe to
re-run: the input is read from the current registry and the canonical mapping
is idempotent.  Generated views are produced by build_project_views.py.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "experiments" / "registry.csv"
META = ROOT / "experiments" / "registry.meta.json"

OLD_TO_NEW = {
    "P8-01A": "P8-01", "P9-NC": "P9-01", "P9-NEUT": "P9-02",
    "P9-DG": "P9-03", "P9-A-LITE": "P9-04", "P9-B-LITE": "P9-05",
    "P9-C-LITE": "P9-06", "P9-A-V1": "P9-07", "P9-A-V2": "P9-08",
    "P10-PROD-RQ": "P10-01", "P6-R20260821": "P6-02", "BLSM-G0": "P11-01",
}

LEGACY = {"B0", "B1", "A1", "A2", "B1-LGO", "B2", "C1-FE", "D1", "E1-TW", "F1", "F2", "G1", "G2", "G3", "H1"}

PHASE_DIRS = {
    "baseline": "baseline", "P0_protocol": "P0_protocol", "P1_representation": "P1_representation",
    "P2_calibration": "P2_calibration", "C_clean": "C_clean_baseline", "C_clean_baseline": "C_clean_baseline",
    "P3_nextgen": "P3_nextgen", "P4_hidden": "P4_hidden_info", "M_representation": "M_representation",
    "E_conditional": "E_conditional", "P5_market": "P5_market", "P6_production": "P6_production",
    "P7_amplitude": "P7_amplitude", "P8_ot": "P8_ot", "P9_quant": "P9_quant", "p9_lite": "P9_lite",
    "P10_prod": "P10_prod", "P10_feature_mining": "P10_feature_mining", "P9_blsm": "P9_blsm",
    "S_submissions": "S_submissions",
}

NEW_COLUMNS = [
    "experiment_id", "id_status", "title", "name", "phase", "created_at", "status", "decision",
    "parent", "successor", "aliases", "tags", "data_market", "data_order", "data_transaction",
    "base_model", "objective", "validation", "baseline_score", "score", "delta", "public_lb",
    "conclusion", "failure_reason", "do_not_repeat", "script_path", "report_path", "artifact_path",
    "record_kind", "evidence_state", "ownership", "route_id", "failure_category", "provenance_state", "source_refs",
]

MISSING = [
    {"experiment_id": "P8-02", "title": "GAP-N1/N3/N5 时序缺口补证", "name": "otlag-gap-followup", "phase": "P8_ot", "status": "completed", "decision": "RED", "parent": "P8-01", "successor": "", "score": "N/A", "delta": "N/A", "conclusion": "N1/N3/N5 均未形成可迁移时序 alpha", "failure_reason": "跨月时序关系不足且不可复现", "do_not_repeat": "不重复扫描已否定的 O→T gap 变体", "script_path": "N/A", "report_path": "RESULTS.md#P8", "artifact_path": "output/p8_gap_*", "source_refs": "git:3d2e139|docs:p8-ot-gap-report.md"},
    {"experiment_id": "P10-02", "title": "SCFI plain realization", "name": "scfi-plain-realization", "phase": "P10_feature_mining", "status": "completed", "decision": "GREEN", "parent": "P5-04", "successor": "P10-03", "score": "0.143", "delta": "+0.0075", "conclusion": "SCFI plain realization 提供条件增量", "failure_reason": "", "do_not_repeat": "", "script_path": "N/A", "report_path": "docs/p5b-scfi-report.md#plain", "artifact_path": "output/p10_scfi_plain", "source_refs": "git:38dab2c|docs:p5b-scfi-report.md"},
    {"experiment_id": "P10-03", "title": "SCFI RQ realization", "name": "scfi-rq-realization", "phase": "P10_feature_mining", "status": "completed", "decision": "GREEN", "parent": "P10-02", "successor": "P10-04", "score": "0.1438", "delta": "+0.0040", "conclusion": "RQ 条件表示在本地验证有增量", "failure_reason": "生产定标协议未闭合，不能直接外推", "do_not_repeat": "不得把 arm_C 定标套用 full-history 生产模型", "script_path": "N/A", "report_path": "RESULTS.md#P10", "artifact_path": "output/p10_scfi_rq", "source_refs": "git:eb33215|docs:p10-scfi-rq-report.md"},
    {"experiment_id": "P10-04", "title": "Z_O2 二阶条件特征", "name": "z-o2-second-order", "phase": "P10_feature_mining", "status": "completed", "decision": "GREEN", "parent": "P10-03", "successor": "P10-05", "score": "N/A", "delta": "+0.0060", "conclusion": "二阶条件项进入 152+73Z 冻结资产", "failure_reason": "", "do_not_repeat": "", "script_path": "N/A", "report_path": "RESULTS.md#P10", "artifact_path": "output/p10_z_o2", "source_refs": "git:38dab2c|docs:p10-feature-mining.md"},
    {"experiment_id": "P10-05", "title": "M1 L2 档位", "name": "m1-l2-level", "phase": "P10_feature_mining", "status": "completed", "decision": "YELLOW", "parent": "P10-04", "successor": "P10-06", "score": "N/A", "delta": "N/A", "conclusion": "L2 变体有局部信息但未形成独立候选", "failure_reason": "信息与既有 Z 高度重叠", "do_not_repeat": "不单独提交 L2 变体", "script_path": "N/A", "report_path": "RESULTS.md#P10", "artifact_path": "output/p10_m1_l2", "source_refs": "git:38dab2c|docs:p10-feature-mining.md"},
    {"experiment_id": "P10-06", "title": "M2 event/jump", "name": "m2-event-jump", "phase": "P10_feature_mining", "status": "completed", "decision": "YELLOW", "parent": "P10-05", "successor": "P10-07", "score": "N/A", "delta": "N/A", "conclusion": "事件跳变条件项未超过已冻结 Z", "failure_reason": "与 event/time primitive 冗余", "do_not_repeat": "不重复扫描相同 event/jump 组合", "script_path": "N/A", "report_path": "RESULTS.md#P10", "artifact_path": "output/p10_m2_event_jump", "source_refs": "git:38dab2c|docs:p10-feature-mining.md"},
    {"experiment_id": "P10-07", "title": "TX H1/H2/H3", "name": "tx-h123", "phase": "P10_feature_mining", "status": "completed", "decision": "YELLOW", "parent": "P10-06", "successor": "P10-01", "score": "N/A", "delta": "N/A", "conclusion": "成交条件 H1/H2/H3 只作诊断，不独立生产", "failure_reason": "跨折增量不足且与 Z 重叠", "do_not_repeat": "不独立提交 TX H1/H2/H3", "script_path": "scripts/p10_tx_feat_build.py", "report_path": "RESULTS.md#P10", "artifact_path": "output/p10_tx_*", "source_refs": "git:38dab2c|docs:p10-feature-mining.md"},
    {"experiment_id": "P6-03", "title": "P6-ORIG 纯原创融合构建", "name": "p6-original-build", "phase": "P6_production", "status": "completed", "decision": "GREEN", "parent": "P6-02", "successor": "S-11", "score": "+0.001", "delta": "+0.001", "conclusion": "纯原创 v5 Table + RealMLP-C 融合构建完成", "failure_reason": "", "do_not_repeat": "", "script_path": "N/A", "report_path": "docs/p6-production-inference.md#orig", "artifact_path": "output/p6_orig", "source_refs": "git:9448232|docs:p6-production-inference.md"},
    {"experiment_id": "S-09", "title": "P10 RQ 提交", "name": "submission-p10-rq", "phase": "S_submissions", "status": "completed", "decision": "RED", "parent": "P10-01", "successor": "", "score": "0.1438", "delta": "N/A", "public_lb": "0.116", "conclusion": "格式错误 ref 后的最终 LB 显著低于 v8b", "failure_reason": "protocol_invalid：校准模型与提交模型历史范围不一致", "do_not_repeat": "不得跨模型套用 neutralization/rq scale", "script_path": "N/A", "report_path": "RESULTS.md#P10", "artifact_path": "output/submissions/p10_rq", "source_refs": "kaggle:format-error|kaggle:final-LB-0.116"},
    {"experiment_id": "S-10", "title": "Z 线重跑提交", "name": "submission-z-rerun", "phase": "S_submissions", "status": "completed", "decision": "YELLOW", "parent": "P10-04", "successor": "", "score": "N/A", "delta": "N/A", "public_lb": "0.141", "conclusion": "Z 线重跑低于 external lb142 但高于纯原创锚点", "failure_reason": "", "do_not_repeat": "", "script_path": "scripts/p6_prod_realmlp.py", "report_path": "docs/p6-production-inference.md#rerun", "artifact_path": "output/submissions/z-rerun", "source_refs": "kaggle:Z-rerun-LB-0.141"},
    {"experiment_id": "S-11", "title": "P6-ORIG 纯原创提交", "name": "submission-p6-original", "phase": "S_submissions", "status": "completed", "decision": "GREEN", "parent": "P6-03", "successor": "", "score": "N/A", "delta": "N/A", "public_lb": "0.132", "conclusion": "纯原创基准提交，作为 self-owned anchor", "failure_reason": "", "do_not_repeat": "", "script_path": "scripts/p6_orig_submit.py", "report_path": "docs/p6-production-inference.md#orig-submit", "artifact_path": "output/submissions/p6-orig", "source_refs": "kaggle:55657080|LB:0.132"},
]
BACKFILL_IDS = tuple(x["experiment_id"] for x in MISSING)

def _pipe_paths(value: str) -> str:
    if not value:
        return ""
    value = value.replace("\\", "/")
    # Never publish machine-specific roots in the ledger.
    value = re.sub(r"[A-Za-z]:/[^\s|,)]+", "<local-data-root>", value)
    value = re.sub(r"\(([^)]+)\)", r"\1", value)
    value = re.sub(r"\s*(?:kernel|gitignore)", "", value)
    value = re.sub(r"\s+", "|", value)
    value = re.sub(r"\|(?=\*)", "|", value)
    value = re.sub(r"\|(?=(?:scripts|docs|output|src|processed)/)", "|", value)
    return value.strip(" |")

def _report(value: str) -> str:
    if not value:
        return ""
    value = value.replace("\\", "/")
    value = re.sub(r"\s+§\s*", "#", value)
    value = re.sub(r"\s+\((N\d+)\)", r"#\1", value)
    return value

def _route(cid: str, row: dict) -> str:
    if cid in LEGACY or cid.startswith(("B", "A", "F", "G", "H")):
        return "R01-table-baseline"
    if cid.startswith("P0-"): return "R02-r2-drift"
    if cid == "P1-05": return "R05-sequence"
    if cid.startswith("P1-"): return "R03-micro-primitives"
    if cid.startswith("C-") and cid not in {"C-01", "C-02", "C-03", "C-04"}: return "R17-realmlp-recipe"
    if cid.startswith(("P2-", "C-")): return "R04-realmlp-clean"
    if cid.startswith("P3-"): return "R08-unsupervised-latent"
    if cid.startswith("P4-"): return "R09-hidden-information"
    if cid in {"M-02", "M-03", "M-04"} or cid == "P5-05": return "R12-geometry-signature"
    if cid.startswith("M-"): return "R06-m-residual"
    if cid.startswith("E-"): return "R07-state-conditioned"
    if cid.startswith("P5-"): return "R11-scfi-z"
    if cid.startswith(("P6-", "P6R-")): return "R13-p6r-production"
    if cid.startswith("P7-"): return "R10-amplitude-gate"
    if cid.startswith("P8-"): return "R14-o-to-t"
    if cid.startswith("P9-"):
        return "R15-p9-quant" if cid in {"P9-01", "P9-02", "P9-03"} else "R16-cancel-eventtime"
    if cid.startswith("P10-"): return "R11-scfi-z"
    if cid.startswith("P11-"): return "R18-blsm"
    if cid == "S-08": return "R19-production-calibration"
    if cid.startswith("S-"): return "R20-submissions"
    return "R01-table-baseline"

def _record_kind(cid: str, row: dict) -> str:
    title = (row.get("title", "") + " " + row.get("name", "")).lower()
    if cid.startswith("S-"): return "submission"
    if "audit" in title or "取证" in title or cid in {"M-07", "P11-01"}: return "audit" if cid == "M-07" else "diagnostic"
    if "build" in title or "构建" in title: return "build"
    if "protocol" in title or "协议" in title or cid in {"P0-01", "P0-02", "P0-04", "C-04"}: return "protocol"
    if "ablation" in title or "消融" in title or cid.startswith("C-"): return "ablation"
    if "blend" in title or "fusion" in title or "融合" in title: return "ensemble"
    if "feature" in title or "特征" in title or cid.startswith(("P1-", "M-")): return "feature"
    return "model"

def _failure_category(cid: str, row: dict, evidence: str) -> str:
    if evidence == "invalid": return "protocol_invalid"
    if evidence == "not_identifiable": return "not_identifiable"
    text = (row.get("title", "") + row.get("conclusion", "") + row.get("failure_reason", "")).lower()
    if "cancel" in text or "重叠" in text or "overlap" in text: return "redundancy"
    if "漂移" in text or "跨月" in text or "nonstation" in text: return "nonstationarity"
    if "目标" in text or "objective" in text or "cosine" in text and "弱" in text: return "objective_mismatch"
    if row.get("decision") == "RED": return "model_saturation"
    return "none"

def _evidence(cid: str, row: dict) -> str:
    if cid == "P10-01" or cid == "S-09": return "invalid"
    if cid == "M-07": return "not_identifiable"
    if cid in {"E-02", "E-03", "P11-01"}: return "descriptive"
    if cid in {"E-01", "P9-02", "P9-06", "M-01", "M-02", "M-03", "M-04", "M-05", "M-06"}: return "insufficient"
    if row.get("status") == "planned": return "pending"
    if row.get("status") == "superseded": return "superseded"
    if row.get("decision") == "RED": return "negative"
    return "validated"

def _ownership(cid: str) -> str:
    if cid in {"S-08", "P4-02", "P4-03", "P4-04", "P4-05", "P4-06", "P4-07", "P4-08", "P4-09", "P4-10", "P4-11", "P4-12", "P4-13", "P4-14", "P4-16", "P4-17"}: return "external_assisted"
    if cid == "M-07": return "not_applicable"
    return "historical" if cid in LEGACY else "self_owned"

def load_rows() -> list[dict]:
    with REG.open(encoding="utf-8-sig", newline="") as f:
        old = list(csv.DictReader(f))
    rows = []
    for src in old:
        old_id = src["experiment_id"]
        cid = OLD_TO_NEW.get(old_id, old_id)
        row = dict(src)
        row["experiment_id"] = cid
        row["id_status"] = "legacy" if cid in LEGACY else "canonical"
        for rel in ("parent", "successor"):
            row[rel] = "|".join(OLD_TO_NEW.get(x, x) for x in (row.get(rel, "") or "").split("|") if x)
        if old_id != cid:
            aliases = [x for x in (src.get("aliases", "").split("|") + [old_id]) if x and x != cid]
            row["aliases"] = "|".join(dict.fromkeys(aliases))
        rows.append(row)
    existing = {r["experiment_id"] for r in rows}
    for extra in MISSING:
        if extra["experiment_id"] not in existing:
            x = dict(extra)
            x["id_status"] = "canonical"
            x["aliases"] = ""
            rows.append(x)
    # stable order: historical source order, then backfills
    for row in rows:
        cid = row["experiment_id"]
        evidence = _evidence(cid, row)
        row.setdefault("public_lb", "")
        row["phase"] = row.get("phase", "") or "baseline"
        if cid == "P11-01" and row.get("decision") == "EXIST":
            row["decision"] = "GREEN"
        row["script_path"] = _pipe_paths(row.get("script_path", ""))
        row["report_path"] = _report(row.get("report_path", ""))
        row["artifact_path"] = _pipe_paths(row.get("artifact_path", ""))
        # Backfilled records are evidence-index entries; when an implementation
        # script/report was never committed, keep the scope in source_refs and
        # use N/A rather than a dangling repository path.
        if cid in {"P8-02", "P10-02", "P10-03", "P10-04", "P10-05", "P10-06", "P10-07", "P6-03", "S-09"}:
            row["script_path"] = "N/A"
        if cid == "P8-02": row["report_path"] = "RESULTS.md#P8"
        if cid in {"P10-03", "P10-04", "P10-05", "P10-06", "P10-07"}: row["report_path"] = "RESULTS.md#P10"
        if cid == "M-03": row["script_path"] = "src/mscapital/__main__.py"
        row["record_kind"] = _record_kind(cid, row)
        row["evidence_state"] = evidence
        row["ownership"] = _ownership(cid)
        row["route_id"] = _route(cid, row)
        row["failure_category"] = _failure_category(cid, row, evidence)
        row["provenance_state"] = "historical" if cid in LEGACY or cid in OLD_TO_NEW else ("verified" if row.get("source_refs") else "partial")
        row.setdefault("source_refs", "")
        if not row.get("source_refs"):
            row["source_refs"] = f"RESULTS.md#{cid}"
        if not row.get("created_at"): row["created_at"] = "2026-08-21"
        # Missing historical facts are explicit, never inferred.  Non-model
        # records use N/A; model/ablation/ensemble records use unknown.
        fact_default = "unknown" if row["record_kind"] in {"model", "ablation", "ensemble", "feature"} else "N/A"
        for key in ("objective", "validation", "baseline_score", "score", "delta"):
            if not row.get(key) or (fact_default == "unknown" and row.get(key) == "N/A"): row[key] = fact_default
        if not row.get("public_lb"): row["public_lb"] = "N/A"
        for c in NEW_COLUMNS:
            row.setdefault(c, "")
    assert len(MISSING) == 11
    assert len(rows) == 116, len(rows)
    assert sum(r["id_status"] == "legacy" for r in rows) == 15
    return [{c: r.get(c, "") for c in NEW_COLUMNS} for r in rows]

def write_registry(rows: list[dict]) -> None:
    with REG.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=NEW_COLUMNS)
        w.writeheader(); w.writerows(rows)
    meta = {
        "schema_version": 3, "experiment_id_spec_version": "1.1", "generated_at": str(date.today()),
        "entries": len(rows), "canonical_entries": sum(r["id_status"] == "canonical" for r in rows),
        "legacy_entries": sum(r["id_status"] == "legacy" for r in rows), "canonical_series": ["P0-P11", "P6R", "C", "E", "M", "S"],
        "legacy_allowlist": sorted(LEGACY), "migration": "experiments/_tools/migrate_ssot_v3.py",
    }
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "experiments" / "id-migration-v1.1.json").write_text(json.dumps(OLD_TO_NEW, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def write_dirs(rows: list[dict]) -> None:
    exp_root = ROOT / "experiments"
    for row in rows:
        d = exp_root / PHASE_DIRS.get(row["phase"], row["phase"]) / f"{row['experiment_id']}_{row['name'][:48]}"
        d.mkdir(parents=True, exist_ok=True)
        readme = d / "README.md"
        if not readme.exists():
            readme.write_text(f"# {row['experiment_id']} — {row['title']}\n\n状态：{row['status']}；证据：{row['evidence_state']}；路线：{row['route_id']}。\n\n本目录为实验索引入口；详细事实以 registry.csv 和 source_refs 为准。\n", encoding="utf-8")

def write_submissions() -> None:
    out = ROOT / "submissions" / "registry.csv"
    out.parent.mkdir(exist_ok=True)
    cols = ["submission_id", "experiment_id", "kaggle_ref", "submitted_at", "status", "local_evidence", "public_lb", "ownership", "recipe", "generation_script"]
    records = [
        ("S-01-v1", "S-01", "v1", "2026-08-10", "submitted", "RESULTS.md#v1", "0.122", "self_owned", "G3 three-model blend", "scripts/18_final_submission_v2.py"),
        ("S-02-v2", "S-02", "v2", "2026-08-10", "submitted", "RESULTS.md#v2", "0.122", "self_owned", "H1 five-model blend", "scripts/18_final_submission_v2.py"),
        ("S-03-v3", "S-03", "v3", "2026-08-10", "submitted", "RESULTS.md#v3", "0.122", "self_owned", "temporal reweight", "scripts/22_final_submission_v3.py"),
        ("S-04-v4", "S-04", "v4", "2026-08-11", "submitted", "RESULTS.md#v4", "0.123", "self_owned", "R2+temporal", "scripts/22_final_submission_v3.py"),
        ("S-05-v5", "S-05", "v5", "2026-08-11", "submitted", "RESULTS.md#v5", "0.125", "self_owned", "R2+22 micro", "scripts/27_build_micro_features.py"),
        ("S-06-v6", "S-06", "v6", "2026-08-11", "submitted", "RESULTS.md#v6", "0.082", "self_owned", "TCN fusion", "scripts/37_final_v6.py"),
        ("S-07-v7", "S-07", "v7", "2026-08-12", "submitted", "RESULTS.md#v7", "0.135", "self_owned", "RealMLP blend", "scripts/42_realmlp_fusion.py"),
        ("S-08-v8", "S-08", "v8", "2026-08-12", "submitted", "RESULTS.md#v8", "0.139", "external_assisted", "lb142 reference blend", "scripts/43_lb142_fusion.py"),
        ("S-08-v8b", "S-08", "v8b", "2026-08-12", "submitted", "RESULTS.md#v8b", "0.142", "external_assisted", "lb142 reference blend", "scripts/43_lb142_fusion.py"),
        ("S-09-p10", "S-09", "p10-rq", "2026-08-20", "submitted", "RESULTS.md#P10", "0.116", "self_owned", "152+73Z+neutralization", "scripts/p10_submit.py"),
        ("S-10-z", "S-10", "z-rerun", "2026-08-21", "submitted", "docs/p6-production-inference.md#rerun", "0.141", "self_owned", "Z line rerun", "scripts/p6_prod_realmlp.py"),
        ("S-11-p6-orig", "S-11", "55657080", "2026-08-21", "submitted", "docs/p6-production-inference.md#orig-submit", "0.132", "self_owned", "P6-ORIG", "scripts/p6_orig_submit.py"),
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(cols); w.writerows(records)

def write_routes(rows: list[dict]) -> None:
    specs = [
        ("R01-table-baseline", "表格 baseline 与树模型融合", "frozen", "Clean Table lineage is historical anchor", "", "R04-realmlp-clean"),
        ("R02-r2-drift", "R2 漂移归一化", "frozen", "R2 retained in 152+73Z", "", "R04-realmlp-clean"),
        ("R03-micro-primitives", "22 个微观 primitive", "frozen", "micro features absorbed into frozen asset", "", "R04-realmlp-clean"),
        ("R04-realmlp-clean", "RealMLP 与 Clean Baseline", "frozen", "Clean Baseline v2 frozen", "", "R17-realmlp-recipe"),
        ("R05-sequence", "TCN/序列模型", "closed", "TCN catastrophic LB and weak stable evidence", "model_saturation", "R06-m-residual"),
        ("R06-m-residual", "M01–M06 residual representation", "closed", "all residual families are insufficient for the gate", "insufficient", "R12-geometry-signature"),
        ("R07-state-conditioned", "E01–E03 状态条件化", "closed", "E01 positive but PSEUDO below gate; E02/E03 diagnostic", "insufficient", "R18-blsm"),
        ("R08-unsupervised-latent", "P3 无监督/latent", "closed", "SAE/TinyLOBERT/grid/NHP no stable gain", "model_saturation", "R18-blsm"),
        ("R09-hidden-information", "P4 隐藏信息与协议审计", "external", "LB142 forensic and market history evidence", "", "R20-submissions"),
        ("R10-amplitude-gate", "幅度/confidence gate", "closed", "outer gate non-positive", "objective_mismatch", "R11-scfi-z"),
        ("R11-scfi-z", "SCFI/Z 条件创新", "frozen", "152+73Z is production asset; P10 RQ run invalid", "", "R20-submissions"),
        ("R12-geometry-signature", "Geometry/RICS/Signature", "closed", "M02/M03/M04 below gate; no extension", "insufficient", "R18-blsm"),
        ("R13-p6r-production", "P6R residual retrieval", "closed", "positive but below gate; P6-ORIG anchor retained", "insufficient", "R20-submissions"),
        ("R14-o-to-t", "P8 O→T 时序", "closed", "no reproducible lead-lag", "nonstationarity", "R15-p9-quant"),
        ("R15-p9-quant", "P9 quant neutralization/V-REx/NCL", "closed", "neutralization insufficient; NCL/V-REx negative", "objective_mismatch", "R16-cancel-eventtime"),
        ("R16-cancel-eventtime", "Cancel/Event-Time/M55", "closed", "Cancel absorbed by Z; Event-Time negative; M55 and neutralization insufficient", "redundancy", "R18-blsm"),
        ("R17-realmlp-recipe", "RealMLP recipe 组合", "candidate", "ParamMish/PL/PBLD/schedreg/coslog4 are candidate components", "", "R18-blsm"),
        ("R18-blsm", "BLSM behavior state", "active", "G0 validates behavior latent existence; G1 pre-registered", "", "R17-realmlp-recipe"),
        ("R19-production-calibration", "生产校准和 Kaggle submission", "external", "external lb142 and self-owned anchors separated", "", ""),
        ("R20-submissions", "Kaggle submissions", "frozen", "submission ledger is evidence-only", "", ""),
    ]
    byroute = {r[0]: [x["experiment_id"] for x in rows if x["route_id"] == r[0]] for r in specs}
    out = []
    for rid, title, state, outcome, reason, gate in specs:
        out.append({"route_id": rid, "title": title, "state": state, "outcome": outcome, "evidence_ids": byroute.get(rid, []), "terminal_reason": reason or "N/A", "do_not_repeat": "see route outcome; do not retune closed family" if state == "closed" else "N/A", "next_gate": gate or "N/A"})
    (ROOT / "experiments" / "routes.yaml").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def main() -> None:
    rows = load_rows(); write_registry(rows); write_dirs(rows); write_submissions(); write_routes(rows)
    print(f"migrated {len(rows)} rows: {sum(r['id_status']=='legacy' for r in rows)} legacy, {sum(r['id_status']=='canonical' for r in rows)} canonical")

if __name__ == "__main__":
    main()
