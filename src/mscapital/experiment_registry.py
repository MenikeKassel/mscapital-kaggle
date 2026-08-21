# -*- coding: utf-8 -*-
"""MSCapital Experiment Registry 核心库 (Experiment ID Spec v1.1).

提供: canonical/legacy 分类, 审计, resolver, allocator, consistency 校验。
SSOT: experiments/registry.csv + experiments/registry.meta.json
"""
from __future__ import annotations

import csv
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGISTRY_PATH = os.path.join(REPO_ROOT, "experiments", "registry.csv")
META_PATH = os.path.join(REPO_ROOT, "experiments", "registry.meta.json")

# ---- Experiment ID Spec v1.1 ----
# Primary IDs are numeric.  P6R remains a historical residual series.
CANONICAL_RE = re.compile(r"^(P(?:[0-9]|1[0-1])|P6R|C|E|M|S)-[0-9]{2}[a-z]?$")
VALID_SERIES = tuple([f"P{i}" for i in range(12)] + ["P6R", "C", "E", "M", "S"])
SPECIAL_SERIES = ("P6R",)  # 历史特殊 series, 非开放模板

# Lifecycle (任务书 §21)
VALID_STATUS = ("planned", "running", "completed", "aborted", "deprecated", "superseded")
# Decision (任务书 §22)
VALID_DECISIONS = ("GREEN", "YELLOW", "RED", "NA")

SCHEMA_VERSION = 3
SPEC_VERSION = "1.1"

REQUIRED_COLUMNS = [
    "experiment_id", "id_status", "title", "name", "phase", "created_at",
    "status", "decision", "parent", "successor", "aliases", "tags",
    "data_market", "data_order", "data_transaction", "base_model", "objective",
    "validation", "baseline_score", "score", "delta", "public_lb",
    "conclusion", "failure_reason", "do_not_repeat",
    "script_path", "report_path", "artifact_path",
    "record_kind", "evidence_state", "ownership", "route_id", "failure_category",
    "provenance_state", "source_refs",
]

VALID_RECORD_KINDS = ("build", "protocol", "diagnostic", "feature", "model", "ablation", "ensemble", "audit", "submission")
VALID_EVIDENCE_STATES = ("pending", "validated", "descriptive", "insufficient", "negative", "invalid", "not_identifiable", "superseded")
VALID_OWNERSHIP = ("self_owned", "external_assisted", "local_protocol", "not_applicable", "historical")
VALID_FAILURE_CATEGORIES = ("none", "redundancy", "nonstationarity", "objective_mismatch", "nontransferable_residual", "protocol_invalid", "not_identifiable", "model_saturation", "implementation_only")
VALID_PROVENANCE_STATES = ("verified", "partial", "historical")


# ---- 加载 ----
def load_registry(path: str = REGISTRY_PATH) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_meta(path: str = META_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---- 分类 ----
def classify_id(experiment_id: str) -> str:
    """返回: canonical / legacy / invalid"""
    if CANONICAL_RE.fullmatch(experiment_id):
        return "canonical"
    if experiment_id in LEGACY_ALLOWLIST:
        return "legacy"
    return "invalid"


# ---- Resolver (任务书 §11, 单步解析) ----
def build_alias_index(registry: List[Dict[str, str]]) -> Dict[str, str]:
    """alias / legacy_id / canonical_id -> canonical_id (一步).
    canonical 自身 -> 自身; legacy 主 ID -> 自身 (legacy 不是 alias).
    """
    index: Dict[str, str] = {}
    for row in registry:
        cid = row["experiment_id"]
        if row.get("id_status") == "canonical":
            index[cid] = cid
            for alias in _split_aliases(row.get("aliases", "")):
                index[alias] = cid
        else:  # legacy
            index[cid] = cid  # legacy 解析自身
    return index


def _split_aliases(s: str) -> List[str]:
    return [a.strip() for a in s.split("|") if a.strip()]


def resolve_experiment_id(query: str, registry: Optional[List[Dict[str, str]]] = None) -> str:
    """统一解析: canonical 自身 / alias 一步 / legacy 自身; 未知 -> KeyError. (任务书 §11)"""
    reg = registry if registry is not None else load_registry()
    index = build_alias_index(reg)
    q = query.strip()
    if q not in index:
        raise KeyError(f"unknown experiment id/alias: {q!r}")
    return index[q]


def resolve_row(query: str, registry: Optional[List[Dict[str, str]]] = None) -> Dict[str, str]:
    cid = resolve_experiment_id(query, registry)
    reg = registry if registry is not None else load_registry()
    for row in reg:
        if row["experiment_id"] == cid:
            return row
    raise KeyError(cid)


# ---- Allocator (任务书 §15/§16/§17: max-ever + 1, 永不回收) ----
def allocated_ids_for_series(registry: List[Dict[str, str]], series: str) -> List[int]:
    nums = []
    prefix = series + "-"
    for row in registry:
        cid = row["experiment_id"]
        if cid.startswith(prefix) and row.get("id_status") != "legacy":
            m = re.fullmatch(re.escape(prefix) + r"(\d{2})[a-z]?", cid)
            if m:
                nums.append(int(m.group(1)))
    return sorted(nums)


def allocate_next_id(registry: List[Dict[str, str]], series: str) -> str:
    """monotonic only: max(ever) + 1, 不回填空洞. (任务书 §16)"""
    if series not in VALID_SERIES:
        raise ValueError(f"invalid series {series!r}; valid: {VALID_SERIES}")
    nums = allocated_ids_for_series(registry, series)
    nxt = (nums[-1] + 1) if nums else 1
    return f"{series}-{nxt:02d}"


def allocate_next_arm(registry: List[Dict[str, str]], parent_id: str) -> str:
    """arm 分配: 只能在已存在 canonical parent 下. (任务书 §37)"""
    parent = resolve_row(parent_id, registry)
    if parent["id_status"] != "canonical":
        raise ValueError(f"arm parent must be canonical: {parent_id}")
    base = parent["experiment_id"]
    used = {row["experiment_id"] for row in registry}
    for letter in "abcdefghijklmnopqrstuvwxyz":
        cand = f"{base}{letter}"
        if cand not in used:
            return cand
    raise RuntimeError(f"no arm slot left for {base}")


# ---- 审计 / consistency (任务书 §30/§33/§35) ----
def audit_registry(registry: List[Dict[str, str]], root: str = REPO_ROOT) -> Dict[str, Any]:
    stats = {
        "total_registry_entries": len(registry), "canonical_valid": 0, "legacy_exempt": 0,
        "invalid_unclassified": 0, "alias_count": 0, "alias_collision_count": 0,
        "missing_directory": [], "missing_script": [], "missing_report": [], "missing_artifact": [],
        "orphan_experiments": [], "orphan_registry_rows": [], "id_reuse": [],
        "status_invalid": [], "decision_invalid": [], "record_kind_invalid": [],
        "evidence_state_invalid": [], "ownership_invalid": [], "route_missing": [],
        "failure_category_invalid": [], "provenance_state_invalid": [],
        "dup_ids": [], "completed_without_decision": [], "red_without_failure": [], "alias_chain": [],
    }
    seen_ids: Dict[str, int] = {}
    alias_owner: Dict[str, str] = {}

    for row in registry:
        cid = row["experiment_id"]
        # 唯一性
        seen_ids[cid] = seen_ids.get(cid, 0) + 1
        # 分类
        st = row.get("id_status", "")
        if st == "canonical":
            if CANONICAL_RE.fullmatch(cid):
                stats["canonical_valid"] += 1
            else:
                stats["invalid_unclassified"] += 1
        elif st == "legacy":
            if cid in LEGACY_ALLOWLIST:
                stats["legacy_exempt"] += 1
            else:
                stats["invalid_unclassified"] += 1
        else:
            stats["invalid_unclassified"] += 1
        # status / evidence taxonomy enums
        if row.get("status") not in VALID_STATUS:
            stats["status_invalid"].append(cid)
        if row.get("decision") not in VALID_DECISIONS:
            stats["decision_invalid"].append(cid)
        if row.get("record_kind") not in VALID_RECORD_KINDS:
            stats["record_kind_invalid"].append(cid)
        if row.get("evidence_state") not in VALID_EVIDENCE_STATES:
            stats["evidence_state_invalid"].append(cid)
        if row.get("ownership") not in VALID_OWNERSHIP:
            stats["ownership_invalid"].append(cid)
        if row.get("failure_category") not in VALID_FAILURE_CATEGORIES:
            stats["failure_category_invalid"].append(cid)
        if row.get("provenance_state") not in VALID_PROVENANCE_STATES:
            stats["provenance_state_invalid"].append(cid)
        if not row.get("route_id"):
            stats["route_missing"].append(cid)
        # lifecycle 完整性
        if row.get("status") == "completed" and row.get("decision") in ("", "NA") and row.get("record_kind") not in {"build", "protocol", "diagnostic", "audit"}:
            stats["completed_without_decision"].append(cid)
        if row.get("decision") == "RED" and not row.get("failure_reason"):
            stats["red_without_failure"].append(cid)
        # aliases
        for a in _split_aliases(row.get("aliases", "")):
            stats["alias_count"] += 1
            if a in alias_owner:
                stats["alias_collision_count"] += 1
            else:
                alias_owner[a] = cid
            if a in seen_ids:
                stats["alias_collision_count"] += 1  # alias 撞 primary ID
            if a == cid:
                stats["alias_collision_count"] += 1  # alias == canonical
        # filesystem
        ph = row.get("phase", "")
        phdir = os.path.join(root, "experiments", _phase_dir(ph))
        found_dir = None
        if os.path.isdir(phdir):
            for d in os.listdir(phdir):
                if d.startswith(cid + "_"):
                    found_dir = os.path.join(phdir, d)
                    break
        if not found_dir:
            stats["missing_directory"].append(cid)
        elif not os.path.exists(os.path.join(found_dir, "README.md")):
            stats["missing_directory"].append(cid + "/README")
        # Pipe-separated paths are evidence scopes.  Ignored artifacts and
        # globs are allowed to be absent from a public checkout.
        for key in ("script_path", "report_path", "artifact_path"):
            for p in _split_paths(row.get(key, "")):
                if key == "artifact_path" or not p or p.upper() in {"N/A", "NA"} or "*" in p or p.startswith(("output/", "processed/")):
                    continue
                base = p.split("#", 1)[0]
                if base and not os.path.exists(os.path.join(root, base)):
                    stats[f"missing_{key.replace('_path','')}"].append(f"{cid}: {p}")

    # 双向一致: filesystem -> registry (任务书 §35)
    exp_root = os.path.join(root, "experiments")
    for ph in _all_phase_dirs(root):
        phdir = os.path.join(exp_root, ph)
        if not os.path.isdir(phdir):
            continue
        for d in os.listdir(phdir):
            if not os.path.isdir(os.path.join(phdir, d)) or d.startswith("_"):
                continue
            rid = d.split("_")[0]
            if rid not in seen_ids:
                stats["orphan_experiments"].append(f"{ph}/{d}")

    stats["dup_ids"] = [k for k, v in seen_ids.items() if v > 1]
    stats["canonical_count"] = sum(r.get("id_status") == "canonical" for r in registry)
    stats["legacy_count"] = sum(r.get("id_status") == "legacy" for r in registry)
    return stats


def _split_paths(value: str) -> List[str]:
    return [p.strip() for p in (value or "").split("|") if p.strip()]


def _phase_dir(phase: str) -> str:
    return PHASE_DIRS.get(phase, phase)


def _all_phase_dirs(root: str) -> List[str]:
    exp_root = os.path.join(root, "experiments")
    if not os.path.isdir(exp_root):
        return []
    return [d for d in os.listdir(exp_root)
            if os.path.isdir(os.path.join(exp_root, d)) and not d.startswith("_")]


# phase -> 目录 (与 build_registry 一致)
PHASE_DIRS = {
    "baseline": "baseline", "P0_protocol": "P0_protocol", "P1_representation": "P1_representation",
    "P2_calibration": "P2_calibration", "C_clean": "C_clean_baseline", "C_clean_baseline": "C_clean_baseline", "P3_nextgen": "P3_nextgen",
    "P4_hidden": "P4_hidden_info", "M_representation": "M_representation",
    "E_conditional": "E_conditional", "P5_market": "P5_market",
    "P6_production": "P6_production", "P7_amplitude": "P7_amplitude",
    "P8_ot": "P8_ot", "P9_quant": "P9_quant", "p9_lite": "P9_lite",
    "P10_prod": "P10_prod", "P10_feature_mining": "P10_feature_mining", "P9_blsm": "P9_blsm",
    "S_submissions": "S_submissions",
}


# ---- LEGACY ALLOWLIST (任务书 §8/§9: frozen, 禁止新增) ----
LEGACY_ALLOWLIST = frozenset({
    # baseline 阶段早期表格实验 (进入 v1-v3 提交链与历史报告, 冻结保留)
    "B0", "B1", "A1", "A2", "B1-LGO", "B2", "C1-FE", "D1", "E1-TW",
    "F1", "F2", "G1", "G2", "G3", "H1",
})
