# -*- coding: utf-8 -*-
"""Experiment ID v1.0 Consistency Tests (任务书 §33/§34/§35).

覆盖: ID 语法 / Legacy allowlist / 唯一性 / Alias / Filesystem / Allocator / Resolver /
Lifecycle / 研究质量 / 双向一致 (registry ↔ filesystem).
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from mscapital.experiment_registry import (
    CANONICAL_RE, LEGACY_ALLOWLIST, VALID_STATUS, VALID_DECISIONS,
    allocate_next_arm, allocate_next_id, audit_registry, build_alias_index,
    load_meta, load_registry, resolve_experiment_id, resolve_row,
)

REG = load_registry()


# ---------- ID ----------
def test_canonical_regex():
    for row in REG:
        if row["id_status"] == "canonical":
            assert CANONICAL_RE.fullmatch(row["experiment_id"]), row["experiment_id"]


def test_legacy_allowlist():
    for row in REG:
        if row["id_status"] == "legacy":
            assert row["experiment_id"] in LEGACY_ALLOWLIST, row["experiment_id"]


def test_no_unknown_id_status():
    for row in REG:
        assert row["id_status"] in ("canonical", "legacy"), row["experiment_id"]


def test_ids_unique():
    ids = [r["experiment_id"] for r in REG]
    assert len(ids) == len(set(ids))


def test_series_closed():
    for row in REG:
        if row["id_status"] == "canonical":
            series = re.split(r"-", row["experiment_id"])[0]
            assert series in ("P0","P1","P2","P3","P4","P5","P6","P6R","P7","C","E","M","S")


def test_special_series_only_p6r():
    # 禁止 P5R/P7R/P6A 等衍生特殊 series
    for row in REG:
        if row["id_status"] == "canonical":
            series = re.split(r"-", row["experiment_id"])[0]
            assert not (series not in ("P6R",) and len(series) == 3 and series[1:].isalpha())


# ---------- Alias (任务书 §10) ----------
def test_alias_unique_and_no_collision():
    idx = build_alias_index(REG)
    ids = {r["experiment_id"] for r in REG}
    for a, target in idx.items():
        if a == target:
            continue  # canonical/legacy 自身
        assert a not in ids, f"alias 撞 primary ID: {a}"
        assert target in ids, f"alias target 不存在: {a} -> {target}"


def test_alias_single_step():
    # 一步解析: alias 的解析结果必须本身是 primary (无 chain)
    idx = build_alias_index(REG)
    for a, target in idx.items():
        if a == target:
            continue
        assert idx[target] == target, f"alias chain: {a} -> {target} -> {idx[target]}"


def test_alias_global_unique():
    seen = {}
    for row in REG:
        for a in [x.strip() for x in row["aliases"].split("|") if x.strip()]:
            assert a not in seen, f"alias 重复: {a}"
            seen[a] = row["experiment_id"]


# ---------- Resolver (任务书 §11) ----------
def test_resolver_canonical_self():
    for row in REG:
        if row["id_status"] == "canonical":
            assert resolve_experiment_id(row["experiment_id"], REG) == row["experiment_id"]


def test_resolver_alias_one_step():
    cases = {"P5-A": "P5-03", "MAG-Gate": "P5-03", "P7-AMP": "P7-01",
             "SUB-v8": "S-08", "P0.5-C": "P0-05", "P4-08A": "P4-10",
             "M02-T": "M-03", "P5-02I": "P5-02", "P2": "P2-01", "P1-1c": "P1-03"}
    for alias, target in cases.items():
        assert resolve_experiment_id(alias, REG) == target, alias


def test_resolver_legacy_self():
    for row in REG:
        if row["id_status"] == "legacy":
            assert resolve_experiment_id(row["experiment_id"], REG) == row["experiment_id"]


def test_resolver_unknown_raises():
    with pytest.raises(KeyError):
        resolve_experiment_id("DOES-NOT-EXIST", REG)
    with pytest.raises(KeyError):
        resolve_experiment_id("TEMP-01", REG)


# ---------- Allocator (任务书 §15/§16/§17/§37) ----------
def test_allocator_monotonic_no_gap_reuse():
    # 用副本避免污染
    ids = [r["experiment_id"] for r in REG]
    for series in ("P0","P1","P5","P6R","C","M","S"):
        nums = []
        for i in ids:
            if re.fullmatch(re.escape(series) + r"-\d{2}[a-z]?", i):
                nums.append(int(i.split("-")[1][:2]))
        if nums:
            nxt = allocate_next_id(REG, series)
            assert int(nxt.split("-")[1]) == max(nums) + 1, (series, nxt, max(nums))


def test_allocator_never_reuses():
    new_id = allocate_next_id(REG, "P5")
    assert new_id not in {r["experiment_id"] for r in REG}


def test_allocator_arm_requires_existing_parent():
    with pytest.raises(KeyError):
        allocate_next_arm(REG, "P5-99")
    arm = allocate_next_arm(REG, "P5-03")
    assert re.fullmatch(r"P5-03[a-z]", arm)
    assert arm not in {r["experiment_id"] for r in REG}


def test_allocator_invalid_series():
    with pytest.raises(ValueError):
        allocate_next_id(REG, "P8")
    with pytest.raises(ValueError):
        allocate_next_id(REG, "P5R")


# ---------- Lifecycle (任务书 §21/§22) ----------
def test_status_enum():
    for row in REG:
        assert row["status"] in VALID_STATUS, row["experiment_id"]


def test_decision_enum():
    for row in REG:
        assert row["decision"] in VALID_DECISIONS, row["experiment_id"]


def test_status_decision_separated():
    # completed + RED 合法; completed 必须有 decision 值 (NA 允许: 纯构建步骤如 P1-01)
    for row in REG:
        if row["status"] == "completed":
            assert row["decision"] in VALID_DECISIONS, row["experiment_id"]
    # planned/running 必须 NA (未裁决)
    for row in REG:
        if row["status"] in ("planned", "running"):
            assert row["decision"] == "NA", row["experiment_id"]


def test_red_requires_failure_reason():
    for row in REG:
        if row["decision"] == "RED":
            assert row["failure_reason"], row["experiment_id"]


# ---------- Filesystem (任务书 §35 双向一致) ----------
def test_directories_and_readmes_exist():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for row in REG:
        phdir = os.path.join(root, "experiments", row["phase"])
        if not os.path.isdir(phdir):
            continue
        found = [d for d in os.listdir(phdir) if d.startswith(row["experiment_id"] + "_")]
        assert found, f"{row['experiment_id']}: no directory in {phdir}"
        assert os.path.exists(os.path.join(phdir, found[0], "README.md")), row["experiment_id"]


def test_no_orphan_directories():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_root = os.path.join(root, "experiments")
    ids = {r["experiment_id"] for r in REG}
    for ph in os.listdir(exp_root):
        phdir = os.path.join(exp_root, ph)
        if not os.path.isdir(phdir) or ph.startswith("_"):
            continue
        for d in os.listdir(phdir):
            if d.startswith("_"):
                continue
            rid = re.split(r"_", d)[0]
            assert rid in ids, f"orphan directory: {ph}/{d}"


def test_script_paths_exist():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for row in REG:
        for sp in [x.strip() for x in row["script_path"].split() if x.strip()]:
            if sp.startswith("scripts/") and "*" not in sp and sp.endswith(".py"):
                assert os.path.exists(os.path.join(root, sp)), f"{row['experiment_id']}: {sp}"


# ---------- 研究质量 (任务书 §34) ----------
def test_completed_has_conclusion():
    for row in REG:
        if row["status"] == "completed" and row["decision"] in ("GREEN", "RED"):
            assert row["conclusion"], row["experiment_id"]


def test_registry_schema_version():
    meta = load_meta()
    assert meta.get("schema_version") == 2
    assert meta.get("experiment_id_spec_version") == "1.0"
