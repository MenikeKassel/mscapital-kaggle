# -*- coding: utf-8 -*-
"""SSOT schema v3 / Experiment ID Spec v1.1 regression tests."""
import csv
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from mscapital.experiment_registry import (
    CANONICAL_RE, LEGACY_ALLOWLIST, REQUIRED_COLUMNS, VALID_DECISIONS,
    VALID_EVIDENCE_STATES, VALID_FAILURE_CATEGORIES, VALID_OWNERSHIP,
    VALID_PROVENANCE_STATES, VALID_RECORD_KINDS, VALID_STATUS,
    allocate_next_arm, allocate_next_id, audit_registry, build_alias_index,
    load_meta, load_registry, resolve_experiment_id,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = load_registry()

def test_exact_migration_counts_and_schema():
    assert len(REG) == 116
    assert sum(r["id_status"] == "legacy" for r in REG) == 15
    assert sum(r["id_status"] == "canonical" for r in REG) == 101
    assert set(REQUIRED_COLUMNS).issubset(REG[0])
    meta = load_meta()
    assert meta["schema_version"] == 3
    assert meta["experiment_id_spec_version"] == "1.1"

def test_ids_and_alias_migration():
    assert all(CANONICAL_RE.fullmatch(r["experiment_id"]) for r in REG if r["id_status"] == "canonical")
    assert all(r["experiment_id"] in LEGACY_ALLOWLIST for r in REG if r["id_status"] == "legacy")
    assert len({r["experiment_id"] for r in REG}) == 116
    expected = {"P8-01A": "P8-01", "P9-NC": "P9-01", "P9-NEUT": "P9-02", "P9-DG": "P9-03", "P10-PROD-RQ": "P10-01", "P6-R20260821": "P6-02", "BLSM-G0": "P11-01"}
    for old, new in expected.items(): assert resolve_experiment_id(old, REG) == new

def test_taxonomy_enums_and_route_coverage():
    for r in REG:
        assert r["status"] in VALID_STATUS
        assert r["decision"] in VALID_DECISIONS
        assert r["record_kind"] in VALID_RECORD_KINDS
        assert r["evidence_state"] in VALID_EVIDENCE_STATES
        assert r["ownership"] in VALID_OWNERSHIP
        assert r["failure_category"] in VALID_FAILURE_CATEGORIES
        assert r["provenance_state"] in VALID_PROVENANCE_STATES
        assert r["route_id"]
        if r["decision"] == "RED": assert r["failure_reason"]

def test_alias_one_step_and_unique():
    idx = build_alias_index(REG); ids = {r["experiment_id"] for r in REG}; seen = set()
    for alias, target in idx.items():
        if alias == target: continue
        assert alias not in ids
        assert idx[target] == target
        assert alias not in seen; seen.add(alias)

def test_allocator_v11():
    for series in ("P0", "P1", "P5", "P8", "P11", "P6R", "C", "M", "S"):
        nums = [int(r["experiment_id"].split("-")[1][:2]) for r in REG if re.fullmatch(re.escape(series) + r"-\d{2}[a-z]?", r["experiment_id"])]
        if nums: assert int(allocate_next_id(REG, series).split("-")[1]) == max(nums) + 1
    assert re.fullmatch(r"P5-03[a-z]", allocate_next_arm(REG, "P5-03"))
    with pytest.raises(ValueError): allocate_next_id(REG, "P12")
    with pytest.raises(ValueError): allocate_next_id(REG, "P5R")

def test_lifecycle_and_build_na():
    for r in REG:
        if r["status"] in ("planned", "running"): assert r["decision"] == "NA"
        if r["status"] == "completed" and r["record_kind"] not in {"build", "protocol", "diagnostic", "audit"}:
            assert r["decision"] != "NA"
    assert next(r for r in REG if r["experiment_id"] == "P1-01")["decision"] == "NA"

def test_directories_and_paths():
    for r in REG:
        phdir = os.path.join(ROOT, "experiments", {"C_clean":"C_clean_baseline", "C_clean_baseline":"C_clean_baseline", "P4_hidden":"P4_hidden_info"}.get(r["phase"], r["phase"]))
        found = [d for d in os.listdir(phdir) if d.startswith(r["experiment_id"] + "_")]
        assert found and os.path.exists(os.path.join(phdir, found[0], "README.md"))
        for p in r["script_path"].split("|"):
            p = p.strip()
            if p.startswith("scripts/") and p.endswith(".py") and "*" not in p: assert os.path.exists(os.path.join(ROOT, p))

def test_routes_and_submissions_reference_existing_ids():
    ids = {r["experiment_id"] for r in REG}
    routes = json.loads(open(os.path.join(ROOT, "experiments", "routes.yaml"), encoding="utf-8").read())
    assert len(routes) == 20
    assert all(set(x["evidence_ids"]).issubset(ids) for x in routes)
    with open(os.path.join(ROOT, "submissions", "registry.csv"), encoding="utf-8-sig", newline="") as f:
        subs = list(csv.DictReader(f))
    assert len(subs) >= 11 and all(s["experiment_id"] in ids for s in subs)

def test_audit_is_clean():
    a = audit_registry(REG, ROOT)
    assert a["total_registry_entries"] == 116
    assert a["canonical_count"] == 101 and a["legacy_count"] == 15
    for key in ("invalid_unclassified", "alias_collision_count", "missing_directory", "missing_script", "missing_report", "decision_invalid", "record_kind_invalid", "evidence_state_invalid", "route_missing", "orphan_experiments"):
        assert not a[key], (key, a[key])
