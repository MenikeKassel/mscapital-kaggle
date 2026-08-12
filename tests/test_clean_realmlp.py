import numpy as np
import pandas as pd
import hashlib
import json

from mscapital.models.realmlp import (
    CleanRealMLPPreprocessor,
    PreparedFrame,
    RQKMeansEncoder,
    RealMLPConfig,
    TrainResult,
    _build_torch_classes,
    _environment_versions,
    _parameter_groups,
    compare_inner_diagnostics,
    compare_outer_experiments,
    flat_anneal,
    load_frame,
    run_outer,
    summarize_outer,
    uncentered_cosine_torch,
)


def test_preprocessor_is_training_fold_only_and_handles_quantile_ood():
    rng = np.random.default_rng(123)
    n = 160
    train = pd.DataFrame(
        {
            "f_signal": np.linspace(-1, 1, n),
            "f_duplicate": np.linspace(-1, 1, n),
            "f_high_card": rng.permutation(n).astype(float),
            "f_constant": 1.0,
            "t_large_sell_95": (np.arange(n) % 3).astype(float),
        }
    )
    target = train.f_signal.to_numpy() + rng.normal(0, 0.001, n)
    cfg = RealMLPConfig(quantile_bins=40)
    pre = CleanRealMLPPreprocessor(tuple(train.columns), cfg).fit(train, target)
    assert "f_constant" not in pre.selected_numeric
    assert len(pre.selected_numeric) >= 1
    assert "f_high_card" in pre.quantile_edges
    state_before = pre.state_hash
    valid = train.iloc[:3].copy()
    valid.loc[:, "f_high_card"] = [10_000.0, -10_000.0, np.nan]
    valid.loc[:, "t_large_sell_95"] = [99, np.nan, 1]
    numeric, categorical = pre.transform(valid)
    assert numeric.shape[0] == categorical.shape[0] == 3
    assert np.isfinite(numeric).all()
    assert categorical[0, 0] == 3  # unknown category sentinel
    assert categorical[1, 0] == 3  # missing category sentinel
    assert pre.state_hash == state_before


def test_half_mask_and_legacy_optimizer_grouping():
    torch, _, _, model_cls = _build_torch_classes()
    cfg = RealMLPConfig()
    model = model_cls(n_numeric=3, cat_dims=[3], cfg=cfg)
    mask = model.feature_mask.detach().cpu().numpy()
    assert mask.shape[0] == 16
    for member in range(min(16, mask.shape[1])):
        assert not mask[member, member]
        assert mask[member].sum() < mask.shape[1]
    groups = _parameter_groups(model, torch, cfg)
    assert len(groups) == 5
    assert all(group["params"] for group in groups)
    assert cfg.rq_encoder_layers == 3
    assert cfg.rq_head_layers == 2
    assert len(model.code_heads) == 2

    full_model = model_cls(n_numeric=3, cat_dims=[3], cfg=RealMLPConfig(mask_mode="full"))
    full_mask = full_model.feature_mask.detach().cpu().numpy()
    no_mask_model = model_cls(n_numeric=3, cat_dims=[3], cfg=RealMLPConfig(mask_mode="none"))
    no_mask = no_mask_model.feature_mask.detach().cpu().numpy()
    for member in range(16):
        assert np.flatnonzero(~mask[member]).tolist() == list(range(member, mask.shape[1], 8))
        assert np.flatnonzero(~full_mask[member]).tolist() == list(range(member, full_mask.shape[1], 16))
    assert no_mask.all()

    corrected_cfg = RealMLPConfig(optimizer_grouping="first_ntp")
    corrected_model = model_cls(n_numeric=3, cat_dims=[3], cfg=corrected_cfg)
    corrected_groups = _parameter_groups(corrected_model, torch, corrected_cfg)
    corrected_first = corrected_groups[2]["params"]
    first_ntp_weight = dict(corrected_model.named_parameters())["shared.2.weight"]
    assert len(corrected_first) == 1
    assert corrected_first[0] is first_ntp_weight


def test_uncentered_cosine_zero_norm_is_finite_and_schedule_is_explicit():
    torch, _, _, _ = _build_torch_classes()
    value = uncentered_cosine_torch(torch.zeros(4), torch.ones(4), torch)
    assert float(value) == 0.0
    assert flat_anneal(1.0, 0.0) == 1.0
    assert flat_anneal(1.0, 0.5) == 1.0
    assert flat_anneal(1.0, 1.0) == 0.0


def test_environment_manifest_has_runtime_and_accelerator():
    environment = _environment_versions()
    assert environment["python"]
    assert environment["packages"]["numpy"]
    assert "cuda_available" in environment["accelerator"]


def test_rq_fit_is_refit_dependent():
    first = np.linspace(-1, 1, 30)
    second = np.linspace(10, 12, 30)
    left = RQKMeansEncoder(3, 3).fit(first)
    right = RQKMeansEncoder(3, 3).fit(second)
    assert not np.array_equal(left.encode(first), right.encode(first))
    assert left.encode(first).shape == (30, 3)


def test_load_frame_requires_exact_id_month_and_target_alignment(tmp_path):
    features = pd.DataFrame(
        {
            "sample_id": [2, 1],
            "month": [1, 0],
            "target": [0.2, 0.1],
            "feature": [20.0, 10.0],
        }
    )
    labels = features[["sample_id", "month", "target"]].copy()
    feature_path = tmp_path / "features.parquet"
    label_path = tmp_path / "labels.feather"
    features.to_parquet(feature_path)
    labels.to_feather(label_path)

    frame = load_frame(feature_path, label_path)
    assert frame.sample_id.tolist() == [1, 2]
    assert frame.month.tolist() == [0, 1]
    assert frame.target.tolist() == [0.1, 0.2]

    labels.loc[0, "month"] = 9
    labels.to_feather(label_path)
    with np.testing.assert_raises_regex(ValueError, "month columns disagree"):
        load_frame(feature_path, label_path)

    labels = features[["sample_id", "month", "target"]].copy()
    labels.loc[0, "sample_id"] = 99
    labels.to_feather(label_path)
    with np.testing.assert_raises_regex(ValueError, "same sample_id set"):
        load_frame(feature_path, label_path)


def test_load_frame_injects_label_month_when_feature_table_omits_it(tmp_path):
    features = pd.DataFrame(
        {"sample_id": [2, 1], "target": [0.2, 0.1], "feature": [20.0, 10.0]}
    )
    labels = pd.DataFrame(
        {"sample_id": [1, 2], "month": [0, 1], "target": [0.1, 0.2]}
    )
    feature_path = tmp_path / "features.parquet"
    label_path = tmp_path / "labels.feather"
    features.to_parquet(feature_path)
    labels.to_feather(label_path)

    frame = load_frame(feature_path, label_path)
    assert frame.sample_id.tolist() == [1, 2]
    assert frame.month.tolist() == [0, 1]
    assert frame.target.tolist() == [0.1, 0.2]

    labels.loc[0, "target"] = 9.0
    labels.to_feather(label_path)
    with np.testing.assert_raises_regex(ValueError, "target columns disagree"):
        load_frame(feature_path, label_path)


def test_outer_rewrite_cannot_change_inner_or_refit_state(monkeypatch, tmp_path):
    months = np.repeat(np.arange(71), 4)
    sample_id = np.arange(months.size)
    target = np.sin(sample_id / 7.0) * 0.01
    features = pd.DataFrame(
        {
            "signal": target + np.cos(sample_id / 5.0) * 0.001,
            "other": np.linspace(-1.0, 1.0, sample_id.size),
            "t_large_sell_95": (sample_id % 3).astype(float),
        }
    )
    cfg = RealMLPConfig(epochs=10, quantile_bins=8)
    captures: list[tuple[str, str, str]] = []

    def digest(value):
        array = np.ascontiguousarray(value)
        return hashlib.sha256(array.tobytes()).hexdigest()

    def fake_inner(x_train, c_train, y_train, x_tune, c_tune, y_tune, config):
        rq_hash = digest(RQKMeansEncoder(3, 3).fit(y_train).encode(y_train))
        captures.append(("inner", digest(x_train), rq_hash))
        predictions = np.full(y_tune.shape, 0.001, dtype=np.float32)
        return TrainResult(predictions, [{"epoch": 4.0}], 4, 4, 0.4, 1)

    def fake_refit(x_train, c_train, y_train, x_valid, c_valid, progress, config):
        rq_hash = digest(RQKMeansEncoder(3, 3).fit(y_train).encode(y_train))
        captures.append(("refit", digest(x_train), rq_hash))
        return np.full(x_valid.shape[0], 0.001, dtype=np.float32), [{"epoch": 4.0}], 4, progress, rq_hash

    monkeypatch.setattr("mscapital.models.realmlp.train_inner", fake_inner)
    monkeypatch.setattr("mscapital.models.realmlp._train_refit_predict", fake_refit)
    base = PreparedFrame(sample_id, months.astype(np.int16), target.copy(), features.copy())
    first = run_outer(base, "PSEUDO", cfg, tmp_path / "first")
    first_captures = tuple(captures)
    captures.clear()

    changed_features = features.copy()
    outer = months >= 33
    changed_features.loc[outer, ["signal", "other"]] = 1e9
    changed_target = target.copy()
    changed_target[outer] = -999.0
    changed = PreparedFrame(sample_id, months.astype(np.int16), changed_target, changed_features)
    second = run_outer(changed, "PSEUDO", cfg, tmp_path / "second")

    assert first["best_progress"] == second["best_progress"] == 0.4
    assert first["preprocessing"]["inner_state_hash"] == second["preprocessing"]["inner_state_hash"]
    assert first["preprocessing"]["refit_state_hash"] == second["preprocessing"]["refit_state_hash"]
    assert first_captures == tuple(captures)


def test_summary_uses_required_public_report_filenames(tmp_path):
    root = tmp_path / "clean-realmlp-v2a"
    for outer in ("PSEUDO", "H2", "T3", "T4"):
        directory = root / outer
        directory.mkdir(parents=True)
        from mscapital.splits import NESTED_SPLITS

        split = NESTED_SPLITS[outer]
        months = np.arange(split.outer_valid.start, split.outer_valid.end + 1, dtype=np.int16)
        manifest = {
            "experiment_id": f"clean-realmlp-v2a-{outer.lower()}",
            "status": "complete",
            "train_months": list(split.refit_train.as_tuple()),
            "valid_months": list(split.outer_valid.as_tuple()),
            "scores": {"cosine_uncentered": 0.1},
            "best_step": 10,
            "best_progress": 1.0,
            "runtime_seconds": 1.0,
            "diagnostics": {
                "pearson": 0.1,
                "nan_or_inf": 0,
                "prediction": {"mean": 0.0, "std": 1.0, "nan_or_inf": 0},
                "target": {"std": 1.0},
                "n_outer_valid": len(months),
            },
        }
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        np.savez_compressed(
            directory / "predictions.npz",
            sample_id=np.arange(len(months)),
            month=months,
            target=np.ones(len(months)),
            pred=np.ones(len(months)),
            split=np.full(len(months), f"{outer}:outer_valid"),
        )

    summarize_outer(tmp_path)
    assert (tmp_path / "clean_realmlp_v2a_report.json").exists()
    assert (tmp_path / "clean_realmlp_v2a_report.md").exists()

    bad = json.loads((root / "T4" / "manifest.json").read_text(encoding="utf-8"))
    bad["valid_months"] = [51, 60]
    (root / "T4" / "manifest.json").write_text(json.dumps(bad), encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "valid months"):
        summarize_outer(tmp_path)


def test_c2_comparison_requires_alignment_and_applies_frozen_gate(tmp_path):
    rng = np.random.default_rng(42)
    for outer in ("PSEUDO", "H2", "T3", "T4"):
        target = rng.normal(size=200)
        baseline = target + rng.normal(scale=2.0, size=200)
        candidate = target + rng.normal(scale=1.0, size=200)
        payload = {
            "sample_id": np.arange(200),
            "month": np.full(200, 1),
            "target": target,
            "split": np.full(200, f"{outer}:outer_valid"),
        }
        for experiment, prediction in (("baseline", baseline), ("candidate", candidate)):
            directory = tmp_path / experiment / outer
            directory.mkdir(parents=True)
            np.savez_compressed(directory / "predictions.npz", **payload, pred=prediction)

    report = compare_outer_experiments(tmp_path, "baseline", "candidate")
    assert report["gate"]["passed"] is True
    assert report["gate"]["positive_outers"] == 4
    assert (tmp_path / "candidate_vs_baseline.json").exists()
    assert (tmp_path / "candidate_vs_baseline.md").exists()

    broken = np.load(tmp_path / "candidate" / "T4" / "predictions.npz")
    payload = {key: broken[key] for key in broken.files}
    payload["sample_id"] = payload["sample_id"][::-1]
    np.savez_compressed(tmp_path / "candidate" / "T4" / "predictions.npz", **payload)
    with np.testing.assert_raises_regex(ValueError, "sample_id"):
        compare_outer_experiments(tmp_path, "baseline", "candidate")


def test_c2_inner_screening_uses_only_registered_histories(tmp_path):
    from mscapital.splits import NESTED_SPLITS

    for outer in ("PSEUDO", "H2", "T3"):
        split = NESTED_SPLITS[outer]
        baseline_dir = tmp_path / "baseline" / outer
        candidate_dir = tmp_path / "candidate" / outer
        baseline_dir.mkdir(parents=True)
        candidate_dir.mkdir(parents=True)
        (baseline_dir / "training_history.json").write_text(
            json.dumps({"inner": [{"epoch": 10, "tune_cosine_uncentered": 0.10}]}),
            encoding="utf-8",
        )
        (candidate_dir / "training_history.json").write_text(
            json.dumps(
                {
                    "inner": [
                        {"epoch": 10, "tune_cosine_uncentered": 0.099},
                        {"epoch": 20, "tune_cosine_uncentered": 0.102},
                    ]
                }
            ),
            encoding="utf-8",
        )
        (candidate_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "train_months": list(split.inner_train.as_tuple()),
                    "valid_months": list(split.inner_tune.as_tuple()),
                }
            ),
            encoding="utf-8",
        )

    report = compare_inner_diagnostics(tmp_path, "baseline", "candidate")
    assert report["gate"]["passed"] is True
    assert report["gate"]["positive_inner"] == 3
    assert all(row["candidate_best_epoch"] == 20 for row in report["outer"])
