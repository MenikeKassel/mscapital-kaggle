import numpy as np
import pandas as pd

from mscapital.models.realmlp import (
    CleanRealMLPPreprocessor,
    RQKMeansEncoder,
    RealMLPConfig,
    _build_torch_classes,
    _environment_versions,
    _parameter_groups,
    flat_anneal,
    load_frame,
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
