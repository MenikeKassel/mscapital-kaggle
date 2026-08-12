"""Leakage-safe Clean RealMLP-v2a training.

This module ports the small RealMLP/RQ model used by the legacy PSEUDO run,
but makes every learned preprocessing state fold-local.  The public API is
deliberately split into three layers: ``CleanRealMLPPreprocessor`` (numpy),
``train_inner``/``train_refit`` (torch), and ``run_outer`` (temporal
orchestration).  Importing :mod:`mscapital` does not require torch; the model
backend is loaded only when a training function is called.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..artifacts import ExperimentManifest, feature_hash, git_sha, save_predictions
from ..diagnostics import prediction_diagnostics
from ..metrics import cosine_uncentered
from ..splits import NESTED_SPLITS, NestedSplit


def _stable_file_fingerprint(path: str | Path, sample_bytes: int = 1 << 20) -> str:
    """Fingerprint content samples without embedding workstation paths/times."""

    target = Path(path)
    stat = target.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("ascii"))
    with target.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if stat.st_size > sample_bytes:
            handle.seek(max(0, stat.st_size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Clean RealMLP requires torch; install the models runtime") from exc
    return torch, nn, F


@dataclass(frozen=True)
class RealMLPConfig:
    seed: int = 2026
    n_ens: int = 16
    embed_dim: int = 6
    onehotmax: int = 10
    learning_rate: float = 1e-3
    epochs: int = 10
    train_batch_size: int = 256
    eval_batch_size: int = 2048
    hidden_dim: int = 24
    dropout: float = 0.01
    ema_decay: float = 0.998
    gradient_clip: float = 1.0
    target_round: int | None = 4
    quantile_bins: int = 40
    correlation_threshold: float = 0.90
    low_target_correlation: float = 0.0001
    rq_encoder_layers: int = 3
    rq_head_layers: int = 2
    rq_vocab_size: int = 3
    lambda_cos: float = 0.01
    lambda_rq: float = 0.1
    label_noise_std: float = 0.005
    categorical_columns: tuple[str, ...] = ("t_large_sell_95",)
    device: str = "auto"
    max_rows_per_month: int | None = None
    mask_mode: str = "half"
    optimizer_grouping: str = "legacy"

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.mask_mode not in {"half", "full", "none"}:
            raise ValueError("mask_mode must be half, full, or none")
        if self.optimizer_grouping not in {"legacy", "first_ntp"}:
            raise ValueError("optimizer_grouping must be legacy or first_ntp")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RealMLPConfig":
        fields = {field.name for field in cls.__dataclass_fields__.values()}
        values = {key: value for key, value in mapping.items() if key in fields}
        if "categorical_columns" in values:
            values["categorical_columns"] = tuple(values["categorical_columns"])
        return cls(**values)


@dataclass
class PreparedFrame:
    sample_id: np.ndarray
    month: np.ndarray
    target: np.ndarray
    features: Any


@dataclass
class PreprocessorState:
    raw_features: tuple[str, ...]
    selected_numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    quantile_edges: dict[str, list[float]]
    medians: dict[str, float]
    scales: dict[str, float]
    categorical_vocab: dict[str, list[str]]
    dropped_features: tuple[str, ...]
    state_hash: str


class RobustScaleSmoothClip:
    """The legacy robust scale + smooth clip transform, with NaN handling."""

    def __init__(self) -> None:
        self.median: np.ndarray | None = None
        self.factors: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "RobustScaleSmoothClip":
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("robust scaler expects a 2-D array")
        finite = np.isfinite(x)
        med = np.zeros(x.shape[1], dtype=np.float64)
        filled = x.copy()
        for j in range(x.shape[1]):
            col = x[:, j]
            valid = col[np.isfinite(col)]
            med[j] = float(np.median(valid)) if valid.size else 0.0
            filled[~finite[:, j], j] = med[j]
        qdiff = np.quantile(filled, 0.75, axis=0) - np.quantile(filled, 0.25, axis=0)
        zero = qdiff == 0.0
        if np.any(zero):
            qdiff[zero] = 0.5 * (np.max(filled, axis=0)[zero] - np.min(filled, axis=0)[zero])
        factors = 1.0 / (qdiff + 1e-30)
        factors[qdiff == 0.0] = 0.0
        self.median = med
        self.factors = factors
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.median is None or self.factors is None:
            raise RuntimeError("fit must be called before transform")
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.median.size:
            raise ValueError("transform shape does not match fitted scaler")
        filled = np.where(np.isfinite(x), x, self.median[None, :])
        scaled = self.factors[None, :] * (filled - self.median[None, :])
        return (scaled / np.sqrt(1.0 + (scaled / 3.0) ** 2)).astype(np.float32)


def _stable_key(value: object) -> str | None:
    if value is None:
        return None
    try:
        if bool(np.isnan(value)):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


class CleanRealMLPPreprocessor:
    """Fit-only-on-training-fold feature selection and transforms."""

    def __init__(self, feature_names: Sequence[str], config: RealMLPConfig) -> None:
        self.feature_names = tuple(feature_names)
        self.config = config
        self.categorical = tuple(c for c in config.categorical_columns if c in self.feature_names)
        self.numeric_candidates = tuple(c for c in self.feature_names if c not in self.config.categorical_columns)
        self.selected_numeric: tuple[str, ...] = ()
        self.quantile_edges: dict[str, np.ndarray] = {}
        self.medians: dict[str, float] = {}
        self.scaler = RobustScaleSmoothClip()
        self.cat_vocab: dict[str, dict[str, int]] = {}
        self._fitted = False
        self.state: PreprocessorState | None = None

    @staticmethod
    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2:
            return 0.0
        xv, yv = x[valid], y[valid]
        sx, sy = float(np.std(xv)), float(np.std(yv))
        if sx == 0.0 or sy == 0.0:
            return 0.0
        return float(np.dot(xv - xv.mean(), yv - yv.mean()) / (valid.sum() * sx * sy))

    def fit(self, frame: Any, target: np.ndarray) -> "CleanRealMLPPreprocessor":
        y = np.asarray(target, dtype=np.float64).reshape(-1)
        if len(frame) != y.size:
            raise ValueError("target length does not match training frame")
        if not np.isfinite(y).all():
            raise ValueError("training target contains NaN/Inf")
        self.categorical = tuple(
            name for name in self.categorical
            if len({_stable_key(value) for value in frame[name] if _stable_key(value) is not None}) > 1
        )
        self.numeric_candidates = tuple(c for c in self.feature_names if c not in self.config.categorical_columns)
        numeric = np.asarray(frame[list(self.numeric_candidates)], dtype=np.float64)
        if numeric.ndim != 2:
            raise ValueError("numeric feature matrix must be two-dimensional")

        target_corr: dict[str, float] = {}
        filled = numeric.copy()
        for j, name in enumerate(self.numeric_candidates):
            col = filled[:, j]
            valid = np.isfinite(col)
            med = float(np.median(col[valid])) if valid.any() else 0.0
            col[~valid] = med
            target_corr[name] = abs(self._corr(numeric[:, j], y))

        keep = {name for name, corr in target_corr.items() if corr >= self.config.low_target_correlation}
        # The old filter is greedy over strongest pairs; reproduce that policy
        # while keeping the matrix in float32 to avoid a multi-GB float64 copy.
        if keep:
            names = [name for name in self.numeric_candidates if name in keep]
            matrix = np.asarray(filled[:, [self.numeric_candidates.index(name) for name in names]], dtype=np.float32)
            matrix -= matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
            std = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
            std[std == 0.0] = 1.0
            matrix /= std
            corr = (matrix.T @ matrix) / max(matrix.shape[0], 1)
            pairs: list[tuple[float, str, str]] = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    value = abs(float(corr[i, j]))
                    if value >= self.config.correlation_threshold:
                        pairs.append((value, names[i], names[j]))
            pairs.sort(reverse=True)
            dropped: set[str] = set()
            for _, left, right in pairs:
                if left in dropped or right in dropped:
                    continue
                dropped.add(right if target_corr[left] >= target_corr[right] else left)
            keep -= dropped
        self.selected_numeric = tuple(name for name in self.numeric_candidates if name in keep)
        if not self.selected_numeric:
            raise ValueError("fold feature selection removed all numeric features")

        selected_matrix = np.asarray(frame[list(self.selected_numeric)], dtype=np.float64)
        scale_columns: list[np.ndarray] = []
        quantile_edges: dict[str, np.ndarray] = {}
        for j, name in enumerate(self.selected_numeric):
            col = selected_matrix[:, j]
            finite = col[np.isfinite(col)]
            self.medians[name] = float(np.median(finite)) if finite.size else 0.0
            if np.unique(finite).size > 100:
                if finite.size:
                    edges = np.quantile(finite, np.linspace(0.0, 1.0, self.config.quantile_bins + 1))
                    edges = np.unique(np.maximum.accumulate(edges))
                else:
                    edges = np.asarray([0.0, 1.0], dtype=np.float64)
                quantile_edges[name] = edges
            else:
                scale_columns.append(np.where(np.isfinite(col), col, self.medians[name]))
        self.quantile_edges = quantile_edges
        robust_names = [name for name in self.selected_numeric if name not in quantile_edges]
        if robust_names:
            self.scaler.fit(np.column_stack(scale_columns))
            for j, name in enumerate(robust_names):
                assert self.scaler.median is not None and self.scaler.factors is not None
                self.medians[name] = float(self.scaler.median[j])
                self._scaler_index = {n: i for i, n in enumerate(robust_names)}
        else:
            self.scaler = RobustScaleSmoothClip()
            self._scaler_index = {}

        for name in self.categorical:
            values = frame[name]
            mapping: dict[str, int] = {}
            for value in values:
                key = _stable_key(value)
                if key is not None and key not in mapping:
                    mapping[key] = len(mapping)
            self.cat_vocab[name] = mapping
        payload = {
            "raw": self.feature_names,
            "numeric": self.selected_numeric,
            "categorical": self.categorical,
            "quantile_edges": {k: v.tolist() for k, v in self.quantile_edges.items()},
            "medians": self.medians,
            "scales": ({name: float(self.scaler.factors[i]) for name, i in self._scaler_index.items()} if robust_names else {}),
            "cat_vocab": self.cat_vocab,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        self.state = PreprocessorState(
            raw_features=self.feature_names,
            selected_numeric=self.selected_numeric,
            categorical=self.categorical,
            quantile_edges={k: v.tolist() for k, v in self.quantile_edges.items()},
            medians=dict(self.medians),
            scales=payload["scales"],
            categorical_vocab={k: list(v) for k, v in self.cat_vocab.items()},
            dropped_features=tuple(name for name in self.numeric_candidates if name not in self.selected_numeric),
            state_hash=hashlib.sha256(encoded).hexdigest(),
        )
        self._fitted = True
        return self

    def transform(self, frame: Any) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("fit must be called before transform")
        n = len(frame)
        numeric = np.empty((n, len(self.selected_numeric)), dtype=np.float32)
        robust_names = [name for name in self.selected_numeric if name not in self.quantile_edges]
        if robust_names:
            raw = np.asarray(frame[robust_names], dtype=np.float64)
            numeric_robust = self.scaler.transform(raw)
            for j, name in enumerate(robust_names):
                numeric[:, self.selected_numeric.index(name)] = numeric_robust[:, j]
        for name, edges in self.quantile_edges.items():
            col = np.asarray(frame[name], dtype=np.float64)
            values = np.full(n, self.config.quantile_bins, dtype=np.float32)
            valid = np.isfinite(col)
            if valid.any():
                values[valid] = np.searchsorted(edges[1:-1], col[valid], side="right").astype(np.float32)
            numeric[:, self.selected_numeric.index(name)] = values
        cat = np.empty((n, len(self.categorical)), dtype=np.int64)
        for j, name in enumerate(self.categorical):
            mapping = self.cat_vocab[name]
            unknown = len(mapping)
            cat[:, j] = [unknown if (key := _stable_key(value)) is None else mapping.get(key, unknown) for value in frame[name]]
        return numeric, cat

    @property
    def state_hash(self) -> str:
        if self.state is None:
            raise RuntimeError("fit must be called before state_hash")
        return self.state.state_hash


def _set_seed(seed: int) -> None:
    torch, _, _ = _require_torch()
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RQKMeansEncoder:
    def __init__(self, n_layers: int, codebook_size: int) -> None:
        self.n_layers = int(n_layers)
        self.codebook_size = int(codebook_size)
        self.codebooks: list[Any] = []

    def fit(self, values: np.ndarray) -> "RQKMeansEncoder":
        try:
            from sklearn.cluster import KMeans
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("RQ encoding requires scikit-learn") from exc
        residual = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        if np.unique(residual).size < self.codebook_size:
            raise ValueError("RQ codebook has more clusters than unique training targets")
        self.codebooks = []
        for _ in range(self.n_layers):
            kmeans = KMeans(n_clusters=self.codebook_size, random_state=42, n_init=10)
            kmeans.fit(residual)
            self.codebooks.append(kmeans)
            codes = kmeans.predict(residual)
            residual = residual - kmeans.cluster_centers_[codes]
        return self

    def encode(self, values: np.ndarray) -> np.ndarray:
        if not self.codebooks:
            raise RuntimeError("fit must be called before encode")
        residual = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        codes: list[np.ndarray] = []
        for kmeans in self.codebooks:
            code = kmeans.predict(residual)
            codes.append(code)
            residual = residual - kmeans.cluster_centers_[code]
        return np.stack(codes, axis=1).astype(np.int64)

    @property
    def state_hash(self) -> str:
        if not self.codebooks:
            raise RuntimeError("fit must be called before state_hash")
        digest = hashlib.sha256()
        digest.update(f"{self.n_layers}|{self.codebook_size}".encode("ascii"))
        for codebook in self.codebooks:
            centers = np.asarray(codebook.cluster_centers_, dtype=np.float64)
            digest.update(str(centers.shape).encode("ascii"))
            digest.update(np.ascontiguousarray(centers).tobytes())
        return digest.hexdigest()


def _build_torch_classes():
    torch, nn, F = _require_torch()

    class ScalingLayer(nn.Module):
        def __init__(self, n_ens: int, n_features: int):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(n_ens, n_features))

        def forward(self, x):
            return x * self.scale[None, :, :]

    class CategoricalFeatureLayer(nn.Module):
        def __init__(self, n_ens: int, cat_dims: Sequence[int], embed_dim: int, onehotmax: int):
            super().__init__()
            self.n_ens = n_ens
            self.cat_dims = list(cat_dims)
            self.onehot_features = [i for i, dim in enumerate(cat_dims) if dim <= onehotmax]
            self.embed_features = [i for i, dim in enumerate(cat_dims) if dim > onehotmax]
            self.embed_dims = [cat_dims[i] for i in self.embed_features]
            if self.embed_features:
                self.combined_emb = nn.Embedding(sum(self.embed_dims) * n_ens, embed_dim, padding_idx=0)
                self.embed_offsets = []
                offset = 0
                for dim in self.embed_dims:
                    self.embed_offsets.append(offset)
                    offset += dim
                self.per_ens_offset = sum(self.embed_dims)

        def forward(self, x):
            batch, n_ens, _ = x.shape
            pieces = []
            if self.onehot_features:
                values = x[:, :, self.onehot_features].long()
                width = sum(self.cat_dims[i] for i in self.onehot_features)
                encoded = torch.zeros(batch, n_ens, width, device=x.device)
                start = 0
                for idx, feature_idx in enumerate(self.onehot_features):
                    dim = self.cat_dims[feature_idx]
                    encoded.scatter_(2, values[:, :, idx:idx + 1].clamp(0, dim - 1) + start, 1.0)
                    start += dim
                pieces.append(encoded)
            if self.embed_features:
                values = x[:, :, self.embed_features].long()
                ensemble_offset = torch.arange(n_ens, device=x.device) * self.per_ens_offset
                feature_offset = torch.tensor(self.embed_offsets, device=x.device)
                indices = values + feature_offset[None, None, :] + ensemble_offset[None, :, None]
                pieces.append(self.combined_emb(indices).reshape(batch, n_ens, -1))
            if not pieces:
                return torch.empty(batch, n_ens, 0, device=x.device)
            return torch.cat(pieces, dim=2)

    class PBLDEmbedding(nn.Module):
        def __init__(self, n_ens: int, n_features: int, hidden_dim: int = 24, out_dim: int = 3):
            super().__init__()
            self.n_ens = n_ens
            self.n_features = n_features
            self.w1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim) * 1.0)
            self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
            self.w2 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim, out_dim - 1) * (1.0 / np.sqrt(hidden_dim)))
            self.b2 = nn.Parameter(torch.randn(n_ens, n_features, out_dim - 1))
            self.act = nn.GELU()
            nn.init.uniform_(self.b1, -np.pi, np.pi)

        def forward(self, x):
            periodic = torch.cos(2 * np.pi * (x.unsqueeze(-1) * self.w1[None] + self.b1[None]))
            transformed = torch.einsum("bnfh,nfhd->bnfd", periodic, self.w2)
            transformed = self.act(transformed + self.b2[None])
            return torch.cat([x.unsqueeze(-1), transformed], dim=-1).reshape(x.shape[0], self.n_ens, -1)

    class NTPLinear(nn.Module):
        def __init__(self, n_ens: int, in_features: int, out_features: int, bias: bool = True):
            super().__init__()
            self.in_features = in_features
            self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
            self.bias = nn.Parameter(torch.randn(n_ens, out_features)) if bias else None

        def forward(self, x):
            out = torch.einsum("bni,nio->bno", x, self.weight) / np.sqrt(self.in_features)
            return out + self.bias[None] if self.bias is not None else out

    class RealMLPRQ(nn.Module):
        def __init__(self, n_numeric: int, cat_dims: Sequence[int], cfg: RealMLPConfig):
            super().__init__()
            self.n_ens = cfg.n_ens
            self.mask_mode = cfg.mask_mode
            self.cate = CategoricalFeatureLayer(cfg.n_ens, cat_dims, cfg.embed_dim, cfg.onehotmax)
            self.num_embed = PBLDEmbedding(cfg.n_ens, n_numeric, cfg.hidden_dim, 3)
            total_dim = n_numeric * 3 + sum(dim if dim <= cfg.onehotmax else cfg.embed_dim for dim in cat_dims)
            self.dropout = nn.Dropout(cfg.dropout)
            self.shared = nn.Sequential(
                nn.LayerNorm(total_dim),
                ScalingLayer(cfg.n_ens, total_dim),
                NTPLinear(cfg.n_ens, total_dim, 512), nn.GELU(), self.dropout,
                NTPLinear(cfg.n_ens, 512, 512), nn.GELU(), self.dropout,
                NTPLinear(cfg.n_ens, 512, 128), nn.GELU(), self.dropout,
            )
            self.code_heads = nn.ModuleList([NTPLinear(cfg.n_ens, 128, cfg.rq_vocab_size) for _ in range(cfg.rq_head_layers)])
            self.reg_head = NTPLinear(cfg.n_ens, 128, 1)
            self.register_buffer("feature_mask", self._create_mask(total_dim))

        def _create_mask(self, n_features: int):
            mask = torch.ones(self.n_ens, n_features, dtype=torch.bool)
            if self.mask_mode == "none":
                return mask
            stride = self.n_ens // 2 if self.mask_mode == "half" else self.n_ens
            for i in range(self.n_ens):
                mask[i, i::stride] = False
            return mask

        def forward(self, x_num, x_cat, return_codes: bool = False):
            x_num = x_num[:, None, :].expand(-1, self.n_ens, -1)
            x_cat = x_cat[:, None, :].expand(-1, self.n_ens, -1)
            combined = torch.cat([self.num_embed(x_num), self.cate(x_cat)], dim=2)
            combined = combined * self.feature_mask[None].to(combined.dtype)
            features = self.shared(combined)
            logits = [head(features) for head in self.code_heads]
            pred = self.reg_head(features)
            return (logits, pred) if return_codes else pred.mean(dim=1)

    return torch, nn, F, RealMLPRQ


def flat_anneal(init_value: float, progress: float, flat_ratio: float = 0.5) -> float:
    if progress < flat_ratio:
        return init_value
    return init_value * (1.0 - (progress - flat_ratio) / (1.0 - flat_ratio))


def uncentered_cosine_torch(pred, target, torch_module):
    p = pred.reshape(-1)
    y = target.reshape(-1)
    return (p * y).sum() / (p.norm() + 1e-8) / (y.norm() + 1e-8)


def _parameter_groups(model, torch_module, cfg: RealMLPConfig):
    scale, pbld, first, other, bias = [], [], [], [], []
    first_id = None
    target_name = "shared.0.weight" if cfg.optimizer_grouping == "legacy" else "shared.2.weight"
    for name, param in model.named_parameters():
        if target_name in name:
            first_id = id(param)
            break
    for name, param in model.named_parameters():
        if "scale" in name:
            scale.append(param)
        elif "num_embed" in name:
            pbld.append(param)
        elif first_id is not None and id(param) == first_id:
            first.append(param)
        elif "bias" in name:
            bias.append(param)
        else:
            other.append(param)
    groups = [
        {"params": scale, "lr": cfg.learning_rate * 20.0, "weight_decay": 1e-3},
        {"params": pbld, "lr": cfg.learning_rate * 0.093, "weight_decay": 1e-2},
        {"params": first, "lr": cfg.learning_rate, "weight_decay": 1e-3},
        {"params": other, "lr": cfg.learning_rate, "weight_decay": 1e-2},
        {"params": bias, "lr": cfg.learning_rate * 0.1, "weight_decay": 5e-3},
    ]
    if any(not group["params"] for group in groups):
        raise RuntimeError("legacy RealMLP parameter grouping produced an empty group")
    return groups


class _EMA:
    def __init__(self, model, decay: float, torch_module):
        self.model = model
        self.decay = decay
        self.torch = torch_module
        self.state = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}

    def update(self):
        with self.torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    self.state[name].mul_(self.decay).add_(p, alpha=1.0 - self.decay)

    def apply(self):
        original = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                original[name] = p.detach().clone()
                p.data.copy_(self.state[name])
        return original

    def restore(self, original):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(original[name])


@dataclass
class TrainResult:
    predictions: np.ndarray
    history: list[dict[str, float]]
    best_epoch: int
    best_step: int
    best_progress: float
    model_parameters: int
    rq_state_hash: str = ""


def _loss(model_pred, target, logits, codes, progress: float, cfg: RealMLPConfig, torch_module, F):
    pred = model_pred.squeeze(-1)
    expanded = target[:, None].expand_as(pred)
    weights = torch_module.where(expanded.abs() > 0.001, 0.5, 1.0)
    mse = (weights * (pred - expanded) ** 2).mean()
    cos = uncentered_cosine_torch(pred, expanded, torch_module)
    rq = torch_module.zeros((), device=pred.device)
    expanded_codes = codes[:, None, :].expand(-1, pred.shape[1], -1)
    for layer, layer_logits in enumerate(logits):
        rq = rq + F.cross_entropy(layer_logits.reshape(-1, layer_logits.shape[-1]), expanded_codes[:, :, layer].reshape(-1))
    rq = rq / len(logits)
    return mse + cfg.lambda_cos * (1.0 - cos) + flat_anneal(cfg.lambda_rq, progress) * rq, cos, mse, rq


def _predict(model, x_num: np.ndarray, x_cat: np.ndarray, cfg: RealMLPConfig, device: str, torch_module) -> np.ndarray:
    model.eval()
    device_obj = torch_module.device(device)
    result: list[np.ndarray] = []
    with torch_module.no_grad():
        for start in range(0, len(x_num), cfg.eval_batch_size):
            n = torch_module.from_numpy(x_num[start:start + cfg.eval_batch_size]).to(device_obj)
            c = torch_module.from_numpy(x_cat[start:start + cfg.eval_batch_size]).to(device_obj)
            result.append(model(n, c).reshape(-1).detach().cpu().numpy())
    return np.concatenate(result).astype(np.float32) if result else np.empty(0, dtype=np.float32)


def _make_optimizer(model, cfg, torch_module):
    return torch_module.optim.AdamW(_parameter_groups(model, torch_module, cfg), betas=(0.9, 0.98))


def train_inner(x_train, c_train, y_train, x_tune, c_tune, y_tune, cfg: RealMLPConfig) -> TrainResult:
    torch, _, F, RealMLPRQ = _build_torch_classes()
    _set_seed(cfg.seed)
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else ("cpu" if cfg.device == "auto" else cfg.device))
    rq = RQKMeansEncoder(cfg.rq_encoder_layers, cfg.rq_vocab_size).fit(y_train)
    train_codes = rq.encode(y_train)
    # One extra category is reserved for the fold-fitted unknown/missing
    # sentinel.  Its size is derived from train only; validation categories
    # must never alter the model vocabulary.
    cat_dims = [int(np.max(c_train[:, j]) + 2) if c_train.shape[1] else 1 for j in range(c_train.shape[1])]
    if c_train.shape[1]:
        cat_dims = [max(dim, 1) for dim in cat_dims]
    model = RealMLPRQ(x_train.shape[1], cat_dims, cfg).to(device)
    optimizer = _make_optimizer(model, cfg, torch)
    ema = _EMA(model, cfg.ema_decay, torch)
    x = torch.from_numpy(x_train).to(device)
    c = torch.from_numpy(c_train).to(device)
    y = torch.from_numpy(np.asarray(y_train, dtype=np.float32)).to(device)
    codes = torch.from_numpy(train_codes).to(device)
    total_batches = (len(y_train) + cfg.train_batch_size - 1) // cfg.train_batch_size
    total_steps = max(total_batches * cfg.epochs, 1)
    best_score = -float("inf")
    best_state = None
    history: list[dict[str, float]] = []
    best_epoch = 1
    best_step = total_batches
    for epoch in range(cfg.epochs):
        model.train()
        permutation = torch.randperm(len(y), device=device)
        sums = np.zeros(4, dtype=np.float64)
        batches = 0
        for batch_idx, start in enumerate(range(0, len(y), cfg.train_batch_size)):
            step = epoch * total_batches + batch_idx
            progress = min(step / total_steps, 1.0)
            for group, base in zip(optimizer.param_groups, (20.0, 0.093, 1.0, 1.0, 0.1)):
                group["lr"] = flat_anneal(cfg.learning_rate * base, progress)
            indices = permutation[start:start + cfg.train_batch_size]
            target = y[indices]
            noisy = target + torch.randn_like(target) * (cfg.label_noise_std * (1.0 - progress))
            optimizer.zero_grad(set_to_none=True)
            logits, pred = model(x[indices], c[indices], return_codes=True)
            loss, cos, mse, rq_loss = _loss(pred, noisy, logits, codes[indices], progress, cfg, torch, F)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
            ema.update()
            sums += np.asarray([loss.item(), cos.item(), mse.item(), rq_loss.item()])
            batches += 1
        original = ema.apply()
        tune_pred = _predict(model, x_tune, c_tune, cfg, str(device), torch)
        score = cosine_uncentered(tune_pred, y_tune)
        ema.restore(original)
        row = {"epoch": float(epoch + 1), "loss": float(sums[0] / max(batches, 1)), "train_cosine": float(sums[1] / max(batches, 1)), "mse": float(sums[2] / max(batches, 1)), "rq": float(sums[3] / max(batches, 1)), "tune_cosine_uncentered": float(score)}
        history.append(row)
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_step = (epoch + 1) * total_batches
            best_state = {name: value.detach().cpu().clone() for name, value in ema.state.items()}
    if best_state is None:
        raise RuntimeError("inner training did not produce a best model")
    for name, parameter in model.named_parameters():
        parameter.data.copy_(best_state[name].to(device))
    predictions = _predict(model, x_tune, c_tune, cfg, str(device), torch)
    return TrainResult(predictions, history, best_epoch, best_step, best_epoch / cfg.epochs, sum(p.numel() for p in model.parameters()), rq.state_hash)


def train_refit(x_train, c_train, y_train, progress_limit: float, cfg: RealMLPConfig) -> tuple[np.ndarray, list[dict[str, float]], int, int]:
    torch, _, F, RealMLPRQ = _build_torch_classes()
    _set_seed(cfg.seed)
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else ("cpu" if cfg.device == "auto" else cfg.device))
    rq = RQKMeansEncoder(cfg.rq_encoder_layers, cfg.rq_vocab_size).fit(y_train)
    codes_np = rq.encode(y_train)
    cat_dims = [int(np.max(c_train[:, j]) + 2) if c_train.shape[1] else 1 for j in range(c_train.shape[1])]
    model = RealMLPRQ(x_train.shape[1], cat_dims, cfg).to(device)
    optimizer = _make_optimizer(model, cfg, torch)
    ema = _EMA(model, cfg.ema_decay, torch)
    x = torch.from_numpy(x_train).to(device)
    c = torch.from_numpy(c_train).to(device)
    y = torch.from_numpy(np.asarray(y_train, dtype=np.float32)).to(device)
    codes = torch.from_numpy(codes_np).to(device)
    total_batches = (len(y_train) + cfg.train_batch_size - 1) // cfg.train_batch_size
    total_steps = max(total_batches * cfg.epochs, 1)
    stop_step = max(1, min(total_steps, int(math.ceil(progress_limit * total_steps))))
    history: list[dict[str, float]] = []
    step = 0
    while step < stop_step:
        permutation = torch.randperm(len(y), device=device)
        sums = np.zeros(4, dtype=np.float64)
        batches = 0
        for start in range(0, len(y), cfg.train_batch_size):
            if step >= stop_step:
                break
            progress = min(step / total_steps, 1.0)
            for group, base in zip(optimizer.param_groups, (20.0, 0.093, 1.0, 1.0, 0.1)):
                group["lr"] = flat_anneal(cfg.learning_rate * base, progress)
            indices = permutation[start:start + cfg.train_batch_size]
            target = y[indices]
            noisy = target + torch.randn_like(target) * (cfg.label_noise_std * (1.0 - progress))
            optimizer.zero_grad(set_to_none=True)
            logits, pred = model(x[indices], c[indices], return_codes=True)
            loss, cos, mse, rq_loss = _loss(pred, noisy, logits, codes[indices], progress, cfg, torch, F)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
            ema.update()
            sums += np.asarray([loss.item(), cos.item(), mse.item(), rq_loss.item()])
            batches += 1
            step += 1
        history.append({"epoch": float(len(history) + 1), "loss": float(sums[0] / max(batches, 1)), "train_cosine": float(sums[1] / max(batches, 1)), "mse": float(sums[2] / max(batches, 1)), "rq": float(sums[3] / max(batches, 1))})
    original = ema.apply()
    predictions = _predict(model, x_train, c_train, cfg, str(device), torch)
    ema.restore(original)
    return predictions, history, stop_step, stop_step / total_steps


def load_frame(train_path: str | Path, labels_path: str | Path, *, max_rows_per_month: int | None = None) -> PreparedFrame:
    import pandas as pd

    train_path, labels_path = Path(train_path), Path(labels_path)
    if train_path.suffix.lower() == ".parquet":
        features = pd.read_parquet(train_path)
    else:
        features = pd.read_feather(train_path)
    if labels_path.suffix.lower() == ".parquet":
        labels = pd.read_parquet(labels_path)
    else:
        labels = pd.read_feather(labels_path)
    if "sample_id" not in features.columns:
        raise ValueError("features must contain sample_id")
    required_labels = {"sample_id", "month", "target"}
    if not required_labels.issubset(labels.columns):
        raise ValueError(f"labels must contain {sorted(required_labels)}")
    if features.sample_id.duplicated().any() or labels.sample_id.duplicated().any():
        raise ValueError("sample_id must be unique in both feature and label files")
    comparison_columns = [name for name in ("month", "target") if name in features.columns]
    merged = features.merge(
        labels[["sample_id", "month", "target"]],
        on="sample_id",
        how="inner",
        suffixes=("_feature", "_label"),
        validate="one_to_one",
    )
    if len(merged) != len(features) or len(merged) != len(labels):
        raise ValueError("feature and label files must contain exactly the same sample_id set")
    for name in comparison_columns:
        feature_values = merged.pop(f"{name}_feature").to_numpy()
        label_values = merged.pop(f"{name}_label").to_numpy()
        if not np.array_equal(feature_values, label_values):
            raise ValueError(f"feature and label {name} columns disagree")
        merged[name] = label_values
    merged = merged.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if merged.sample_id.duplicated().any() or merged.month.isna().any():
        raise ValueError("sample_id must be unique and month must be complete")
    if max_rows_per_month is not None:
        pieces = [group.head(max_rows_per_month) for _, group in merged.groupby("month", sort=True)]
        merged = pd.concat(pieces, ignore_index=True).sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    feature_names = tuple(c for c in features.columns if c not in {"sample_id", "target", "month"})
    return PreparedFrame(merged.sample_id.to_numpy(), merged.month.to_numpy(dtype=np.int16), merged.target.to_numpy(dtype=np.float64), merged[list(feature_names)])


def _outer_report_md(name: str, result: Mapping[str, Any]) -> str:
    score = result["diagnostics"].get("cosine_uncentered")
    lines = [f"# Clean RealMLP-v2a - {name}", "", f"- outer score (uncentered cosine): `{score:.9f}`", f"- best epoch/progress: `{result['best_epoch']}` / `{result['best_progress']:.4f}`", f"- best step: `{result['best_step']}`", f"- model parameters: `{result['model_parameters']:,}`", "", "## Diagnostics", "", "```json", json.dumps(result["diagnostics"], indent=2, sort_keys=True), "```", ""]
    if result.get("legacy_correlation") is not None:
        lines.extend([f"- legacy PSEUDO prediction Pearson correlation: `{result['legacy_correlation']:.9f}`", ""])
    return "\n".join(lines)


def _environment_versions() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for distribution in ("numpy", "pandas", "scikit-learn", "torch", "pyarrow"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = "not-installed"
    accelerator: dict[str, Any] = {"cuda_available": False}
    try:
        import torch

        accelerator = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        }
    except (ImportError, RuntimeError):
        pass
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "accelerator": accelerator,
        "executable_name": Path(sys.executable).name,
    }


def run_inner_diagnostic(
    frame: PreparedFrame,
    outer_name: str,
    cfg: RealMLPConfig,
    output_dir: str | Path,
    *,
    experiment_id: str = "c2-realmlp-ceiling",
    data_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Run only the inner search for a C2 diagnostic without touching outer valid."""

    if outer_name not in NESTED_SPLITS:
        raise KeyError(f"unknown outer split {outer_name}")
    split = NESTED_SPLITS[outer_name]
    output = Path(output_dir) / outer_name
    output.mkdir(parents=True, exist_ok=True)
    months = frame.month
    train_mask = split.inner_train.contains(months)
    tune_mask = split.inner_tune.contains(months)
    for partition_name, partition_range, partition_mask in (
        ("inner_train", split.inner_train, train_mask),
        ("inner_tune", split.inner_tune, tune_mask),
    ):
        actual = set(int(value) for value in np.unique(months[partition_mask]))
        expected = set(range(partition_range.start, partition_range.end + 1))
        if actual != expected:
            raise ValueError(f"{outer_name} {partition_name} has incomplete month coverage")

    started = time.perf_counter()
    preprocessor = CleanRealMLPPreprocessor(tuple(frame.features.columns), cfg).fit(
        frame.features.loc[train_mask], frame.target[train_mask]
    )
    x_train, c_train = preprocessor.transform(frame.features.loc[train_mask])
    x_tune, c_tune = preprocessor.transform(frame.features.loc[tune_mask])
    train_target = frame.target[train_mask]
    if cfg.target_round is not None:
        train_target = np.round(train_target, cfg.target_round)
    result = train_inner(
        x_train,
        c_train,
        train_target,
        x_tune,
        c_tune,
        frame.target[tune_mask],
        cfg,
    )
    diagnostics = prediction_diagnostics(result.predictions, frame.target[tune_mask])
    diagnostics["pearson"] = (
        float(np.corrcoef(result.predictions, frame.target[tune_mask])[0, 1])
        if np.std(result.predictions) and np.std(frame.target[tune_mask])
        else 0.0
    )
    diagnostics.update(
        {
            "nan_or_inf": int((~np.isfinite(result.predictions)).sum()),
            "n_inner_train": int(train_mask.sum()),
            "n_inner_tune": int(tune_mask.sum()),
            "preprocessing_state_hash": preprocessor.state_hash,
            "rq_state_hash": result.rq_state_hash,
        }
    )
    runtime = time.perf_counter() - started
    save_predictions(
        output / "inner_predictions.npz",
        sample_id=frame.sample_id[tune_mask],
        month=months[tune_mask],
        target=frame.target[tune_mask],
        pred=result.predictions,
        split=np.full(result.predictions.size, f"{outer_name}:inner_tune"),
    )
    config_payload = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    manifest = ExperimentManifest(
        experiment_id=f"{experiment_id}-{outer_name.lower()}-inner",
        status="complete",
        git_sha=os.environ.get("MSCAP_GIT_SHA") or git_sha(),
        config_hash=hashlib.sha256(config_payload).hexdigest(),
        train_months=split.inner_train.as_tuple(),
        valid_months=split.inner_tune.as_tuple(),
        feature_hash=feature_hash(list(preprocessor.selected_numeric) + list(preprocessor.categorical)),
        best_step=result.best_step,
        best_progress=result.best_progress,
        runtime_seconds=runtime,
        scores={"cosine_uncentered": float(diagnostics["cosine_uncentered"])},
        diagnostics=diagnostics,
        environment=_environment_versions(),
    )
    manifest.data_fingerprints = {
        Path(path).name: _stable_file_fingerprint(path) for path in data_paths if Path(path).exists()
    }
    manifest.write(output)
    (output / "training_history.json").write_text(
        json.dumps({"inner": result.history}, indent=2), encoding="utf-8"
    )
    (output / "report.md").write_text(
        "\n".join(
            [
                f"# C2 RealMLP inner diagnostic - {outer_name}",
                "",
                f"- epochs scanned: `{cfg.epochs}`",
                f"- best epoch/progress: `{result.best_epoch}` / `{result.best_progress:.4f}`",
                f"- tune cosine: `{diagnostics['cosine_uncentered']:.9f}`",
                f"- runtime seconds: `{runtime:.1f}`",
                "",
                "This diagnostic never reads or scores the registered outer validation rows.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "outer": outer_name,
        "best_epoch": result.best_epoch,
        "best_progress": result.best_progress,
        "score": float(diagnostics["cosine_uncentered"]),
        "runtime_seconds": runtime,
        "artifact": str(output),
    }


def run_outer(frame: PreparedFrame, outer_name: str, cfg: RealMLPConfig, output_dir: str | Path, *, experiment_id: str = "clean-realmlp-v2a", data_paths: Sequence[str | Path] = (), legacy_pseudo_path: str | Path | None = None) -> dict[str, Any]:
    if outer_name not in NESTED_SPLITS:
        raise KeyError(f"unknown outer split {outer_name}")
    split: NestedSplit = NESTED_SPLITS[outer_name]
    output = Path(output_dir) / outer_name
    output.mkdir(parents=True, exist_ok=True)
    months = frame.month
    inner_train_mask = (months >= split.inner_train.start) & (months <= split.inner_train.end)
    inner_tune_mask = (months >= split.inner_tune.start) & (months <= split.inner_tune.end)
    refit_mask = (months >= split.refit_train.start) & (months <= split.refit_train.end)
    outer_mask = (months >= split.outer_valid.start) & (months <= split.outer_valid.end)
    if not inner_train_mask.any() or not inner_tune_mask.any() or not refit_mask.any() or not outer_mask.any():
        raise ValueError(f"{outer_name} has an empty temporal partition")
    for partition_name, partition_range, partition_mask in (
        ("inner_train", split.inner_train, inner_train_mask),
        ("inner_tune", split.inner_tune, inner_tune_mask),
        ("refit_train", split.refit_train, refit_mask),
        ("outer_valid", split.outer_valid, outer_mask),
    ):
        actual_months = set(int(value) for value in np.unique(months[partition_mask]))
        expected_months = set(range(partition_range.start, partition_range.end + 1))
        if actual_months != expected_months:
            raise ValueError(f"{outer_name} {partition_name} has incomplete month coverage")
    raw_features = tuple(frame.features.columns)
    started = time.perf_counter()
    inner_pre = CleanRealMLPPreprocessor(raw_features, cfg).fit(frame.features.loc[inner_train_mask], frame.target[inner_train_mask])
    x_inner, c_inner = inner_pre.transform(frame.features.loc[inner_train_mask])
    x_tune, c_tune = inner_pre.transform(frame.features.loc[inner_tune_mask])
    inner_result = train_inner(x_inner, c_inner, np.round(frame.target[inner_train_mask], cfg.target_round) if cfg.target_round is not None else frame.target[inner_train_mask], x_tune, c_tune, frame.target[inner_tune_mask], cfg)
    save_predictions(output / "inner_predictions.npz", sample_id=frame.sample_id[inner_tune_mask], month=months[inner_tune_mask], target=frame.target[inner_tune_mask], pred=inner_result.predictions, split=np.full(inner_result.predictions.size, f"{outer_name}:inner_tune"))

    refit_pre = CleanRealMLPPreprocessor(raw_features, cfg).fit(frame.features.loc[refit_mask], frame.target[refit_mask])
    x_refit, c_refit = refit_pre.transform(frame.features.loc[refit_mask])
    x_outer, c_outer = refit_pre.transform(frame.features.loc[outer_mask])
    outer_pred, refit_history, refit_step, refit_progress, refit_rq_hash = _train_refit_predict(x_refit, c_refit, np.round(frame.target[refit_mask], cfg.target_round) if cfg.target_round is not None else frame.target[refit_mask], x_outer, c_outer, inner_result.best_progress, cfg)
    diagnostics = prediction_diagnostics(outer_pred, frame.target[outer_mask])
    diagnostics["pearson"] = float(np.corrcoef(outer_pred, frame.target[outer_mask])[0, 1]) if np.std(outer_pred) and np.std(frame.target[outer_mask]) else 0.0
    diagnostics["nan_or_inf"] = int((~np.isfinite(outer_pred)).sum())
    save_predictions(output / "predictions.npz", sample_id=frame.sample_id[outer_mask], month=months[outer_mask], target=frame.target[outer_mask], pred=outer_pred, split=np.full(outer_pred.size, f"{outer_name}:outer_valid"))
    legacy_corr = None
    if outer_name == "PSEUDO" and legacy_pseudo_path is not None:
        legacy = np.load(legacy_pseudo_path)
        legacy_pred = np.asarray(legacy["pred"]).reshape(-1)
        legacy_y = np.asarray(legacy["y"]).reshape(-1)
        current_y = frame.target[outer_mask]
        if legacy_y.shape != current_y.shape or not np.array_equal(legacy_y, current_y):
            raise ValueError("legacy PSEUDO target is not exactly aligned with current month 33-70 target")
        legacy_corr = float(np.corrcoef(outer_pred, legacy_pred)[0, 1]) if np.std(outer_pred) and np.std(legacy_pred) else 0.0
        diagnostics["legacy_cosine"] = cosine_uncentered(legacy_pred, current_y)
    result = {
        "outer": outer_name,
        "best_epoch": inner_result.best_epoch,
        "best_step": inner_result.best_step,
        "best_progress": inner_result.best_progress,
        "refit_step": refit_step,
        "refit_progress": refit_progress,
        "model_parameters": inner_result.model_parameters,
        "inner_history": inner_result.history,
        "refit_history": refit_history,
        "diagnostics": diagnostics,
        "legacy_correlation": legacy_corr,
        "preprocessing": {"inner_state_hash": inner_pre.state_hash, "refit_state_hash": refit_pre.state_hash, "inner_rq_state_hash": inner_result.rq_state_hash, "refit_rq_state_hash": refit_rq_hash, "feature_hash": feature_hash(list(refit_pre.selected_numeric) + list(refit_pre.categorical)), "n_inner_train": int(inner_train_mask.sum()), "n_inner_tune": int(inner_tune_mask.sum()), "n_refit": int(refit_mask.sum()), "n_outer_valid": int(outer_mask.sum())},
        "runtime_seconds": time.perf_counter() - started,
    }
    config_payload = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    manifest = ExperimentManifest(experiment_id=f"{experiment_id}-{outer_name.lower()}", status="complete", git_sha=os.environ.get("MSCAP_GIT_SHA") or git_sha(), config_hash=hashlib.sha256(config_payload).hexdigest(), train_months=split.refit_train.as_tuple(), valid_months=split.outer_valid.as_tuple(), feature_hash=result["preprocessing"]["feature_hash"], best_step=inner_result.best_step, best_progress=inner_result.best_progress, runtime_seconds=result["runtime_seconds"], scores={"cosine_uncentered": float(diagnostics["cosine_uncentered"])}, diagnostics=result["diagnostics"] | result["preprocessing"], environment=_environment_versions())
    manifest.data_fingerprints = {Path(path).name: _stable_file_fingerprint(path) for path in data_paths if Path(path).exists()}
    manifest.write(output)
    (output / "training_history.json").write_text(json.dumps({"inner": inner_result.history, "refit": refit_history}, indent=2), encoding="utf-8")
    (output / "report.md").write_text(_outer_report_md(outer_name, result), encoding="utf-8")
    return result


def _train_refit_predict(x_train, c_train, y_train, x_valid, c_valid, progress_limit: float, cfg: RealMLPConfig) -> tuple[np.ndarray, list[dict[str, float]], int, float, str]:
    """Train a refit model and predict the outer validation rows."""
    torch, _, F, RealMLPRQ = _build_torch_classes()
    _set_seed(cfg.seed)
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else ("cpu" if cfg.device == "auto" else cfg.device))
    rq = RQKMeansEncoder(cfg.rq_encoder_layers, cfg.rq_vocab_size).fit(y_train)
    codes_np = rq.encode(y_train)
    cat_dims = [int(np.max(c_train[:, j]) + 2) if c_train.shape[1] else 1 for j in range(c_train.shape[1])]
    model = RealMLPRQ(x_train.shape[1], cat_dims, cfg).to(device)
    optimizer = _make_optimizer(model, cfg, torch)
    ema = _EMA(model, cfg.ema_decay, torch)
    x = torch.from_numpy(x_train).to(device); c = torch.from_numpy(c_train).to(device); y = torch.from_numpy(np.asarray(y_train, dtype=np.float32)).to(device); codes = torch.from_numpy(codes_np).to(device)
    total_batches = (len(y_train) + cfg.train_batch_size - 1) // cfg.train_batch_size
    total_steps = max(total_batches * cfg.epochs, 1)
    stop_step = max(1, min(total_steps, int(math.ceil(progress_limit * total_steps))))
    step = 0
    history: list[dict[str, float]] = []
    while step < stop_step:
        permutation = torch.randperm(len(y), device=device)
        sums = np.zeros(4, dtype=np.float64)
        batches = 0
        for start in range(0, len(y), cfg.train_batch_size):
            if step >= stop_step:
                break
            progress = min(step / total_steps, 1.0)
            for group, base in zip(optimizer.param_groups, (20.0, 0.093, 1.0, 1.0, 0.1)):
                group["lr"] = flat_anneal(cfg.learning_rate * base, progress)
            indices = permutation[start:start + cfg.train_batch_size]
            target = y[indices]
            noisy = target + torch.randn_like(target) * (cfg.label_noise_std * (1.0 - progress))
            optimizer.zero_grad(set_to_none=True)
            logits, pred = model(x[indices], c[indices], return_codes=True)
            loss, cos, mse, rq_loss = _loss(pred, noisy, logits, codes[indices], progress, cfg, torch, F)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip); optimizer.step(); ema.update(); step += 1
            sums += np.asarray([loss.item(), cos.item(), mse.item(), rq_loss.item()]); batches += 1
        history.append({"epoch": float(len(history) + 1), "loss": float(sums[0] / max(batches, 1)), "train_cosine": float(sums[1] / max(batches, 1)), "mse": float(sums[2] / max(batches, 1)), "rq": float(sums[3] / max(batches, 1))})
    original = ema.apply()
    prediction = _predict(model, x_valid, c_valid, cfg, str(device), torch)
    ema.restore(original)
    return prediction, history, stop_step, stop_step / total_steps, rq.state_hash


def summarize_outer(artifact_root: str | Path, experiment_id: str = "clean-realmlp-v2a", legacy_pseudo_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(artifact_root)
    rows = []
    for name in NESTED_SPLITS:
        path = root / experiment_id / name / "manifest.json"
        if not path.exists():
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        split = NESTED_SPLITS[name]
        if row.get("status") != "complete":
            raise ValueError(f"{name} manifest is not complete")
        if tuple(row.get("train_months") or ()) != split.refit_train.as_tuple():
            raise ValueError(f"{name} manifest train months do not match the registry")
        if tuple(row.get("valid_months") or ()) != split.outer_valid.as_tuple():
            raise ValueError(f"{name} manifest valid months do not match the registry")
        prediction_path = path.with_name("predictions.npz")
        if not prediction_path.exists():
            raise FileNotFoundError(f"missing predictions for {name}")
        with np.load(prediction_path) as prediction:
            required = {"sample_id", "month", "target", "pred", "split"}
            if not required.issubset(prediction.files):
                raise ValueError(f"{name} predictions are missing required arrays")
            row_count = len(prediction["pred"])
            if row_count != int(row["diagnostics"]["n_outer_valid"]):
                raise ValueError(f"{name} prediction row count does not match its manifest")
            prediction_months = set(int(value) for value in np.unique(prediction["month"]))
            expected_months = set(range(split.outer_valid.start, split.outer_valid.end + 1))
            if prediction_months != expected_months:
                raise ValueError(f"{name} predictions have incomplete month coverage")
            if len(np.unique(prediction["sample_id"])) != row_count:
                raise ValueError(f"{name} predictions contain duplicate sample_id values")
            if not np.isfinite(prediction["pred"]).all():
                raise ValueError(f"{name} predictions contain NaN or infinite values")
        rows.append(row)
    if len(rows) != 4:
        raise FileNotFoundError(f"expected four completed outer manifests under {root / experiment_id}")
    scores = [float(row["scores"]["cosine_uncentered"]) for row in rows]
    report = {"experiment_id": experiment_id, "metric": "cosine_uncentered", "outer": rows, "mean_score": float(np.mean(scores)), "note": "Outer folds are correlated temporal stress tests, not independent samples."}
    if legacy_pseudo_path is not None:
        current = np.load(root / experiment_id / "PSEUDO" / "predictions.npz")
        legacy = np.load(legacy_pseudo_path)
        current_target = np.asarray(current["target"]).reshape(-1)
        legacy_target = np.asarray(legacy["y"]).reshape(-1)
        legacy_pred = np.asarray(legacy["pred"]).reshape(-1)
        current_pred = np.asarray(current["pred"]).reshape(-1)
        if current_target.shape != legacy_target.shape or not np.array_equal(current_target, legacy_target):
            raise ValueError("legacy PSEUDO target is not exactly aligned with the downloaded clean PSEUDO prediction")
        report["legacy_pseudo_cosine"] = cosine_uncentered(legacy_pred, legacy_target)
        report["clean_vs_legacy_prediction_pearson"] = float(np.corrcoef(current_pred, legacy_pred)[0, 1]) if np.std(current_pred) and np.std(legacy_pred) else 0.0
    report_stem = f"{experiment_id.replace('-', '_')}_report"
    target = root / f"{report_stem}.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [f"# {experiment_id}", "", "| Outer | Cosine | Pearson | Pred mean | Pred std | Target std | NaN/Inf | Best step | Best progress | Runtime (s) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        d = row["diagnostics"]
        lines.append(f"| {row['experiment_id'].rsplit('-', 1)[-1].upper()} | {row['scores']['cosine_uncentered']:.9f} | {d.get('pearson', float('nan')):.9f} | {d['prediction']['mean']:.9g} | {d['prediction']['std']:.9g} | {d['target']['std']:.9g} | {d.get('nan_or_inf', d['prediction'].get('nan_or_inf', 0))} | {row.get('best_step', 0)} | {row.get('best_progress', float('nan')):.4f} | {row.get('runtime_seconds', float('nan')):.1f} |")
    lines += ["", f"Mean cosine: `{report['mean_score']:.9f}`"]
    if "legacy_pseudo_cosine" in report:
        lines += [f"Legacy PSEUDO cosine: `{report['legacy_pseudo_cosine']:.9f}`", f"Clean-vs-legacy PSEUDO prediction Pearson: `{report['clean_vs_legacy_prediction_pearson']:.9f}`"]
    lines += ["", report["note"], ""]
    target.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    return report
