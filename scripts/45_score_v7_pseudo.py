"""Validate RealMLP PSEUDO output and score its blend with the v5 table model."""

import argparse
import json
from pathlib import Path

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        pred = np.asarray(data["pred"], dtype=np.float64)
        y = np.asarray(data["y"], dtype=np.float64)
    if pred.shape != y.shape or pred.ndim != 1:
        raise ValueError(f"Invalid shapes in {path}: pred={pred.shape}, y={y.shape}")
    if not np.isfinite(pred).all() or not np.isfinite(y).all():
        raise ValueError(f"Non-finite values in {path}")
    return pred, y


def load_test_prediction(path: Path) -> np.ndarray:
    with np.load(path) as data:
        pred = np.asarray(data["pred"], dtype=np.float64)
    if pred.ndim != 1 or not np.isfinite(pred).all():
        raise ValueError(f"Invalid test prediction in {path}: shape={pred.shape}")
    return pred


def distribution(a: np.ndarray) -> dict[str, float]:
    q01, q05, q50, q95, q99 = np.quantile(a, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "p01": float(q01),
        "p05": float(q05),
        "p50": float(q50),
        "p95": float(q95),
        "p99": float(q99),
        "max": float(a.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--realmlp", type=Path, required=True)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--test-realmlp", type=Path)
    parser.add_argument("--test-table", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    realmlp, y = load_prediction(args.realmlp)
    result: dict[str, object] = {
        "n": int(y.size),
        "realmlp_cos": cosine(realmlp, y),
        "realmlp_mean": float(realmlp.mean()),
        "realmlp_std": float(realmlp.std()),
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
    }

    if args.table:
        table, table_y = load_prediction(args.table)
        if not np.array_equal(y, table_y):
            max_abs = float(np.max(np.abs(y - table_y)))
            raise ValueError(f"Target arrays are not aligned; max_abs_diff={max_abs}")
        result.update(
            {
                "table_cos": cosine(table, y),
                "table_mean": float(table.mean()),
                "table_std": float(table.std()),
                "table_realmlp_pearson": float(np.corrcoef(table, realmlp)[0, 1]),
                "table_realmlp_cos": cosine(table, realmlp),
            }
        )
        blends = {}
        best_weight = None
        best_score = -np.inf
        for realmlp_weight in np.arange(0.0, 0.301, 0.01):
            pred = (1.0 - realmlp_weight) * table + realmlp_weight * realmlp
            score = cosine(pred, y)
            blends[f"realmlp_{realmlp_weight:.2f}"] = score
            if score > best_score:
                best_weight = float(realmlp_weight)
                best_score = score
        result["blends"] = blends
        result["best_local_raw_weight"] = best_weight
        result["best_local_raw_score"] = best_score
        valid_v7 = 0.8 * table + 0.2 * realmlp
        result["valid_v7_w0.20_distribution"] = distribution(valid_v7)

        if args.test_realmlp and args.test_table:
            test_realmlp = load_test_prediction(args.test_realmlp)
            test_table = load_test_prediction(args.test_table)
            if test_realmlp.shape != test_table.shape:
                raise ValueError(
                    f"Test shapes do not align: {test_realmlp.shape} vs {test_table.shape}"
                )
            test_v7 = 0.8 * test_table + 0.2 * test_realmlp
            result.update(
                {
                    "test_table_realmlp_pearson": float(
                        np.corrcoef(test_table, test_realmlp)[0, 1]
                    ),
                    "test_table_distribution": distribution(test_table),
                    "test_realmlp_distribution": distribution(test_realmlp),
                    "test_v7_w0.20_distribution": distribution(test_v7),
                    "v7_test_valid_std_ratio": float(
                        test_v7.std() / (valid_v7.std() + 1e-12)
                    ),
                }
            )

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
