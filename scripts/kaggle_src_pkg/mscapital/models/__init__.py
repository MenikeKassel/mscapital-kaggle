"""Model adapters.  Imports are lazy so protocol tests do not require CatBoost."""

from .residual_catboost import CatBoostResidualRegressor

__all__ = ["CatBoostResidualRegressor"]
