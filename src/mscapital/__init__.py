"""MSCapital protocol-v2 experiment package.

The package deliberately contains only leakage-safe, reusable primitives.  The
numbered scripts in ``scripts/`` remain historical evidence and are not
imported by this package.
"""

from .metrics import cosine_uncentered
from .splits import NESTED_SPLITS, OUTER_SPLITS, ROLLING_WINDOWS

__all__ = ["cosine_uncentered", "NESTED_SPLITS", "OUTER_SPLITS", "ROLLING_WINDOWS"]
__version__ = "0.1.0"
