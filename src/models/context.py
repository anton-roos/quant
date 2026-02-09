"""
PipelineContext — replaces module-level global mutable state.

Holds the scaler, feature column list, and classification thresholds
so that they can be passed explicitly to data-processing functions
instead of being mutated as module globals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sklearn.preprocessing import StandardScaler


@dataclass
class PipelineContext:
    """Immutable-ish bag of fitted artefacts shared across pipeline stages.

    Attributes
    ----------
    scaler : StandardScaler
        Fitted on training data only.
    feature_cols : list[str]
        Ordered list of feature column names used by the model.
    thresholds_up : dict[str, float]
        Per-horizon upper thresholds (mu + 2·sigma from training returns).
    thresholds_down : dict[str, float]
        Per-horizon lower thresholds (mu − 2·sigma from training returns).
    """

    scaler: StandardScaler = field(default_factory=StandardScaler)
    feature_cols: List[str] = field(default_factory=list)
    thresholds_up: Dict[str, float] = field(default_factory=dict)
    thresholds_down: Dict[str, float] = field(default_factory=dict)
