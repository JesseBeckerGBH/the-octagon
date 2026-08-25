"""
The Council of Prophets — blends active prophets into one consensus
probability, with configurable weights and dissent detection.

Two blend modes:
  - Weighted average (default, works from prediction #1)
  - Stacked meta-learner (LightGBM), once enough validated predictions
    exist (see config.validation.min_predictions_before_stacking) — trained
    on out-of-fold prophet outputs plus the original features, per the
    original design notes on ensemble stacking.
"""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl
import yaml

from models.base import Prophet


@dataclass
class CouncilResult:
    blended_prob_a: float
    prophet_probs: dict[str, float]
    dissent: float  # max pairwise disagreement among active prophets


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class Council:
    def __init__(self, prophets: list[Prophet], weights: dict[str, float] | None = None):
        self.prophets = {p.name: p for p in prophets}
        self.weights = weights or {name: 1.0 / len(prophets) for name in self.prophets}
        self.meta_model: lgb.Booster | None = None

    @classmethod
    def from_config(cls, prophets: list[Prophet], config_path: str = "config/settings.yaml") -> "Council":
        cfg = load_config(config_path)
        status = cfg["council"]["status"]
        active = [p for p in prophets if status.get(p.name) in ("active", "experimental")]
        weights = {name: w for name, w in cfg["council"]["weights"].items()
                   if status.get(name) in ("active", "experimental")}
        return cls(active, weights)

    def consensus(self, features: pl.DataFrame) -> list[CouncilResult]:
        prophet_probs = {name: p.predict_proba(features) for name, p in self.prophets.items()}

        results = []
        n = len(features)
        for i in range(n):
            row_probs = {name: probs[i] for name, probs in prophet_probs.items()}

            if self.meta_model is not None:
                blended = self._stacked_predict(row_probs)
            else:
                total_weight = sum(self.weights.get(name, 0) for name in row_probs) or 1.0
                blended = sum(row_probs[name] * self.weights.get(name, 0) for name in row_probs) / total_weight

            values = list(row_probs.values())
            dissent = max(values) - min(values) if len(values) > 1 else 0.0

            results.append(CouncilResult(blended_prob_a=blended, prophet_probs=row_probs, dissent=dissent))
        return results

    def _stacked_predict(self, row_probs: dict[str, float]) -> float:
        ordered = np.array([[row_probs[name] for name in sorted(row_probs)]])
        return float(self.meta_model.predict(ordered)[0])

    def fit_stacker(self, oof_prophet_probs: pl.DataFrame, labels: pl.Series) -> None:
        """Train the LightGBM meta-learner on out-of-fold prophet outputs.

        Call this once enough validated predictions exist
        (config.validation.min_predictions_before_stacking). Until then the
        council uses the simple weighted average, which is honest and
        avoids overfitting a stacker on too little validated history.
        """
        X = oof_prophet_probs.select(sorted(oof_prophet_probs.columns)).to_numpy()
        y = labels.to_numpy()
        train_set = lgb.Dataset(X, label=y)
        self.meta_model = lgb.train({"objective": "binary", "verbose": -1}, train_set, num_boost_round=100)
