"""
Random Forest Prophet — bagged decision trees, fourth base learner in the
production ensemble recipe. Hyperparameters ported verbatim from
ufc-predictions-saas/engine/model/train_ensemble.py. Deliberately shallow
(max_depth=8) and leaf-regularized (min_samples_leaf=5) — the production
recipe uses RF for its variance-reduction properties alongside the
gradient-boosted learners, not as the sharpest individual model.
"""

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier

from features.engineer import get_feature_cols
from models.base import Prophet


class RandomForestProphet(Prophet):
    name = "randomforest"

    def __init__(self, feature_cols: list[str] | None = None, **rf_params):
        params = {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        }
        params.update(rf_params)
        self.model = RandomForestClassifier(**params)
        self.feature_cols = feature_cols
        self._fitted = False

    def fit(self, features: pl.DataFrame, labels: pl.Series) -> "RandomForestProphet":
        cols = self.feature_cols or get_feature_cols(features)
        self.feature_cols = cols
        X = features.select(cols).fill_null(0).to_numpy()
        y = labels.to_numpy()
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, features: pl.DataFrame) -> list[float]:
        if not self._fitted:
            raise RuntimeError("RandomForestProphet.fit() must be called before predict_proba().")
        X = features.select(self.feature_cols).fill_null(0).to_numpy()
        probs = self.model.predict_proba(X)[:, 1]
        return np.clip(probs, 0.0, 1.0).tolist()
