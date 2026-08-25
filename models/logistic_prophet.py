"""
Logistic Prophet — scaled logistic regression, the calibrated-baseline
base learner in the production ensemble recipe. Ported verbatim from
ufc-predictions-saas/engine/model/train_ensemble.py — its whole job in the
ensemble is to be the simple, well-calibrated anchor the tree-based
learners (which can be locally overconfident) get averaged against.
"""

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features.engineer import get_feature_cols
from models.base import Prophet


class LogisticProphet(Prophet):
    name = "logistic"

    def __init__(self, feature_cols: list[str] | None = None):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)),
        ])
        self.feature_cols = feature_cols
        self._fitted = False

    def fit(self, features: pl.DataFrame, labels: pl.Series) -> "LogisticProphet":
        cols = self.feature_cols or get_feature_cols(features)
        self.feature_cols = cols
        X = features.select(cols).fill_null(0).to_numpy()
        y = labels.to_numpy()
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, features: pl.DataFrame) -> list[float]:
        if not self._fitted:
            raise RuntimeError("LogisticProphet.fit() must be called before predict_proba().")
        X = features.select(self.feature_cols).fill_null(0).to_numpy()
        probs = self.model.predict_proba(X)[:, 1]
        return np.clip(probs, 0.0, 1.0).tolist()
