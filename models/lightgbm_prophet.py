"""
LightGBM Prophet — gradient-boosted trees, second of the four base
learners in the production ensemble recipe (see gbm_prophet.py for
XGBoost, logistic_prophet.py and randomforest_prophet.py for the other
two). Hyperparameters ported verbatim from
ufc-predictions-saas/engine/model/train_ensemble.py.
"""

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier

from features.engineer import get_feature_cols
from models.base import Prophet


class LightGBMProphet(Prophet):
    name = "lightgbm"

    def __init__(self, feature_cols: list[str] | None = None, **lgb_params):
        params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 3,
            "verbose": -1,
            "random_state": 42,
        }
        params.update(lgb_params)
        self.model = LGBMClassifier(**params)
        self.feature_cols = feature_cols
        self._fitted = False

    def fit(self, features: pl.DataFrame, labels: pl.Series) -> "LightGBMProphet":
        cols = self.feature_cols or get_feature_cols(features)
        self.feature_cols = cols
        X = features.select(cols).fill_null(0).to_numpy()
        y = labels.to_numpy()
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, features: pl.DataFrame) -> list[float]:
        if not self._fitted:
            raise RuntimeError("LightGBMProphet.fit() must be called before predict_proba().")
        X = features.select(self.feature_cols).fill_null(0).to_numpy()
        probs = self.model.predict_proba(X)[:, 1]
        return np.clip(probs, 0.0, 1.0).tolist()
