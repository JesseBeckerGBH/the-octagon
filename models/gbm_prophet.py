"""
GBM Prophet — gradient-boosted trees on tabular differential features.

This is the first prophet the project recommended standing up (fastest to
train and validate), and it's the one implementation in this council that's
fully real end-to-end: fit() trains an actual XGBoost classifier,
predict_proba() returns real calibrated-ish probabilities.
"""

import numpy as np
import polars as pl
import xgboost as xgb

from models.base import Prophet

FEATURE_COLUMNS = ["reach_diff", "age_diff", "slpm_diff"]


class GBMProphet(Prophet):
    name = "gbm"

    def __init__(self, **xgb_params):
        params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "eval_metric": "logloss",
            "objective": "binary:logistic",
        }
        params.update(xgb_params)
        self.model = xgb.XGBClassifier(**params)
        self._fitted = False

    def fit(self, features: pl.DataFrame, labels: pl.Series) -> "GBMProphet":
        X = features.select(FEATURE_COLUMNS).to_numpy()
        y = labels.to_numpy()
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, features: pl.DataFrame) -> list[float]:
        if not self._fitted:
            raise RuntimeError("GBMProphet.fit() must be called before predict_proba().")
        X = features.select(FEATURE_COLUMNS).to_numpy()
        probs = self.model.predict_proba(X)[:, 1]
        return np.clip(probs, 0.0, 1.0).tolist()
