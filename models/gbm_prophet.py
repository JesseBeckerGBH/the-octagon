"""
GBM Prophet — gradient-boosted trees on tabular differential features.

Hyperparameters below are ported verbatim from the live production engine
(ufc-predictions-saas/engine/model/train_ensemble.py's XGBoost base
learner) — these are the settings actually validated across the
production walk-forward backtest, not a fresh guess. This is one of four
base learners the Council now runs (see lightgbm_prophet.py,
logistic_prophet.py, randomforest_prophet.py for the other three),
matching production's XGBoost + LightGBM + Logistic + RandomForest
ensemble.
"""

import numpy as np
import polars as pl
import xgboost as xgb

from features.engineer import get_feature_cols
from models.base import Prophet


class GBMProphet(Prophet):
    name = "gbm"

    def __init__(self, feature_cols: list[str] | None = None, **xgb_params):
        params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 3,
            "eval_metric": "logloss",
            "objective": "binary:logistic",
            "verbosity": 0,
            "random_state": 42,
        }
        params.update(xgb_params)
        self.model = xgb.XGBClassifier(**params)
        self.feature_cols = feature_cols
        self._fitted = False

    def fit(self, features: pl.DataFrame, labels: pl.Series) -> "GBMProphet":
        cols = self.feature_cols or get_feature_cols(features)
        self.feature_cols = cols
        X = features.select(cols).fill_null(0).to_numpy()
        y = labels.to_numpy()
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, features: pl.DataFrame) -> list[float]:
        if not self._fitted:
            raise RuntimeError("GBMProphet.fit() must be called before predict_proba().")
        X = features.select(self.feature_cols).fill_null(0).to_numpy()
        probs = self.model.predict_proba(X)[:, 1]
        return np.clip(probs, 0.0, 1.0).tolist()
