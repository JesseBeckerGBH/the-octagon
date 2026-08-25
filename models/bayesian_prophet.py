"""
Bayesian Prophet — logistic regression over differential features with
weakly-informative priors, fit via PyMC. Marked "experimental" in
config/settings.yaml (excluded from the live council blend by default)
because it hasn't been validated against real historical outcomes yet —
but the fit/predict path is real, not a stub.

Priors are refit each time fit() is called; a future improvement (tracked
in the README roadmap) is updating posteriors incrementally after each
event rather than refitting from scratch.
"""

import numpy as np
import polars as pl

from features.engineer import get_feature_cols
from models.base import Prophet


class BayesianProphet(Prophet):
    name = "bayesian"

    def __init__(self, draws: int = 200, tune: int = 200, chains: int = 1,
                 feature_cols: list[str] | None = None):
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.feature_cols = feature_cols
        self._trace = None
        self._mean_coefs = None
        self._mean_intercept = None

    def fit(self, features: pl.DataFrame, labels: pl.Series) -> "BayesianProphet":
        import pymc as pm

        cols = self.feature_cols or get_feature_cols(features)
        self.feature_cols = cols
        X = features.select(cols).fill_null(0).to_numpy()
        y = labels.to_numpy()

        with pm.Model():
            intercept = pm.Normal("intercept", mu=0, sigma=1)
            coefs = pm.Normal("coefs", mu=0, sigma=1, shape=X.shape[1])
            logits = intercept + pm.math.dot(X, coefs)
            pm.Bernoulli("obs", logit_p=logits, observed=y)

            trace = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                progressbar=False,
                random_seed=42,
            )

        self._trace = trace
        self._mean_coefs = trace.posterior["coefs"].mean(dim=("chain", "draw")).values
        self._mean_intercept = float(trace.posterior["intercept"].mean())
        return self

    def predict_proba(self, features: pl.DataFrame) -> list[float]:
        if self._mean_coefs is None:
            raise RuntimeError("BayesianProphet.fit() must be called before predict_proba().")
        X = features.select(self.feature_cols).fill_null(0).to_numpy()
        logits = self._mean_intercept + X @ self._mean_coefs
        probs = 1 / (1 + np.exp(-logits))
        return probs.tolist()
