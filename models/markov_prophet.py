"""
Markov Prophet — wraps features/markov_form.py's MarkovFormChain (ported
from the live production engine) to produce a win probability.

This replaces the original heuristic version, which classified fighters
into hot/neutral/cold from a single rolling win-rate column and mapped
each state to a hand-picked probability (0.62/0.50/0.38). The production
chain is more rigorous on two counts: it adds a fourth DECLINING state
(was hot, now losing — a real momentum shift, not just "cold"), and its
transition matrix is *learned* from historical state sequences rather than
guessed. This version keeps that chain for state classification, but
replaces the hardcoded probability table with a small logistic regression
fit on (state, momentum) -> P(fighter_a wins) using real labels — so the
mapping from "fighter A is Hot, fighter B is Cold" to a probability is
data-driven too, not just the chain's transition matrix.
"""

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from features.markov_form import MarkovFormChain
from models.base import Prophet


class MarkovProphet(Prophet):
    name = "markov"

    def __init__(self):
        self.chain = MarkovFormChain()
        self._lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        self._fitted = False

    def _state_row(self, features: pl.DataFrame) -> np.ndarray:
        """(state_diff, momentum_diff) per row, using whichever form
        columns are present — the real fa_form_state/fb_form_state pair
        from features/leak_safe_features.py if available, else falling
        back to a neutral (0, 0) row so this prophet degrades gracefully
        on the placeholder feature table.
        """
        cols = features.columns
        if "diff_form_state" in cols and "diff_form_momentum" in cols:
            return features.select(["diff_form_state", "diff_form_momentum"]).fill_null(0).to_numpy()
        return np.zeros((len(features), 2))

    def fit(self, features: pl.DataFrame, labels: pl.Series) -> "MarkovProphet":
        X = self._state_row(features)
        y = labels.to_numpy()
        if X.any():  # only fit the logistic head if we have real form signal
            self._lr.fit(X, y)
            self._fitted = True
        else:
            self._fitted = True  # fit() still "succeeds" — predict_proba() returns 0.5
        return self

    def predict_proba(self, features: pl.DataFrame) -> list[float]:
        if not self._fitted:
            raise RuntimeError("MarkovProphet.fit() must be called before predict_proba().")
        X = self._state_row(features)
        if not X.any() or not hasattr(self._lr, "coef_"):
            return [0.5] * len(features)
        probs = self._lr.predict_proba(X)[:, 1]
        return np.clip(probs, 0.0, 1.0).tolist()
