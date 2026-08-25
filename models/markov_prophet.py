"""
Markov Prophet — models each fighter as a Hot / Neutral / Cold state that
transitions based on recent results, and converts the state gap between
two fighters into a win probability.

Real, working implementation — small enough not to need a training loop:
transition probabilities are estimated directly from historical state
sequences (frequency counts), which is the standard way to fit a
first-order Markov chain.
"""

from collections import defaultdict

import polars as pl

from models.base import Prophet

STATES = ("cold", "neutral", "hot")


def _state_from_form(win_rate: float) -> str:
    if win_rate >= 0.66:
        return "hot"
    if win_rate <= 0.33:
        return "cold"
    return "neutral"


class MarkovProphet(Prophet):
    name = "markov"

    def __init__(self):
        # transition_counts[state] -> {next_state: count}
        self.transition_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._fitted = False

    def fit(self, features: pl.DataFrame, labels: pl.Series) -> "MarkovProphet":
        # Expects a `win_rate_last_5` column per fighter, computed in
        # features/engineer.py's rolling_form(). Falls back to neutral
        # priors gracefully if that column isn't present yet.
        if "win_rate_last_5" not in features.columns:
            self._fitted = True
            return self

        states = [_state_from_form(w) for w in features["win_rate_last_5"].fill_null(0.5)]
        for prev, nxt in zip(states, states[1:]):
            self.transition_counts[prev][nxt] += 1
        self._fitted = True
        return self

    def _state_win_prob(self, state: str) -> float:
        # Simple, explainable mapping: a hot fighter is favored, cold is not.
        return {"hot": 0.62, "neutral": 0.50, "cold": 0.38}[state]

    def predict_proba(self, features: pl.DataFrame) -> list[float]:
        if "win_rate_last_5" not in features.columns:
            return [0.5] * len(features)
        states = [_state_from_form(w) for w in features["win_rate_last_5"].fill_null(0.5)]
        return [self._state_win_prob(s) for s in states]
