import polars as pl

from models.base import Prophet
from orchestrator.council import Council


class _FixedProphet(Prophet):
    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value

    def fit(self, features, labels):
        return self

    def predict_proba(self, features):
        return [self.value] * len(features)


def test_council_weighted_average():
    council = Council(
        [_FixedProphet("a", 0.8), _FixedProphet("b", 0.4)],
        weights={"a": 0.5, "b": 0.5},
    )
    features = pl.DataFrame({"x": [1, 2]})
    results = council.consensus(features)
    assert len(results) == 2
    assert abs(results[0].blended_prob_a - 0.6) < 1e-9


def test_council_dissent_detection():
    council = Council(
        [_FixedProphet("a", 0.9), _FixedProphet("b", 0.1)],
        weights={"a": 0.5, "b": 0.5},
    )
    features = pl.DataFrame({"x": [1]})
    results = council.consensus(features)
    assert abs(results[0].dissent - 0.8) < 1e-9
