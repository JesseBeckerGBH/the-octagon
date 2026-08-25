import numpy as np
import polars as pl

from models.gbm_prophet import GBMProphet
from models.logistic_prophet import LogisticProphet
from orchestrator.council import Council


def _synthetic_features(n=60, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # Label correlated with x1 so the base learners have real signal to
    # find (and isotonic calibration has something non-degenerate to do).
    prob = 1 / (1 + np.exp(-1.5 * x1))
    y = (rng.uniform(size=n) < prob).astype(int)
    return pl.DataFrame({"x1": x1, "x2": x2}), pl.Series(y)


def test_fit_calibrated_stacker_returns_bounded_oof_metrics():
    features, labels = _synthetic_features()
    council = Council([
        GBMProphet(n_estimators=10, max_depth=2),
        LogisticProphet(),
    ])

    metrics = council.fit_calibrated_stacker(features, labels, n_splits=3)

    assert 0.0 <= metrics["oof_brier"] <= 1.0
    assert 0.0 <= metrics["oof_accuracy"] <= 1.0
    assert metrics["n_fights"] == 60
    assert council.calibrators is not None
    assert council.meta is not None


def test_consensus_uses_calibrated_stacking_when_available():
    features, labels = _synthetic_features()
    council = Council([
        GBMProphet(n_estimators=10, max_depth=2),
        LogisticProphet(),
    ])
    council.fit_calibrated_stacker(features, labels, n_splits=3)

    results = council.consensus(features)
    assert len(results) == 60
    for r in results:
        assert 0.0 <= r.blended_prob_a <= 1.0
        assert not r.gated  # no gate configured in this test


def test_calibrated_stacking_beats_random_on_separable_signal():
    """Sanity check that the whole pipeline (fold OOF -> isotonic ->
    logistic stack -> refit-on-all-data) actually learns something, not
    just that it runs without crashing.
    """
    features, labels = _synthetic_features(n=200, seed=7)
    council = Council([GBMProphet(n_estimators=20, max_depth=3), LogisticProphet()])
    metrics = council.fit_calibrated_stacker(features, labels, n_splits=5)
    assert metrics["oof_accuracy"] > 0.55  # better than a coin flip
