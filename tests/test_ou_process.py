import numpy as np

from features.ou_process import OUProcess, add_ou_features_placeholder


def _mean_reverting_series(theta=0.3, mu=0.6, sigma=0.02, n=50, seed=0):
    rng = np.random.default_rng(seed)
    x = [mu]
    for _ in range(1, n):
        x.append(x[-1] + theta * (mu - x[-1]) + rng.normal(0, sigma))
    return [(float(i), float(v)) for i, v in enumerate(x)]


def test_fit_recovers_a_positive_reversion_speed_on_mean_reverting_data():
    ou = OUProcess().fit(_mean_reverting_series())
    assert ou.a > 0
    assert 0.0 <= ou.b <= 1.0


def test_predict_closing_moves_toward_equilibrium():
    ou = OUProcess().fit(_mean_reverting_series(mu=0.7))
    closing = ou.predict_closing(current_value=0.4, hours_to_close=100.0)
    # With a lot of time left, the prediction should have moved toward b.
    assert closing > 0.4


def test_estimate_clv_positive_when_our_prob_beats_predicted_close():
    ou = OUProcess().fit(_mean_reverting_series(mu=0.5))
    clv = ou.estimate_clv(our_prob=0.65, market_prob=0.5, hours_to_close=1.0)
    assert clv > 0


def test_detect_steam_false_before_fit():
    ou = OUProcess()
    assert ou.detect_steam(recent_move=0.5) is False


def test_get_features_has_expected_keys():
    ou = OUProcess().fit(_mean_reverting_series())
    feats = ou.get_features()
    assert feats["ou_has_data"] == 1.0
    assert set(feats.keys()) == {
        "ou_reversion_speed", "ou_equilibrium", "ou_volatility",
        "ou_half_life_hours", "ou_has_data",
    }


def test_get_features_defaults_before_fit():
    feats = OUProcess().get_features()
    assert feats["ou_has_data"] == 0.0


def test_add_ou_features_placeholder_fills_missing_columns():
    import pandas as pd

    df = pd.DataFrame({"x": [1, 2]})
    out = add_ou_features_placeholder(df)
    assert "ou_clv_estimate" in out.columns
    assert "ou_steam_flag" in out.columns
    assert (out["ou_has_data"] == 0.0).all()
