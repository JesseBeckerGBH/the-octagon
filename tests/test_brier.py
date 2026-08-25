from validator.post_event_validator import brier_score, clv_realized


def test_brier_perfect_prediction_is_zero():
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0


def test_brier_worst_prediction_is_one():
    assert brier_score([0.0, 1.0], [1, 0]) == 1.0


def test_brier_uncertain_prediction():
    # predicting exactly 0.5 on every fight gives 0.25 regardless of outcome
    assert abs(brier_score([0.5, 0.5, 0.5], [1, 0, 1]) - 0.25) < 1e-9


def test_brier_empty_is_zero():
    assert brier_score([], []) == 0.0


def test_clv_realized_positive_when_we_lead_the_market():
    # market moved from 0.50 to 0.60; we were already at 0.60 pre-move
    value = clv_realized(opening_prob=0.50, closing_prob=0.60, our_prob=0.60)
    assert value > 0


def test_clv_realized_zero_when_market_doesnt_move():
    assert clv_realized(opening_prob=0.50, closing_prob=0.50, our_prob=0.55) == 0.0
