import pandas as pd

from features.markov_form import FormState, MarkovFormChain, add_markov_features


def test_classify_state_hot_cold_neutral():
    chain = MarkovFormChain()
    assert chain._classify_state([1, 1, 1]) == FormState.HOT
    assert chain._classify_state([0, 0, 0]) == FormState.COLD
    assert chain._classify_state([1, 0, 1, 0]) == FormState.NEUTRAL
    assert chain._classify_state([]) == FormState.NEUTRAL


def test_classify_state_declining_on_momentum_shift():
    chain = MarkovFormChain()
    # Was winning (first 3), now losing (last 3) -> DECLINING, not just COLD
    assert chain._classify_state([1, 1, 1, 0, 0, 0]) == FormState.DECLINING


def test_fit_produces_row_stochastic_transition_matrix():
    records = pd.DataFrame({
        "fighter": ["A"] * 6 + ["B"] * 6,
        "date": pd.date_range("2020-01-01", periods=6).tolist() * 2,
        "won": [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1],
    })
    chain = MarkovFormChain().fit(records)
    for row in chain.transition_matrix:
        assert abs(row.sum() - 1.0) < 1e-9


def test_get_features_shape():
    chain = MarkovFormChain()
    feats = chain.get_features("Unknown Fighter")
    assert set(feats.keys()) == {
        "form_state", "form_state_hot", "form_state_cold",
        "form_state_declining", "form_momentum", "form_next_hot_prob",
    }


def test_add_markov_features_is_leak_safe_and_adds_diff_columns():
    df = pd.DataFrame({
        "header_fighter_a_name": ["A", "B", "A"],
        "header_fighter_b_name": ["B", "A", "C"],
        "header_fighter_a_outcome": ["W", "L", "W"],
        "header_fighter_b_outcome": ["L", "W", "L"],
        "event_date_parsed": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
    })
    out = add_markov_features(df)
    assert "diff_form_state" in out.columns
    assert "diff_form_momentum" in out.columns
    # First appearance of any fighter has no history yet -> neutral state.
    assert out.loc[0, "fa_form_state"] == int(FormState.NEUTRAL)
