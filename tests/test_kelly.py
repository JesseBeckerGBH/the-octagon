from orchestrator.kelly_staking import KellyStaking


def test_positive_edge_produces_a_stake():
    banker = KellyStaking(bankroll=10_000, kelly_fraction=0.25)
    stake, reason = banker.calculate_stake(model_prob=0.60, decimal_odds=2.0)
    assert reason == "Bet placed"
    assert 0 < stake <= banker.bankroll * banker.max_stake_pct


def test_negative_ev_is_rejected():
    banker = KellyStaking(bankroll=10_000)
    # Model thinks 40% but odds imply ~50% (2.0) -> negative EV
    stake, reason = banker.calculate_stake(model_prob=0.40, decimal_odds=2.0)
    assert stake == 0.0
    assert "Negative EV" in reason or "negative EV" in reason


def test_edge_below_minimum_is_rejected():
    banker = KellyStaking(bankroll=10_000, min_edge=0.10)
    # ~2% edge, below the 10% minimum
    stake, reason = banker.calculate_stake(model_prob=0.52, decimal_odds=2.0)
    assert stake == 0.0
    assert "Edge too small" in reason


def test_stake_never_exceeds_max_stake_pct():
    banker = KellyStaking(bankroll=10_000, kelly_fraction=1.0, max_stake_pct=0.05)
    # Huge apparent edge would blow past the cap without it
    stake, reason = banker.calculate_stake(model_prob=0.95, decimal_odds=3.0)
    assert stake <= banker.bankroll * 0.05 + 1e-9


def test_bankroll_updates_on_win_and_loss():
    banker = KellyStaking(bankroll=1000.0)
    new_bal = banker.update_bankroll(stake=100, won=True, odds=2.0)
    assert new_bal == 1100.0
    new_bal = banker.update_bankroll(stake=100, won=False, odds=2.0)
    assert new_bal == 1000.0
