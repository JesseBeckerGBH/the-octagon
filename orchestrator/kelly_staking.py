"""
Kelly staking — ported from the tennis "Council of Prophets" codebase
(council_of_prophets.zip), which had this fully built and Octagon didn't.
The math is sport-agnostic (probability + decimal odds in, stake out), so
it moves over unchanged aside from the docstrings.

Implements a SAFE fractional Kelly Criterion: full Kelly is mathematically
optimal for long-run growth but assumes your probability estimate is
exactly right, which it never is — a model that's even slightly overconfident
compounds into ruin under full Kelly. Fractional Kelly + a hard cap trades
some growth for survival.
"""


class KellyStaking:
    def __init__(self, bankroll: float = 1000.0, kelly_fraction: float = 0.25,
                 max_stake_pct: float = 0.05, min_edge: float = 0.02):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction  # e.g. 0.25 = quarter Kelly
        self.max_stake_pct = max_stake_pct    # never bet more than this fraction of bankroll
        self.min_edge = min_edge              # skip bets with edge below this
        self.history: list[dict] = []

    def calculate_stake(self, model_prob: float, decimal_odds: float) -> tuple[float, str]:
        """Kelly formula: f* = (b*p - q) / b, where b = decimal_odds - 1,
        p = model_prob, q = 1 - p. Returns (stake_amount, reason).
        """
        b = decimal_odds - 1
        p = model_prob
        q = 1 - p

        if b <= 0:
            return 0.0, "Invalid odds (must be > 1.0)"

        f_star = (b * p - q) / b
        ev = (p * decimal_odds) - 1  # expected value of a 1-unit bet

        if ev <= 0:
            return 0.0, "No value (negative EV)"
        if ev < self.min_edge:
            return 0.0, f"Edge too small ({ev:.2%})"

        safe_f = f_star * self.kelly_fraction
        final_f = min(safe_f, self.max_stake_pct)
        stake_amount = self.bankroll * final_f

        return stake_amount, "Bet placed"

    def update_bankroll(self, stake: float, won: bool, odds: float) -> float:
        if won:
            self.bankroll += stake * (odds - 1)
            result = "WIN"
        else:
            self.bankroll -= stake
            result = "LOSS"

        self.history.append({"stake": stake, "result": result, "new_bankroll": self.bankroll})
        return self.bankroll
