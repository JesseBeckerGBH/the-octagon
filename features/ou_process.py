"""
Ornstein-Uhlenbeck process — odds line-movement modeling.

Ported from the live production engine (ufc-predictions-saas/engine/models/
ou_process.py), which supersedes Octagon's original
models/sharp_money_prophet.py::estimate_ou_params(). That function is kept
for backward compatibility (it's real, unit-tested, sport-agnostic OU
parameter estimation) but this module is the fuller version actually used
in production: it wraps parameter estimation with CLV estimation and
steam-move (sharp money) detection, both of which Octagon's Sharp Money
prophet previously only sketched.

Models UFC odds movement as a mean-reverting stochastic process:

    dX(t) = a*(b - X(t))*dt + sigma*dW(t)

    X(t)  = implied probability at time t
    a     = mean-reversion speed (how fast odds snap back)
    b     = long-term equilibrium probability
    sigma = volatility of line movement
    W(t)  = Brownian motion

Applications: predicting where a line will close (for CLV), detecting
steam moves (abnormally fast reversion = sharp money), and estimating how
long an edge persists before the market corrects it.
"""

from typing import Dict, List, Tuple

import numpy as np


class OUProcess:
    """OU parameter estimation and feature generation for betting line
    dynamics. `fit()` on a single fight's line history; `estimate_clv()`
    and `detect_steam()` are the two things the validator and Sharp Money
    prophet actually consume.
    """

    def __init__(self):
        self.a = 0.0       # mean-reversion speed
        self.b = 0.5       # long-term mean
        self.sigma = 0.0   # volatility
        self._fitted = False
        self._line_history: List[Tuple[float, float]] = []

    def fit(self, line_history: List[Tuple[float, float]]):
        """`line_history`: chronological (timestamp_hours_before_fight,
        implied_probability) tuples.
        """
        self._line_history = line_history

        if len(line_history) < 3:
            self.a = 0.5
            self.b = line_history[-1][1] if line_history else 0.5
            self.sigma = 0.01
            self._fitted = True
            return self

        times = np.array([t for t, _ in line_history])
        values = np.array([v for _, v in line_history])

        dt_arr = np.diff(times)
        dt_arr = np.where(dt_arr == 0, 1e-6, dt_arr)

        x = values[:-1]
        y = values[1:]
        mean_dt = np.mean(dt_arr)

        if np.var(x) > 1e-10:
            x_mean, y_mean = np.mean(x), np.mean(y)
            cov_xy = np.mean((x - x_mean) * (y - y_mean))
            var_x = np.var(x)

            phi = cov_xy / var_x
            c = y_mean - phi * x_mean

            phi = np.clip(phi, 0.001, 0.999)
            self.a = -np.log(phi) / mean_dt if phi > 0 else 0.5
            self.b = c / (1 - phi) if abs(1 - phi) > 1e-6 else np.mean(values)

            residuals = y - (c + phi * x)
            self.sigma = np.std(residuals) / np.sqrt(max(abs(mean_dt), 1e-6))
        else:
            self.a = 0.1
            self.b = np.mean(values)
            self.sigma = np.std(values)

        self.a = np.clip(self.a, 0.01, 50.0)
        self.b = np.clip(self.b, 0.01, 0.99)
        self.sigma = np.clip(self.sigma, 0.001, 1.0)
        self._fitted = True
        return self

    def predict_closing(self, current_value: float, hours_to_close: float) -> float:
        """E[X(T)] = X(t)*exp(-a*(T-t)) + b*(1 - exp(-a*(T-t)))"""
        if not self._fitted:
            return current_value
        decay = np.exp(-self.a * hours_to_close)
        return current_value * decay + self.b * (1 - decay)

    def estimate_clv(self, our_prob: float, market_prob: float, hours_to_close: float = 2.0) -> float:
        """Estimated closing-line value: our probability vs. where the OU
        model expects the line to close. Positive = we bet before the line
        moved in our direction.
        """
        predicted_close = self.predict_closing(market_prob, hours_to_close)
        return our_prob - predicted_close

    def detect_steam(self, recent_move: float, lookback_hours: float = 1.0) -> bool:
        """A move is "steam" (sharp money) if it exceeds 2 standard
        deviations of the OU-expected movement in the lookback window.
        """
        if not self._fitted:
            return False
        expected_std = self.sigma * np.sqrt(
            (1 - np.exp(-2 * self.a * lookback_hours)) / (2 * self.a + 1e-10)
        )
        return abs(recent_move) > 2.0 * expected_std

    def get_features(self) -> Dict[str, float]:
        if not self._fitted:
            return {
                "ou_reversion_speed": 0.0,
                "ou_equilibrium": 0.5,
                "ou_volatility": 0.0,
                "ou_half_life_hours": 24.0,
                "ou_has_data": 0.0,
            }
        half_life = np.log(2) / (self.a + 1e-10)
        return {
            "ou_reversion_speed": round(float(self.a), 4),
            "ou_equilibrium": round(float(self.b), 4),
            "ou_volatility": round(float(self.sigma), 4),
            "ou_half_life_hours": round(float(min(half_life, 168)), 2),
            "ou_has_data": 1.0,
        }


def add_ou_features_placeholder(df):
    """Add OU feature columns (neutral defaults) when no odds history is
    available yet, so the feature matrix has a consistent shape whether or
    not a given fight has odds data. Populated for real once
    ingestion/odds_ingestion.py is wired to a live provider.
    """
    ou_cols = {
        "ou_reversion_speed": 0.0,
        "ou_equilibrium": 0.5,
        "ou_volatility": 0.0,
        "ou_half_life_hours": 24.0,
        "ou_has_data": 0.0,
        "ou_clv_estimate": 0.0,
        "ou_steam_flag": 0.0,
    }
    for col, default in ou_cols.items():
        if col not in df.columns:
            df[col] = default
    return df
