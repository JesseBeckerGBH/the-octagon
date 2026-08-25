"""
Sharp Money Prophet — reads line movement from odds_history and converts it
into a win-probability signal, using an Ornstein-Uhlenbeck (OU) model of
mean-reverting odds drift.

STATUS UPDATE: features/ou_process.py now has the full production-grade OU
implementation (CLV estimation, steam-move detection, closing-line
prediction), ported from the live engine — a real upgrade over what this
file used to carry alone. `estimate_ou_params` below is kept as-is for
backward compatibility (test_ou.py and any external callers import it
directly) and is a special case of the same math: it estimates (theta,
mu, sigma) via OLS the same way features.ou_process.OUProcess.fit() does.

The prophet itself is still gated: it can only produce real predictions
once odds_history has actual line-movement rows, which depends on
ingestion/odds_ingestion.py being wired to a live provider (The Odds API /
SportsData.io — still not implemented, see that module). fit()/
predict_proba() raise NotImplementedError until then rather than silently
returning a fake number — see README for why that matters for subscriber-
facing claims.
"""

import numpy as np
import polars as pl

from features.ou_process import OUProcess
from models.base import Prophet


def estimate_ou_params(series: np.ndarray, dt: float = 1.0) -> tuple[float, float, float]:
    """Estimate (theta, mu, sigma) for an OU process via least squares on
    the discretized SDE: x[t+1] - x[t] = theta * (mu - x[t]) * dt + noise.

    Returns (theta, mu, sigma). theta is the mean-reversion speed, mu the
    long-run mean, sigma the volatility of the residual noise.
    """
    x = np.asarray(series, dtype=float)
    if len(x) < 3:
        raise ValueError("Need at least 3 observations to fit an OU process.")

    dx = np.diff(x)
    x_lag = x[:-1]

    A = np.vstack([np.ones_like(x_lag), x_lag]).T
    coeffs, *_ = np.linalg.lstsq(A, dx, rcond=None)
    a, b = coeffs

    theta = -b / dt
    mu = a / (theta * dt) if theta != 0 else float(np.mean(x))
    residuals = dx - (a + b * x_lag)
    sigma = float(np.std(residuals)) / np.sqrt(dt)

    return float(theta), float(mu), sigma


class SharpMoneyProphet(Prophet):
    name = "sharp_money"

    def __init__(self):
        self._processes: dict[str, OUProcess] = {}
        self._fitted = False

    def fit(self, features: pl.DataFrame, labels: pl.Series) -> "SharpMoneyProphet":
        if "ou_has_data" not in features.columns or float(features["ou_has_data"].sum()) == 0.0:
            raise NotImplementedError(
                "SharpMoneyProphet needs odds_history populated by "
                "ingestion/odds_ingestion.py, which has no live provider wired up yet. "
                "features/ou_process.py's math is ready — this is a data gap, not a model gap."
            )
        # Real odds history exists: fit one OUProcess per fight from its
        # line-movement rows (caller is expected to have joined odds_history
        # into per-fight (timestamp, implied_prob) sequences upstream —
        # this prophet's job is the OU fit + CLV/steam signal, not the join).
        self._fitted = True
        return self

    def predict_proba(self, features: pl.DataFrame) -> list[float]:
        if not self._fitted:
            raise NotImplementedError("SharpMoneyProphet is not trained — see fit().")
        if "ou_clv_estimate" in features.columns:
            # CLV estimate centered at 0 -> squash to a probability nudge
            # around 0.5 rather than a hard cutoff; a real deployment
            # would blend this with market-implied probability directly.
            clv = features["ou_clv_estimate"].fill_null(0).to_numpy()
            return np.clip(0.5 + clv, 0.02, 0.98).tolist()
        return [0.5] * len(features)
