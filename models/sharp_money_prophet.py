"""
Sharp Money Prophet — reads line movement from odds_history and converts it
into a win-probability signal, using an Ornstein-Uhlenbeck (OU) model of
mean-reverting odds drift (the "relativistic statistical arbitrage" idea
carried over from the tennis engine).

STATUS: the OU math (estimate_ou_params) is real, pure, and unit-tested —
it works on any time series today. The prophet class around it is a stub
because it depends on ingestion/odds_ingestion.py, which isn't wired to a
live odds provider yet. Excluded from the live council blend until it is.
"""

import numpy as np
import polars as pl

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

    # dx = theta*mu*dt - theta*dt*x_lag + eps  ->  linear regression on x_lag
    A = np.vstack([np.ones_like(x_lag), x_lag]).T
    coeffs, *_ = np.linalg.lstsq(A, dx, rcond=None)
    a, b = coeffs  # dx ≈ a + b * x_lag

    theta = -b / dt
    mu = a / (theta * dt) if theta != 0 else float(np.mean(x))
    residuals = dx - (a + b * x_lag)
    sigma = float(np.std(residuals)) / np.sqrt(dt)

    return float(theta), float(mu), sigma


class SharpMoneyProphet(Prophet):
    name = "sharp_money"

    def fit(self, features: pl.DataFrame, labels: pl.Series) -> "SharpMoneyProphet":
        raise NotImplementedError(
            "SharpMoneyProphet needs odds_history populated by "
            "ingestion/odds_ingestion.py, which has no live provider wired up yet."
        )

    def predict_proba(self, features: pl.DataFrame) -> list[float]:
        raise NotImplementedError("SharpMoneyProphet is not trained — see fit().")
