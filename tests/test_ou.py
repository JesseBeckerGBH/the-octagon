import numpy as np

from models.sharp_money_prophet import estimate_ou_params


def test_ou_recovers_known_mean_reverting_series():
    rng = np.random.default_rng(42)
    theta_true, mu_true, sigma_true = 0.3, 100.0, 1.0

    x = np.empty(500)
    x[0] = mu_true
    for t in range(1, len(x)):
        x[t] = x[t - 1] + theta_true * (mu_true - x[t - 1]) + rng.normal(0, sigma_true)

    theta_hat, mu_hat, sigma_hat = estimate_ou_params(x)

    assert abs(mu_hat - mu_true) < 5.0
    assert theta_hat > 0  # mean-reverting, not diverging


def test_ou_requires_minimum_observations():
    import pytest

    with pytest.raises(ValueError):
        estimate_ou_params([1.0, 2.0])
