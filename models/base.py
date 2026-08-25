"""
Common interface every prophet in the Council implements.

Keeping this tiny and dependency-free on purpose: orchestrator/council.py
only needs fit/predict_proba/name to blend prophets, and a shared base
class is what lets us swap the 3-model MEGA council for the full 5-model
one without touching orchestration code.
"""

from abc import ABC, abstractmethod

import polars as pl


class Prophet(ABC):
    name: str

    @abstractmethod
    def fit(self, features: pl.DataFrame, labels: pl.Series) -> "Prophet":
        """Train on a feature table + label column of prior fight outcomes."""

    @abstractmethod
    def predict_proba(self, features: pl.DataFrame) -> list[float]:
        """Return P(fighter_a wins) for each row in features, in [0, 1]."""
