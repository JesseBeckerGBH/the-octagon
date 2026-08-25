"""
Method-of-Victory Classifier — predicts HOW a fight ends, not just who wins.

Ported from the live production engine (ufc-predictions-saas/engine/models/
method_classifier.py). This is genuinely new for Octagon — not a
port-and-replace of an existing stub, an entirely new prediction
surface: P(KO/TKO), P(Submission), P(Decision) per fight.

Doesn't implement the Prophet interface (models/base.py) on purpose: a
Prophet returns a single P(fighter_a wins) float, and method-of-victory is
a 3-class problem over the fight itself, not a per-fighter win
probability. orchestrator/council.py treats it as an auxiliary model:
Council.consensus() calls it separately and attaches method_probs to the
result, rather than blending it into the win-probability vote.
"""

from typing import Optional

import numpy as np
import pandas as pd

METHOD_CLASSES = ["KO/TKO", "Submission", "Decision"]
METHOD_MAP = {"KO/TKO": 0, "Submission": 1, "Decision": 2}


def classify_method(method_str: str) -> Optional[str]:
    """Normalize a raw finish-method string into one of our 3 classes.
    Returns None for No Contest, DQ, or unparseable strings.
    """
    if not method_str or not isinstance(method_str, str):
        return None

    m = method_str.lower().strip()

    if any(kw in m for kw in ["ko", "tko", "knockout", "doctor", "punch", "kick",
                                "knee", "elbow", "strikes", "cut", "slam"]):
        return "KO/TKO"

    if any(kw in m for kw in ["sub", "submission", "choke", "armbar", "triangle",
                                "guillotine", "kimura", "rear naked", "rnc",
                                "heel hook", "lock", "crank", "tap"]):
        return "Submission"

    if any(kw in m for kw in ["decision", "unanimous", "split", "majority", "draw", "dec"]):
        return "Decision"

    if m.startswith(("u-dec", "s-dec", "m-dec")):
        return "Decision"

    return None


class MethodClassifier:
    """Multi-class classifier for fight finish method. Prefers LightGBM;
    falls back to multinomial logistic regression, then to bare class
    priors if neither is importable (mirrors production's degradation
    path — it should never hard-fail just because a dependency is
    missing in some environment).
    """

    def __init__(self):
        self.model = None
        self._fitted = False
        self.class_priors = np.array([0.30, 0.20, 0.50])  # rough UFC priors

    def fit(self, X: np.ndarray, y: np.ndarray):
        try:
            from lightgbm import LGBMClassifier
            self.model = LGBMClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.05,
                num_class=3, objective="multiclass", verbose=-1, random_state=42,
            )
            self.model.fit(X, y)
        except ImportError:
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                self.model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(multi_class="multinomial", max_iter=1000,
                                               solver="lbfgs", random_state=42)),
                ])
                self.model.fit(X, y)
            except ImportError:
                unique, counts = np.unique(y, return_counts=True)
                self.class_priors = np.zeros(3)
                for u, c in zip(unique, counts):
                    self.class_priors[int(u)] = c
                self.class_priors /= self.class_priors.sum()
                self.model = None

        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns shape (n, 3): [P(KO/TKO), P(Submission), P(Decision)]."""
        if self.model is not None:
            probs = self.model.predict_proba(X)
            if probs.shape[1] < 3:
                full = np.tile(self.class_priors, (len(X), 1))
                for i, cls in enumerate(self.model.classes_):
                    full[:, int(cls)] = probs[:, i]
                probs = full
            return probs
        return np.tile(self.class_priors, (len(X), 1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


def add_method_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add method_class (str) / method_class_idx (int) columns from
    whichever raw finish-method column is present. Defaults unparseable
    rows to Decision, the modal outcome.
    """
    method_col = next(
        (c for c in ("header_finish_details_detailed", "finish_method", "method", "header_method")
         if c in df.columns), None,
    )
    if method_col is None:
        df["method_class"] = "Decision"
        df["method_class_idx"] = 2
        return df

    df["method_class"] = df[method_col].apply(lambda x: classify_method(str(x)) if pd.notna(x) else "Decision")
    df["method_class"] = df["method_class"].fillna("Decision")
    df["method_class_idx"] = df["method_class"].map(METHOD_MAP).fillna(2).astype(int)
    return df
