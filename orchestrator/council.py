"""
The Council of Prophets — blends active prophets into one consensus
probability, with configurable weights, dissent detection, and a
production-proven calibrated-stacking blend mode.

Three blend modes, tried in this order:

  1. Calibrated stacking (fit_calibrated_stacker()) — the ensemble recipe
     ported from the live production engine
     (ufc-predictions-saas/engine/model/train_ensemble.py): 5-fold
     out-of-fold predictions per prophet, isotonic-calibrated per prophet,
     then a logistic-regression meta-learner fit on the calibrated OOF
     matrix. This is what the production model's honest ~0.21 OOF Brier
     score depends on — averaging raw, uncalibrated prophet outputs (mode
     3) is exactly the kind of thing that produces a headline-looking
     accuracy number with bad calibration underneath.
  2. Stacked meta-learner (LightGBM on raw prophet outputs) — the
     original Octagon design; kept as fit_stacker() for anyone who wants
     the lighter-weight version without per-fold refitting.
  3. Weighted average (default, works from prediction #1, needs no fitting).

Also applies a config-driven confidence/coverage gate
(orchestrator/gating.py) before returning results — the fix for the
finding that the live thebeastufc.com model loses money on 55-70%
confidence picks and has negative-ROI coverage drift in a couple of
weight classes (see README "Calibration gate").
"""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from models.base import Prophet
from orchestrator.gating import Gate, load_gate


@dataclass
class CouncilResult:
    blended_prob_a: float
    prophet_probs: dict[str, float]
    dissent: float  # max pairwise disagreement among active prophets
    method_probs: dict[str, float] | None = None
    gated: bool = False
    gate_reason: str | None = None


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class Council:
    def __init__(self, prophets: list[Prophet], weights: dict[str, float] | None = None,
                 gate: Gate | None = None):
        self.prophets = {p.name: p for p in prophets}
        self.weights = weights or {name: 1.0 / len(prophets) for name in self.prophets}
        self.meta_model: lgb.Booster | None = None  # mode 2 (legacy LightGBM stacker)
        self.calibrators: dict[str, IsotonicRegression] | None = None  # mode 1
        self.meta: LogisticRegression | None = None  # mode 1
        self._stack_order: list[str] = []
        self.gate = gate or Gate()
        self.method_classifier = None  # optional auxiliary model, see set_method_classifier()

    @classmethod
    def from_config(cls, prophets: list[Prophet], config_path: str = "config/settings.yaml") -> "Council":
        cfg = load_config(config_path)
        status = cfg["council"]["status"]
        active = [p for p in prophets if status.get(p.name) in ("active", "experimental")]
        weights = {name: w for name, w in cfg["council"]["weights"].items()
                   if status.get(name) in ("active", "experimental")}
        return cls(active, weights, gate=load_gate(cfg))

    def set_method_classifier(self, clf) -> None:
        """Attach a fitted models.method_classifier.MethodClassifier. Its
        output rides along in CouncilResult.method_probs — it's an
        auxiliary prediction (method of victory), not part of the win-
        probability vote, so it never affects blended_prob_a.
        """
        self.method_classifier = clf

    def consensus(self, features: pl.DataFrame) -> list[CouncilResult]:
        prophet_probs = {name: p.predict_proba(features) for name, p in self.prophets.items()}

        method_probs_rows = None
        if self.method_classifier is not None and self.method_classifier._fitted:
            from features.engineer import get_feature_cols
            X = features.select(get_feature_cols(features)).fill_null(0).to_numpy()
            probs = self.method_classifier.predict_proba(X)
            method_probs_rows = [
                {"ko_tko": float(p[0]), "submission": float(p[1]), "decision": float(p[2])}
                for p in probs
            ]

        results = []
        n = len(features)
        for i in range(n):
            row_probs = {name: probs[i] for name, probs in prophet_probs.items()}

            if self.calibrators is not None and self.meta is not None:
                blended = self._calibrated_stacked_predict(row_probs)
            elif self.meta_model is not None:
                blended = self._stacked_predict(row_probs)
            else:
                total_weight = sum(self.weights.get(name, 0) for name in row_probs) or 1.0
                blended = sum(row_probs[name] * self.weights.get(name, 0) for name in row_probs) / total_weight

            values = list(row_probs.values())
            dissent = max(values) - min(values) if len(values) > 1 else 0.0

            weight_class = None
            if "weight_class" in features.columns:
                weight_class = features["weight_class"][i]

            gated, reason = self.gate.check(blended, weight_class)

            results.append(CouncilResult(
                blended_prob_a=blended,
                prophet_probs=row_probs,
                dissent=dissent,
                method_probs=method_probs_rows[i] if method_probs_rows else None,
                gated=gated,
                gate_reason=reason,
            ))
        return results

    # ── Mode 1: calibrated stacking (the production-proven recipe) ──────

    def fit_calibrated_stacker(self, features: pl.DataFrame, labels: pl.Series, n_splits: int = 5) -> dict:
        """5-fold out-of-fold isotonic calibration per prophet + logistic
        stacking meta-learner, ported from
        ufc-predictions-saas/engine/model/train_ensemble.py. Refits every
        active prophet fresh on each fold (via copy.deepcopy — cheap for
        the sklearn/xgboost/lightgbm wrappers here) so the OOF predictions
        are honestly out-of-sample, then refits every prophet on the FULL
        dataset afterward so they're ready to serve.

        Returns a small metrics dict (oof_accuracy, oof_brier) so callers
        can log/report what the calibration step actually achieved,
        rather than trusting it blindly.
        """
        import copy

        from sklearn.metrics import accuracy_score, brier_score_loss

        y = labels.to_numpy().astype(int)
        self._stack_order = sorted(self.prophets.keys())

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_raw = {name: np.zeros(len(y)) for name in self._stack_order}

        for tr_idx, te_idx in skf.split(np.zeros(len(y)), y):
            tr_mask = np.zeros(len(y), dtype=bool)
            tr_mask[tr_idx] = True
            te_mask = ~tr_mask

            X_train = features.filter(pl.Series(tr_mask))
            X_test = features.filter(pl.Series(te_mask))
            y_train = pl.Series(y[tr_idx])

            for name in self._stack_order:
                fold_prophet = copy.deepcopy(self.prophets[name])
                fold_prophet.fit(X_train, y_train)
                oof_raw[name][te_idx] = fold_prophet.predict_proba(X_test)

        self.calibrators = {}
        cal_oof = []
        for name in self._stack_order:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(oof_raw[name], y)
            self.calibrators[name] = iso
            cal_oof.append(iso.transform(oof_raw[name]))

        self.meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
        self.meta.fit(np.column_stack(cal_oof), y)

        ens_oof = self.meta.predict_proba(np.column_stack(cal_oof))[:, 1]

        # Refit every prophet on ALL the data — these are the versions
        # that actually serve predictions; the fold copies above only
        # ever existed to produce honest OOF estimates.
        for name, prophet in self.prophets.items():
            prophet.fit(features, labels)

        return {
            "oof_accuracy": float(accuracy_score(y, ens_oof >= 0.5)),
            "oof_brier": float(brier_score_loss(y, ens_oof)),
            "n_fights": len(y),
        }

    def _calibrated_stacked_predict(self, row_probs: dict[str, float]) -> float:
        cal = [self.calibrators[name].transform([row_probs[name]])[0] for name in self._stack_order]
        return float(self.meta.predict_proba(np.array(cal).reshape(1, -1))[:, 1][0])

    # ── Mode 2: legacy LightGBM-on-raw-probs stacker ─────────────────────

    def _stacked_predict(self, row_probs: dict[str, float]) -> float:
        ordered = np.array([[row_probs[name] for name in sorted(row_probs)]])
        return float(self.meta_model.predict(ordered)[0])

    def fit_stacker(self, oof_prophet_probs: pl.DataFrame, labels: pl.Series) -> None:
        """Train a LightGBM meta-learner directly on raw (uncalibrated)
        out-of-fold prophet outputs. Lighter-weight than
        fit_calibrated_stacker() (no per-prophet isotonic calibration
        step, no refit-per-fold), but for that same reason it's the one
        prior analysis flagged as a likely contributor to the live
        model's calibration gap — prefer fit_calibrated_stacker() unless
        you specifically want this simpler mode.
        """
        X = oof_prophet_probs.select(sorted(oof_prophet_probs.columns)).to_numpy()
        y = labels.to_numpy()
        train_set = lgb.Dataset(X, label=y)
        self.meta_model = lgb.train({"objective": "binary", "verbose": -1}, train_set, num_boost_round=100)
