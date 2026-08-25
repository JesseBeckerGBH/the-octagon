#!/usr/bin/env python3
"""
Walk-forward backtest engine.

Ported from the live production engine
(ufc-predictions-saas/engine/backtesting/backtest_walkforward.py) — this
is the actual tool that exposed the confidence-bucket ROI problem and the
Women's Flyweight / Catch Weight coverage drift in the analysis of
thebeastufc.com. Octagon had nothing here before (walk_forward_backtest()
was a NotImplementedError stub); this port is what turns "we validated on
a held-out set" into "we validated the way we'll actually operate" —
training only on fights strictly before the fights being scored, sliding
forward, and aggregating Brier/ROI/accuracy across every window.

Deliberately reuses Octagon's own Prophet classes (models/gbm_prophet.py
etc.) rather than re-declaring separate XGBoost/LightGBM/etc. wrapper
classes the way the production script did — same hyperparameters, but one
source of truth instead of two copies that can drift apart. The cost is a
pandas<->polars conversion per walk-forward step, which is a fine trade
for a codebase this size; if that ever becomes the bottleneck on a much
larger fight history, reintroducing raw-numpy wrappers (like production's
BaseXGB/BaseLGB) is the documented escape hatch.

Usage:
    python -m backtesting.engine                    # full backtest
    python -m backtesting.engine --start-year 2020
    python -m backtesting.engine --models gbm lightgbm
    python -m backtesting.engine --report
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")

from features.leak_safe_features import get_feature_cols  # noqa: E402
from models.gbm_prophet import GBMProphet  # noqa: E402
from models.lightgbm_prophet import LightGBMProphet  # noqa: E402
from models.logistic_prophet import LogisticProphet  # noqa: E402
from models.randomforest_prophet import RandomForestProphet  # noqa: E402

FEATURES_PATH = Path("data/processed/ufc_features_ml_ready.csv")
BACKTEST_DIR = Path("backtesting/results")

PROPHET_REGISTRY = {
    "gbm": GBMProphet,
    "lightgbm": LightGBMProphet,
    "logistic": LogisticProphet,
    "randomforest": RandomForestProphet,
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def isotonic_calibrate(train_probs: np.ndarray, train_labels: np.ndarray, test_probs: np.ndarray) -> np.ndarray:
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(train_probs, train_labels)
    return ir.transform(test_probs)


def stack_predictions(base_preds_train: list, y_train: np.ndarray, base_preds_test: list):
    """Logistic-regression stacking meta-learner over calibrated base
    predictions. Returns (ensemble_prob, {base_i: coef}).
    """
    from sklearn.linear_model import LogisticRegression
    X_train = np.column_stack(base_preds_train)
    X_test = np.column_stack(base_preds_test)
    meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
    meta.fit(X_train, y_train)
    ensemble_prob = meta.predict_proba(X_test)[:, 1]
    weights = dict(zip((f"base_{i}" for i in range(len(base_preds_train))), meta.coef_[0].tolist()))
    return ensemble_prob, weights


def compute_metrics(y_true, y_prob, label: str = "") -> dict:
    """Brier, log loss, AUC, accuracy, flat -110 ROI, avg Kelly fraction,
    Sharpe-like ratio, max drawdown. Identical formulas to the production
    script so numbers here are directly comparable to the numbers Jesse
    was handed in the "What needs to be done to Beast UFC.com" analysis.
    """
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    brier = brier_score_loss(y_true, y_prob)
    logloss = log_loss(y_true, y_prob, labels=[0, 1])
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    acc = float(np.mean((y_prob >= 0.5) == y_true))

    correct = (y_prob >= 0.5) == y_true
    roi_per_bet = np.where(correct, 100 / 110, -1.0)  # flat bet at -110 juice
    roi = float(np.mean(roi_per_bet))

    b = 100 / 110
    p = np.where(y_prob >= 0.5, y_prob, 1 - y_prob)
    q = 1 - p
    kelly = float(np.mean(np.clip((b * p - q) / b, 0, 0.25)))

    sharpe = float(np.mean(roi_per_bet) / np.std(roi_per_bet) * np.sqrt(len(roi_per_bet))) if np.std(roi_per_bet) > 0 else 0.0

    cumulative = np.cumsum(roi_per_bet)
    running_max = np.maximum.accumulate(cumulative)
    max_dd = float(np.max(running_max - cumulative)) if len(cumulative) > 0 else 0.0

    return {
        "label": label, "n_fights": int(len(y_true)),
        "brier_score": round(float(brier), 5), "log_loss": round(float(logloss), 5),
        "auc_roc": round(float(auc), 4), "accuracy": round(acc, 4),
        "flat_bet_roi": round(roi, 4), "avg_kelly_fraction": round(kelly, 4),
        "sharpe_ratio": round(sharpe, 2), "max_drawdown": round(max_dd, 2),
    }


def run_walk_forward(df: pd.DataFrame, prophet_keys: list[str], start_year: int = 2018,
                      min_train: int = 500, step_size: int = 50):
    """Expanding-window walk-forward backtest. At each step: train on all
    fights strictly before the window, predict the next `step_size`
    fights, isotonic-calibrate + logistic-stack, record, slide forward.
    """
    log(f"Walk-forward backtest: start_year={start_year}, step={step_size}, min_train={min_train}")

    feature_cols = get_feature_cols(df)
    log(f"  Using {len(feature_cols)} features")

    X = df[feature_cols].fillna(0)
    y = df["winner_is_a"].values.astype(int)
    dates = pd.to_datetime(df["event_date_parsed"])

    start_mask = dates.dt.year >= start_year
    if not start_mask.any():
        log(f"  ERROR: no fights from {start_year} onward")
        return None
    start_idx = max(int(start_mask.idxmax()), min_train)
    log(f"  Training on fights 0-{start_idx - 1}, testing from index {start_idx}")

    all_results = []
    idx, step_count = start_idx, 0

    while idx < len(df):
        end_idx = min(idx + step_size, len(df))
        X_train_pl = pl.from_pandas(X.iloc[:idx].reset_index(drop=True))
        X_test_pl = pl.from_pandas(X.iloc[idx:end_idx].reset_index(drop=True))
        y_train = y[:idx]
        y_test = y[idx:end_idx]
        if len(X_test_pl) == 0:
            break

        base_train, base_test, names = [], [], []
        for key in prophet_keys:
            prophet = PROPHET_REGISTRY[key](feature_cols=feature_cols)
            prophet.fit(X_train_pl, pl.Series(y_train))
            train_probs = np.array(prophet.predict_proba(X_train_pl))
            test_probs = np.array(prophet.predict_proba(X_test_pl))

            base_train.append(isotonic_calibrate(train_probs, y_train, train_probs))
            base_test.append(isotonic_calibrate(train_probs, y_train, test_probs))
            names.append(prophet.name)

        if len(base_train) > 1:
            ensemble_prob, _ = stack_predictions(base_train, y_train, base_test)
        else:
            ensemble_prob = base_test[0]

        for i in range(len(X_test_pl)):
            fight_idx = idx + i
            row = df.iloc[fight_idx]
            result = {
                "fight_idx": fight_idx,
                "date": row["event_date_parsed"],
                "event": row.get("event_name", ""),
                "fighter_a": row["header_fighter_a_name"],
                "fighter_b": row["header_fighter_b_name"],
                "weight_class": row.get("weight_class", ""),
                "actual_winner_is_a": int(y_test[i]),
                "ensemble_prob_a": round(float(ensemble_prob[i]), 4),
                "predicted_winner": row["header_fighter_a_name"] if ensemble_prob[i] >= 0.5 else row["header_fighter_b_name"],
                "correct": int((ensemble_prob[i] >= 0.5) == y_test[i]),
                "confidence": round(float(max(ensemble_prob[i], 1 - ensemble_prob[i])), 4),
            }
            for j, name in enumerate(names):
                result[f"prob_{name}"] = round(float(base_test[j][i]), 4)
            all_results.append(result)

        idx = end_idx
        step_count += 1
        if step_count % 10 == 0:
            log(f"  step {step_count}: trained on {idx} fights")

    log(f"  Completed {step_count} steps, {len(all_results)} predictions")
    return pd.DataFrame(all_results)


def generate_report(results_df: pd.DataFrame) -> str:
    y_true = results_df["actual_winner_is_a"].values
    y_prob = results_df["ensemble_prob_a"].values

    overall = compute_metrics(y_true, y_prob, "ENSEMBLE (overall)")

    model_metrics = [
        compute_metrics(y_true, results_df[col].values, col.replace("prob_", ""))
        for col in results_df.columns if col.startswith("prob_")
    ]

    wc_metrics = [
        compute_metrics(y_true[results_df["weight_class"] == wc], y_prob[results_df["weight_class"] == wc], wc)
        for wc in results_df["weight_class"].unique()
        if (results_df["weight_class"] == wc).sum() >= 20
    ]

    conf_metrics = []
    for lo, hi, label in [(0.5, 0.55, "50-55%"), (0.55, 0.6, "55-60%"), (0.6, 0.65, "60-65%"),
                            (0.65, 0.7, "65-70%"), (0.7, 1.0, "70%+")]:
        mask = (results_df["confidence"] >= lo) & (results_df["confidence"] < hi)
        if mask.sum() >= 10:
            conf_metrics.append(compute_metrics(y_true[mask], y_prob[mask], label))

    lines = ["=" * 70, "OCTAGON WALK-FORWARD BACKTEST REPORT",
              f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "=" * 70]

    def fmt(m, header):
        lines.append(f"\n{'-' * 50}\n{header}\n{'-' * 50}")
        lines.append(f"  Fights:              {m['n_fights']}")
        lines.append(f"  Brier Score:         {m['brier_score']:.5f} (lower is better, baseline=0.25)")
        lines.append(f"  Log Loss:            {m['log_loss']:.5f}")
        lines.append(f"  AUC-ROC:             {m['auc_roc']:.4f}")
        lines.append(f"  Accuracy:            {m['accuracy']:.1%}")
        lines.append(f"  Flat-Bet ROI:        {m['flat_bet_roi']:+.2%} (at -110 juice)")
        lines.append(f"  Avg Kelly Fraction:  {m['avg_kelly_fraction']:.2%}")
        lines.append(f"  Sharpe Ratio:        {m['sharpe_ratio']:.2f}")
        lines.append(f"  Max Drawdown:        {m['max_drawdown']:.2f} units")

    fmt(overall, "ENSEMBLE PERFORMANCE")

    lines.append(f"\n{'-' * 50}\nBASE PROPHET COMPARISON\n{'-' * 50}")
    for m in model_metrics:
        lines.append(f"  {m['label']}: Brier={m['brier_score']:.5f} Acc={m['accuracy']:.1%} "
                       f"ROI={m['flat_bet_roi']:+.2%} AUC={m['auc_roc']:.4f}")

    if wc_metrics:
        lines.append(f"\n{'-' * 50}\nBY WEIGHT CLASS\n{'-' * 50}")
        for m in sorted(wc_metrics, key=lambda x: x["brier_score"]):
            lines.append(f"  {m['label']:<28} n={m['n_fights']:>5} Brier={m['brier_score']:.5f} "
                           f"Acc={m['accuracy']:.1%} ROI={m['flat_bet_roi']:+.2%}")

    if conf_metrics:
        lines.append(f"\n{'-' * 50}\nBY CONFIDENCE BUCKET\n{'-' * 50}")
        for m in conf_metrics:
            lines.append(f"  {m['label']:<10} n={m['n_fights']:>5} Acc={m['accuracy']:.1%} ROI={m['flat_bet_roi']:+.2%}")
        lines.append(
            "\n  Any bucket here with negative ROI is a candidate for "
            "config/settings.yaml's calibration.suppressed_confidence_bands "
            "(orchestrator/gating.py) — that's exactly how the live "
            "thebeastufc.com 55-70% gap was found."
        )

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


def walk_forward_backtest(prophet_keys: list[str] | None = None, start_year: int = 2018,
                            min_train: int = 500, step_size: int = 50) -> pd.DataFrame:
    """Public entry point — the old NotImplementedError stub is gone."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found. Run ingestion/hf_pipeline.py then "
            "features/leak_safe_features.py first."
        )
    df = pd.read_csv(FEATURES_PATH)
    df["event_date_parsed"] = pd.to_datetime(df["event_date_parsed"])
    df = df.sort_values("event_date_parsed").reset_index(drop=True)

    keys = prophet_keys or list(PROPHET_REGISTRY.keys())
    results_df = run_walk_forward(df, keys, start_year=start_year, min_train=min_train, step_size=step_size)
    if results_df is None or results_df.empty:
        raise RuntimeError("Walk-forward backtest produced no results — check start_year/min_train against your data.")

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(BACKTEST_DIR / "backtest_results.csv", index=False)
    report = generate_report(results_df)
    (BACKTEST_DIR / "backtest_report.txt").write_text(report)
    return results_df


def main() -> None:
    args = sys.argv[1:]
    start_year = 2018
    prophet_keys = list(PROPHET_REGISTRY.keys())

    if "--start-year" in args:
        start_year = int(args[args.index("--start-year") + 1])
    if "--models" in args:
        i = args.index("--models") + 1
        prophet_keys = []
        while i < len(args) and not args[i].startswith("--"):
            if args[i] in PROPHET_REGISTRY:
                prophet_keys.append(args[i])
            i += 1
    if "--report" in args:
        report_path = BACKTEST_DIR / "backtest_report.txt"
        print(report_path.read_text() if report_path.exists() else "No saved report — run the backtest first.")
        return

    results_df = walk_forward_backtest(prophet_keys, start_year=start_year)
    print((BACKTEST_DIR / "backtest_report.txt").read_text())
    log(f"Backtest complete: {len(results_df)} predictions, results in {BACKTEST_DIR.resolve()}")


if __name__ == "__main__":
    main()
