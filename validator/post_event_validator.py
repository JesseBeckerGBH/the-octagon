#!/usr/bin/env python3
"""
Post-event validator — the accountability layer. Resolves every prediction
in `predictions` against the real outcome once it's known, and writes an
immutable row to `validation_log`. This table is what makes monthly
subscriber reports and investor demos credible: every number traces back
to a prediction that was logged *before* the fight.

Run daily via the scheduler (deploy/docker-compose.prod.yml's
octagon-validator service) or manually with --once.
"""

import argparse
import uuid
from datetime import datetime
from pathlib import Path

import duckdb

DB_PATH = Path("data/processed/octagon.duckdb")


def brier_score(probs: list[float], outcomes: list[int]) -> float:
    """Mean squared error between predicted probabilities and 0/1 outcomes.

    Lower is better/sharper. This is the project's primary calibration
    metric — see README for why raw accuracy alone is misleading.
    """
    if len(probs) != len(outcomes):
        raise ValueError("probs and outcomes must be the same length")
    if not probs:
        return 0.0
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def log_loss_contrib(prob: float, outcome: int, eps: float = 1e-15) -> float:
    p = min(max(prob, eps), 1 - eps)
    return -(outcome * __import__("math").log(p) + (1 - outcome) * __import__("math").log(1 - p))


def clv_realized(opening_prob: float, closing_prob: float, our_prob: float) -> float:
    """Closing-line-value proxy: did our number move the same direction the
    market moved, and by how much relative to the market's own drift?
    Positive means we anticipated the market move rather than followed it.
    """
    market_drift = closing_prob - opening_prob
    our_edge_at_open = our_prob - opening_prob
    if market_drift == 0:
        return 0.0
    return our_edge_at_open / abs(market_drift)


def resolve_pending_predictions(con: duckdb.DuckDBPyConnection) -> int:
    """Find predictions whose fight now has a known winner and hasn't been
    validated yet, score them, and write validation_log rows. Returns the
    number resolved.
    """
    pending = con.execute(
        """
        SELECT p.pred_id, p.blended_win_prob_a, p.predicted_winner,
               f.fighter_a, f.fighter_b, f.winner, f.method, f.round
        FROM predictions p
        JOIN fights f ON p.fight_id = f.fight_id
        LEFT JOIN validation_log v ON p.pred_id = v.pred_id
        WHERE f.winner IS NOT NULL AND v.validation_id IS NULL
        """
    ).fetchall()

    resolved = 0
    for (pred_id, prob_a, predicted_winner, fighter_a, fighter_b,
         winner, method, round_) in pending:
        outcome_a_won = 1 if winner == fighter_a else 0
        brier = brier_score([prob_a], [outcome_a_won])
        ll = log_loss_contrib(prob_a, outcome_a_won)
        correct = (predicted_winner == winner)

        con.execute(
            """
            INSERT INTO validation_log
            (validation_id, pred_id, actual_winner, actual_method, actual_round,
             delta_win_prob, brier_score_contrib, log_loss_contrib, clv_realized,
             paper_outcome, roi_contrib, validated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()), pred_id, winner, method, round_,
                prob_a - outcome_a_won, brier, ll,
                "win" if correct else "loss",
                None,  # ROI needs paper_bet_size + odds at bet time; TODO once
                       # odds_ingestion is live.
                datetime.utcnow(),
            ),
        )
        resolved += 1

    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily", action="store_true", help="Run as a scheduled daily job")
    parser.add_argument("--report", action="store_true", help="Print a summary after resolving")
    args = parser.parse_args()

    con = duckdb.connect(str(DB_PATH))
    try:
        n = resolve_pending_predictions(con)
        print(f"Resolved {n} prediction(s).")
        if args.report:
            row = con.execute(
                "SELECT AVG(brier_score_contrib), COUNT(*) FROM validation_log"
            ).fetchone()
            print(f"Overall Brier score: {row[0]:.4f} over {row[1]} validated predictions."
                  if row[0] is not None else "No validated predictions yet.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
