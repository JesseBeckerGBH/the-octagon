"""
Markov Form Chain — fighter momentum as a 4-state Markov chain.

Ported from the live production engine (ufc-predictions-saas/engine/models/
markov_form.py), which Octagon's own models/markov_prophet.py only
approximated with a 3-state (hot/neutral/cold) heuristic and hand-picked
win probabilities. This is the real thing: a HOT / NEUTRAL / COLD /
DECLINING state classification, with transition probabilities *learned*
from historical fight data (Laplace-smoothed frequency counts) rather than
guessed.

    HOT        -> 3+ recent wins, rising performance
    NEUTRAL    -> mixed results, stable
    COLD       -> 2+ recent losses, declining
    DECLINING  -> was hot, now losing (a momentum shift, not just "cold")

models/markov_prophet.py wraps this chain and fits a small logistic
regression from (state, momentum) -> P(fighter_a wins) on real labels,
replacing the old hardcoded 0.62/0.50/0.38 table.
"""

from enum import IntEnum
from typing import Optional

import numpy as np
import pandas as pd


class FormState(IntEnum):
    COLD = 0
    DECLINING = 1
    NEUTRAL = 2
    HOT = 3


STATE_LABELS = {
    FormState.COLD: "Cold",
    FormState.DECLINING: "Declining",
    FormState.NEUTRAL: "Neutral",
    FormState.HOT: "Hot",
}


class MarkovFormChain:
    """Estimates fighter form as a discrete Markov chain. Transition
    probabilities are learned from historical data (fit()); state
    classification itself is a simple, explainable rule over the last 3-5
    results so that "why is this fighter Hot?" always has a plain answer.
    """

    def __init__(self):
        # 4x4 transition matrix: T[i][j] = P(next=j | current=i)
        self.transition_matrix = np.ones((4, 4)) / 4  # uniform prior
        self.fighter_histories: dict[str, list] = {}

    def _classify_state(self, recent_results: list) -> FormState:
        """Classify a fighter's current form from their last 3-5 fights.

        `recent_results`: list of 1 (win) or 0 (loss), most recent last.
        """
        if len(recent_results) == 0:
            return FormState.NEUTRAL

        last_3 = recent_results[-3:] if len(recent_results) >= 3 else recent_results
        win_rate = sum(last_3) / len(last_3)

        # Momentum shift: was winning, now losing.
        if len(recent_results) >= 4:
            prev_3 = recent_results[-6:-3] if len(recent_results) >= 6 else recent_results[:-3]
            if prev_3 and sum(prev_3) / len(prev_3) >= 0.66 and win_rate <= 0.33:
                return FormState.DECLINING

        if win_rate >= 0.66:
            return FormState.HOT
        elif win_rate <= 0.33:
            return FormState.COLD
        else:
            return FormState.NEUTRAL

    def fit(self, fight_records: pd.DataFrame):
        """Learn transition probabilities from historical fight data.

        `fight_records` columns: fighter (str), date (datetime), won (0/1).
        """
        records = fight_records.sort_values(["fighter", "date"]).copy()
        transitions = np.zeros((4, 4))

        for fighter, group in records.groupby("fighter"):
            results = group["won"].tolist()
            self.fighter_histories[fighter] = results

            states = []
            for i in range(len(results)):
                window = results[max(0, i - 4): i + 1]
                states.append(self._classify_state(window))

            for i in range(len(states) - 1):
                transitions[states[i]][states[i + 1]] += 1

        for i in range(4):
            row_sum = transitions[i].sum() + 4  # Laplace smoothing
            self.transition_matrix[i] = (transitions[i] + 1) / row_sum

        return self

    def get_state(self, fighter: str, fight_results: Optional[list] = None):
        """Returns (current_state, next_state_transition_probs)."""
        if fight_results is not None:
            results = fight_results
        elif fighter in self.fighter_histories:
            results = self.fighter_histories[fighter]
        else:
            return FormState.NEUTRAL, self.transition_matrix[FormState.NEUTRAL]

        current = self._classify_state(results)
        return current, self.transition_matrix[current]

    def get_features(self, fighter: str, fight_results: Optional[list] = None) -> dict:
        """Feature dict for the ML pipeline: form_state, hot/cold/declining
        flags, a momentum score in [-1, 1], and the Markov-predicted
        P(next state is Hot).
        """
        state, next_probs = self.get_state(fighter, fight_results)

        results = fight_results if fight_results else self.fighter_histories.get(fighter, [])
        if results:
            weights = np.exp(np.linspace(-1, 0, min(len(results), 5)))
            recent = results[-5:] if len(results) >= 5 else results
            scaled = [2 * r - 1 for r in recent]
            weights = weights[-len(scaled):]
            momentum = float(np.average(scaled, weights=weights))
        else:
            momentum = 0.0

        return {
            "form_state": int(state),
            "form_state_hot": 1.0 if state == FormState.HOT else 0.0,
            "form_state_cold": 1.0 if state == FormState.COLD else 0.0,
            "form_state_declining": 1.0 if state == FormState.DECLINING else 0.0,
            "form_momentum": round(momentum, 4),
            "form_next_hot_prob": round(float(next_probs[FormState.HOT]), 4),
        }


def add_markov_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add leak-safe fa_form_*/fb_form_*/diff_form_* columns to a fight-level
    DataFrame with header_fighter_a_name/header_fighter_b_name/
    event_date_parsed/header_fighter_{a,b}_outcome columns.

    Each fight's features are computed from ONLY that fighter's fights
    strictly before it — history is updated after generating features, not
    before.
    """
    df = df.sort_values("event_date_parsed").reset_index(drop=True)

    records = []
    for _, row in df.iterrows():
        for side in ("a", "b"):
            records.append({
                "fighter": row[f"header_fighter_{side}_name"],
                "date": row["event_date_parsed"],
                "won": 1 if row.get(f"header_fighter_{side}_outcome") == "W" else 0,
            })
    records_df = pd.DataFrame(records).sort_values(["fighter", "date"])

    chain = MarkovFormChain()
    chain.fit(records_df)

    fighter_results: dict[str, list] = {}
    form_a, form_b = [], []
    for _, row in df.iterrows():
        fa, fb = row["header_fighter_a_name"], row["header_fighter_b_name"]

        form_a.append({f"fa_{k}": v for k, v in chain.get_features(fa, fighter_results.get(fa, [])).items()})
        form_b.append({f"fb_{k}": v for k, v in chain.get_features(fb, fighter_results.get(fb, [])).items()})

        won_a = 1 if row.get("header_fighter_a_outcome") == "W" else 0
        won_b = 1 if row.get("header_fighter_b_outcome") == "W" else 0
        fighter_results.setdefault(fa, []).append(won_a)
        fighter_results.setdefault(fb, []).append(won_b)

    fa_df = pd.DataFrame(form_a, index=df.index)
    fb_df = pd.DataFrame(form_b, index=df.index)

    diff_df = pd.DataFrame(index=df.index)
    diff_df["diff_form_state"] = fa_df["fa_form_state"] - fb_df["fb_form_state"]
    diff_df["diff_form_momentum"] = fa_df["fa_form_momentum"] - fb_df["fb_form_momentum"]

    return pd.concat([df, fa_df, fb_df, diff_df], axis=1)
