"""
Fighter style clustering — archetype matchup features.

Ported from the live production engine (ufc-predictions-saas/engine/models/
style_cluster.py). Octagon had nothing like this: it clusters fighters
into style archetypes (Pressure Striker, Counter Striker, Wrestler/
Grappler, Submission Artist, Balanced) from career striking/grappling
stats using a hand-rolled numpy KMeans (no sklearn dependency needed for
this piece), then derives matchup features — same-style flag, an
archetype-vs-archetype win-rate advantage, and a centroid distance — that
feed into the ensemble alongside the differential features.

This is genuinely new signal for Octagon, not a redundant port: it's a
different axis (playstyle interaction) than the raw stat differentials
gbm_prophet.py already trains on.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


class StyleClusterer:
    CLUSTER_FEATURES = [
        "career_sig_str_landed", "career_sig_str_pct", "career_td_landed",
        "career_td_pct", "career_sub_att", "career_ctrl_time", "career_kd",
        "career_sig_head_landed", "career_sig_body_landed", "career_sig_leg_landed",
        "career_sig_dist_landed", "career_sig_clinch_landed", "career_sig_ground_landed",
    ]

    ARCHETYPE_LABELS = {
        0: "Pressure Striker",
        1: "Counter Striker",
        2: "Wrestler/Grappler",
        3: "Submission Artist",
        4: "Balanced",
    }

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.centroids = None
        self.scaler_mean = None
        self.scaler_std = None
        self.fighter_clusters: Dict[str, int] = {}
        self.matchup_matrix = None  # n_clusters x n_clusters win-rate matrix

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.scaler_mean) / (self.scaler_std + 1e-8)

    def fit(self, fighter_stats: pd.DataFrame, fight_outcomes: Optional[pd.DataFrame] = None):
        available = [f for f in self.CLUSTER_FEATURES if f in fighter_stats.columns]
        if len(available) < 3:
            available = [
                c for c in fighter_stats.columns
                if fighter_stats[c].dtype in [np.float64, np.int64, float, int]
                and c not in ("career_fights", "career_won")
            ][:10]

        X = fighter_stats[available].fillna(0).values
        names = (
            fighter_stats.index.tolist() if fighter_stats.index.dtype == object
            else (fighter_stats["fighter"].tolist() if "fighter" in fighter_stats.columns
                  else list(range(len(X))))
        )

        self.scaler_mean = X.mean(axis=0)
        self.scaler_std = X.std(axis=0)
        X_scaled = self._standardize(X)

        self.centroids = self._kmeans(X_scaled, self.n_clusters)

        for i, name in enumerate(names):
            dists = np.linalg.norm(X_scaled[i] - self.centroids, axis=1)
            self.fighter_clusters[name] = int(np.argmin(dists))

        if fight_outcomes is not None:
            self._build_matchup_matrix(fight_outcomes)
        return self

    def _kmeans(self, X: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
        """KMeans++ init + Lloyd's algorithm, pure numpy (no sklearn dep)."""
        rng = np.random.default_rng(42)
        n = len(X)
        idx = [int(rng.integers(n))]
        for _ in range(1, k):
            dists = np.min([np.sum((X - X[i]) ** 2, axis=1) for i in idx], axis=0)
            probs = dists / (dists.sum() + 1e-10)
            idx.append(int(rng.choice(n, p=probs)))

        centroids = X[idx].copy()
        for _ in range(max_iter):
            assignments = np.argmin(
                np.array([np.sum((X - c) ** 2, axis=1) for c in centroids]).T, axis=1
            )
            new_centroids = np.array([
                X[assignments == j].mean(axis=0) if (assignments == j).sum() > 0 else centroids[j]
                for j in range(k)
            ])
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids
        return centroids

    def _build_matchup_matrix(self, outcomes: pd.DataFrame):
        wins = np.zeros((self.n_clusters, self.n_clusters))
        total = np.zeros((self.n_clusters, self.n_clusters))

        for _, row in outcomes.iterrows():
            fa = row.get("header_fighter_a_name") or row.get("fighter_a", "")
            fb = row.get("header_fighter_b_name") or row.get("fighter_b", "")
            winner_a = row.get("winner_is_a", 0)

            ca = self.fighter_clusters.get(fa)
            cb = self.fighter_clusters.get(fb)
            if ca is None or cb is None:
                continue

            total[ca][cb] += 1
            total[cb][ca] += 1
            if winner_a:
                wins[ca][cb] += 1
            else:
                wins[cb][ca] += 1

        self.matchup_matrix = (wins + 1) / (total + 2)  # Laplace smoothing

    def get_cluster(self, fighter: str) -> int:
        return self.fighter_clusters.get(fighter, self.n_clusters // 2)

    def get_matchup_features(self, fighter_a: str, fighter_b: str) -> dict:
        ca, cb = self.get_cluster(fighter_a), self.get_cluster(fighter_b)
        features = {
            "style_cluster_a": ca,
            "style_cluster_b": cb,
            "style_same": 1.0 if ca == cb else 0.0,
            "style_cluster_diff": ca - cb,
        }
        if self.matchup_matrix is not None:
            features["style_matchup_adv_a"] = round(float(self.matchup_matrix[ca][cb] - 0.5), 4)
        else:
            features["style_matchup_adv_a"] = 0.0

        if self.centroids is not None:
            features["style_distance"] = round(float(np.linalg.norm(self.centroids[ca] - self.centroids[cb])), 4)
        else:
            features["style_distance"] = 0.0
        return features


def add_style_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add fa_style_cluster/fb_style_cluster/style_* matchup columns.

    Clusters are fit once on the full dataset's latest-known career stats
    (not re-fit per walk-forward window) — cheap and stable enough for
    feature generation; backtesting/engine.py refits everything else on
    each expanding window regardless.
    """
    fighter_stats = {}
    for _, row in df.iterrows():
        for side in ("a", "b"):
            name = row[f"header_fighter_{side}_name"]
            stats = {
                col.replace(f"f{side}_", ""): row[col]
                for col in df.columns if col.startswith(f"f{side}_career_")
            }
            if stats and any(v != 0 for v in stats.values() if isinstance(v, (int, float))):
                fighter_stats[name] = stats  # latest fight wins

    if len(fighter_stats) < 10:
        for col in ["fa_style_cluster", "fb_style_cluster", "style_same",
                     "style_cluster_diff", "style_matchup_adv_a", "style_distance"]:
            df[col] = 0.0
        return df

    stats_df = pd.DataFrame.from_dict(fighter_stats, orient="index").fillna(0)
    clusterer = StyleClusterer(n_clusters=min(5, len(stats_df) // 10))
    clusterer.fit(stats_df, df)

    rows = []
    for _, row in df.iterrows():
        feats = clusterer.get_matchup_features(row["header_fighter_a_name"], row["header_fighter_b_name"])
        rows.append({
            "fa_style_cluster": feats["style_cluster_a"],
            "fb_style_cluster": feats["style_cluster_b"],
            "style_same": feats["style_same"],
            "style_cluster_diff": feats["style_cluster_diff"],
            "style_matchup_adv_a": feats["style_matchup_adv_a"],
            "style_distance": feats["style_distance"],
        })
    return pd.concat([df, pd.DataFrame(rows, index=df.index)], axis=1)
