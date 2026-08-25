import pandas as pd

from features.style_cluster import StyleClusterer


def _fake_fighter_stats(n=20, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    names = [f"Fighter {i}" for i in range(n)]
    data = {c: rng.normal(size=n) for c in StyleClusterer.CLUSTER_FEATURES}
    return pd.DataFrame(data, index=names)


def test_fit_assigns_every_fighter_to_a_cluster():
    stats = _fake_fighter_stats()
    clusterer = StyleClusterer(n_clusters=3).fit(stats)
    assert len(clusterer.fighter_clusters) == len(stats)
    assert all(0 <= c < 3 for c in clusterer.fighter_clusters.values())


def test_matchup_features_same_style_flag():
    stats = _fake_fighter_stats()
    clusterer = StyleClusterer(n_clusters=3).fit(stats)
    names = list(clusterer.fighter_clusters.keys())
    feats = clusterer.get_matchup_features(names[0], names[0])
    assert feats["style_same"] == 1.0
    assert feats["style_cluster_diff"] == 0
    assert feats["style_distance"] == 0.0


def test_unknown_fighter_gets_default_balanced_cluster():
    stats = _fake_fighter_stats()
    clusterer = StyleClusterer(n_clusters=5).fit(stats)
    assert clusterer.get_cluster("Someone Not In The Data") == 2  # n_clusters // 2
