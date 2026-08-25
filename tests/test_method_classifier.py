import numpy as np
import pandas as pd

from models.method_classifier import METHOD_CLASSES, add_method_labels, classify_method


def test_classify_method_ko():
    assert classify_method("KO/TKO (Punches)") == "KO/TKO"
    assert classify_method("Knockout") == "KO/TKO"


def test_classify_method_submission():
    assert classify_method("Submission (Rear Naked Choke)") == "Submission"
    assert classify_method("Triangle Choke") == "Submission"


def test_classify_method_decision():
    assert classify_method("Decision (Unanimous)") == "Decision"
    assert classify_method("U-DEC") == "Decision"


def test_classify_method_unparseable_returns_none():
    assert classify_method("") is None
    assert classify_method(None) is None
    assert classify_method("Overturned") is None


def test_add_method_labels_defaults_to_decision_when_no_column_present():
    df = pd.DataFrame({"x": [1, 2]})
    out = add_method_labels(df)
    assert (out["method_class"] == "Decision").all()
    assert (out["method_class_idx"] == 2).all()


def test_method_classifier_predict_proba_shape_matches_three_classes():
    from models.method_classifier import MethodClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = rng.integers(0, 3, size=60)

    clf = MethodClassifier()
    clf.fit(X, y)
    probs = clf.predict_proba(X)

    assert probs.shape == (60, len(METHOD_CLASSES))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3)
