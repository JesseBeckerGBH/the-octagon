from orchestrator.gating import Gate, load_gate


def test_no_gate_by_default():
    gate = Gate()
    gated, reason = gate.check(0.58)
    assert not gated
    assert reason is None


def test_confidence_band_gates_when_configured():
    gate = Gate(suppressed_confidence_bands=[(0.55, 0.70)])
    gated, reason = gate.check(0.6)  # confidence = max(0.6, 0.4) = 0.6, in [0.55, 0.70)
    assert gated
    assert "60" in reason or "0.6" in reason.replace("%", "")


def test_confidence_outside_band_is_not_gated():
    gate = Gate(suppressed_confidence_bands=[(0.55, 0.70)])
    gated, _ = gate.check(0.9)
    assert not gated


def test_flagged_weight_class_gates_regardless_of_confidence():
    gate = Gate(flagged_weight_classes=["Catch Weight"])
    gated, reason = gate.check(0.95, weight_class="Catch Weight")
    assert gated
    assert "Catch Weight" in reason


def test_load_gate_from_config_dict():
    cfg = {"calibration": {
        "suppressed_confidence_bands": [[0.55, 0.70]],
        "flagged_weight_classes": ["Women's Flyweight"],
    }}
    gate = load_gate(cfg)
    assert gate.suppressed_confidence_bands == [(0.55, 0.70)]
    assert gate.flagged_weight_classes == ["Women's Flyweight"]


def test_load_gate_defaults_empty_when_calibration_section_missing():
    gate = load_gate({})
    assert gate.suppressed_confidence_bands == []
    assert gate.flagged_weight_classes == []
