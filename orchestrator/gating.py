"""
Confidence/coverage gate — the concrete fix for the two biggest findings
in the analysis of the live thebeastufc.com model:

  - Confidence-bucket ROI table showed 55-60% confidence picks losing
    -11.6% ROI and 60-65% picks losing -9.1%, while 70%+ picks made
    +10.4% ROI. The model wasn't wrong to be uncertain there — it was
    wrong to publish a bet recommendation on that uncertainty.
  - Coverage drift: Women's Flyweight ran -2.1% ROI and Catch Weight ran
    -19.6% ROI, both well below the model's overall performance — likely
    a thin-training-data problem in those classes rather than a modeling
    one, but the fix is the same either way: don't publish there until
    it's proven out.

This is a *presentation* gate, not a model change: the Council still
computes its best probability estimate for every fight (gated predictions
are still logged to validation_log so the validator can keep tracking
whether the gate itself is well-calibrated), but CouncilResult.gated=True
tells the serving layer (inference_onnx/predict.py) and any subscriber-
facing surface not to present it as an actionable pick.
"""

from dataclasses import dataclass, field


@dataclass
class Gate:
    # Each band is (low, high), inclusive-exclusive on confidence =
    # max(prob_a, 1 - prob_a). Empty by default — gating is opt-in via
    # config/settings.yaml, never a silent behavior change.
    suppressed_confidence_bands: list[tuple[float, float]] = field(default_factory=list)
    flagged_weight_classes: list[str] = field(default_factory=list)

    def check(self, prob_a: float, weight_class: str | None = None) -> tuple[bool, str | None]:
        confidence = max(prob_a, 1 - prob_a)
        reasons = []

        for lo, hi in self.suppressed_confidence_bands:
            if lo <= confidence < hi:
                reasons.append(
                    f"confidence {confidence:.1%} is in the suppressed {lo:.0%}-{hi:.0%} "
                    f"band (historically negative ROI in production)"
                )
                break

        if weight_class and weight_class in self.flagged_weight_classes:
            reasons.append(f"{weight_class} is flagged for coverage drift (thin/unreliable history)")

        if reasons:
            return True, "; ".join(reasons)
        return False, None


def load_gate(cfg: dict) -> Gate:
    cal = cfg.get("calibration", {})
    bands = [tuple(b) for b in cal.get("suppressed_confidence_bands", [])]
    flagged = list(cal.get("flagged_weight_classes", []))
    return Gate(suppressed_confidence_bands=bands, flagged_weight_classes=flagged)
