"""
THE OCTAGON — inference API.

Named inference_onnx/ to match the project's target end-state (ONNX-
exported prophets for sub-15ms inference), but today it serves predictions
directly from the trained scikit-learn/XGBoost/LightGBM objects in memory —
ONNX export is a real optimization, not a prerequisite, and adding it
before there's a trained model to export would be premature.

Startup now trains the full 4-prophet ensemble (GBM/LightGBM/Logistic/
RandomForest, the recipe ported from the live production engine) with
calibrated stacking once there's enough validated history, and builds a
per-fighter feature lookup so /predict answers a real Fighter A vs Fighter
B question instead of always scoring a neutral zero-vector — see
features/leak_safe_features.py::build_fighter_lookup/build_prediction_row
(ported from production's prediction_engine.py::_ensemble_prob_a). If
fight_stats_raw hasn't been populated yet (a fresh clone that only ran
ingestion/ufc_scraper.py), everything here degrades gracefully back to the
original neutral-placeholder behavior rather than erroring.
"""

from contextlib import asynccontextmanager

import duckdb
import polars as pl
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from features.engineer import build_feature_table, get_feature_cols
from models.gbm_prophet import GBMProphet
from models.lightgbm_prophet import LightGBMProphet
from models.logistic_prophet import LogisticProphet
from models.markov_prophet import MarkovProphet
from models.method_classifier import MethodClassifier
from models.randomforest_prophet import RandomForestProphet
from orchestrator.council import Council, load_config
from orchestrator.kelly_staking import KellyStaking

DB_PATH = "data/processed/octagon.duckdb"

council: Council | None = None
staking: KellyStaking | None = None
fighter_lookup: dict[str, dict] = {}
medians: dict[str, float] = {}
feature_cols: list[str] = []
using_real_data: bool = False


def _train_council(cfg: dict) -> Council:
    global fighter_lookup, medians, feature_cols, using_real_data

    import os
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = duckdb.connect(DB_PATH)
    try:
        table = build_feature_table(con)
    finally:
        con.close()

    prophets = [GBMProphet(), LightGBMProphet(), LogisticProphet(), RandomForestProphet(), MarkovProphet()]

    if table.is_empty() or "label_a_won" not in table.columns:
        return Council(prophets, weights={"gbm": 1.0})

    labels = table["label_a_won"]
    feature_cols = get_feature_cols(table)
    using_real_data = "fa_career_won" in table.columns  # true once hf_pipeline has run

    from orchestrator.gating import load_gate

    min_for_stacking = cfg.get("validation", {}).get("min_predictions_before_stacking", 50)
    council_obj = Council(prophets, weights=cfg["council"]["weights"], gate=load_gate(cfg))

    if len(table) >= min_for_stacking and using_real_data:
        # Enough real history: use the production-proven calibrated
        # stacking recipe rather than the plain weighted average.
        metrics = council_obj.fit_calibrated_stacker(table, labels)
        print(f"[octagon] Calibrated stacker trained: {metrics}")
    else:
        for p in prophets:
            try:
                p.fit(table, labels)
            except NotImplementedError:
                pass  # stub/gated prophets (lstm, sharp_money) sit out until unblocked

    if using_real_data:

        # Rebuild the pandas view once for the lookup/medians helpers,
        # which are pandas-native (they mirror production's
        # prediction_engine.py exactly) — cheap relative to training.
        pdf = table.to_pandas()
        from features.leak_safe_features import build_fighter_lookup, compute_medians
        fighter_lookup = build_fighter_lookup(pdf.rename(columns={"label_a_won": "winner_is_a"}))
        medians = compute_medians(pdf, feature_cols)

        method_clf = MethodClassifier()
        if "method_class_idx" in pdf.columns:
            X = table.select(feature_cols).fill_null(0).to_numpy()
            method_clf.fit(X, pdf["method_class_idx"].values)
            council_obj.set_method_classifier(method_clf)

    return council_obj


@asynccontextmanager
async def lifespan(app: FastAPI):
    global council, staking
    cfg = load_config()
    council = _train_council(cfg)
    staking = KellyStaking(**cfg["staking"])
    yield


app = FastAPI(title="THE OCTAGON — Inference API", lifespan=lifespan)


class PredictRequest(BaseModel):
    fighter_a: str
    fighter_b: str
    # Optional: decimal odds on fighter_a, if you want a suggested paper
    # stake back. Omit it for a pure probability call.
    decimal_odds_a: float | None = None


LANDING_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>THE OCTAGON — Inference API</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font: 15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;
         background:#0b0d10; color:#e6e8eb; }
  main { max-width: 720px; margin: 8vh auto; padding: 0 24px; }
  h1 { font-size: 28px; letter-spacing: .5px; margin: 0 0 4px; }
  .tag { color:#8b95a1; font-size: 13px; margin-bottom: 32px; }
  .pill { display:inline-block; padding: 2px 8px; border-radius: 999px;
          background:#123b23; color:#7ee2a8; font-size:12px; margin-left:8px;
          vertical-align: middle; }
  h2 { font-size: 15px; text-transform: uppercase; letter-spacing: 1px;
       color:#8b95a1; margin: 28px 0 10px; }
  a { color:#7cb7ff; text-decoration: none; }
  a:hover { text-decoration: underline; }
  code, pre { font-family: ui-monospace,SFMono-Regular,Menlo,monospace; font-size: 13px; }
  pre { background:#131720; border:1px solid #1e2530; border-radius: 8px;
        padding: 14px 16px; overflow-x:auto; }
  ul { padding-left: 18px; }
  li { margin: 6px 0; }
  footer { margin-top: 40px; color:#5c6673; font-size: 12px; }
</style>
</head>
<body>
<main>
  <h1>THE OCTAGON <span class="pill">LIVE</span></h1>
  <div class="tag">UFC fight-prediction ensemble by <strong>JBAnalytics</strong></div>

  <h2>Endpoints</h2>
  <ul>
    <li><a href="/docs">/docs</a> — interactive Swagger UI (try predictions in your browser)</li>
    <li><a href="/health">/health</a> — liveness check</li>
    <li><code>POST /predict</code> — fight prediction (see docs)</li>
  </ul>

  <h2>Example</h2>
<pre>curl -X POST https://thebeastufc-engine-production.up.railway.app/predict \\
  -H "Content-Type: application/json" \\
  -d '{"fighter_a":"Islam Makhachev","fighter_b":"Charles Oliveira","decimal_odds_a":1.65}'</pre>

  <footer>5-prophet calibrated ensemble &middot; GBM &middot; LightGBM &middot; Logistic &middot; RandomForest &middot; Markov</footer>
</main>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def root():
    return LANDING_HTML


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if council is None:
        raise HTTPException(status_code=503, detail="Council not initialized yet")

    if using_real_data and feature_cols:
        from features.leak_safe_features import build_prediction_row

        row = build_prediction_row(feature_cols, req.fighter_a, req.fighter_b, fighter_lookup, medians)
        features = pl.from_pandas(row)
    else:
        # No real ingestion has run yet — same honest neutral placeholder
        # behavior as before, so the deploy path (Docker, healthcheck,
        # Cloudflare/Railway routing) is still exercisable end-to-end.
        features = pl.DataFrame({"reach_diff": [0.0], "age_diff": [0.0], "slpm_diff": [0.0]})

    result = council.consensus(features)[0]

    response = {
        "fighter_a": req.fighter_a,
        "fighter_b": req.fighter_b,
        "win_prob_a": result.blended_prob_a,
        "prophet_probs": result.prophet_probs,
        "dissent": result.dissent,
        "method_probs": result.method_probs,
        "gated": result.gated,
        "gate_reason": result.gate_reason,
        "known_fighters": bool(fighter_lookup.get(req.fighter_a) and fighter_lookup.get(req.fighter_b)),
    }

    if req.decimal_odds_a is not None and staking is not None and not result.gated:
        stake, reason = staking.calculate_stake(result.blended_prob_a, req.decimal_odds_a)
        response["suggested_stake"] = stake
        response["stake_reason"] = reason
    elif result.gated:
        response["suggested_stake"] = 0.0
        response["stake_reason"] = f"Gated: {result.gate_reason}"

    return response
