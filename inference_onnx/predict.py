"""
THE OCTAGON — inference API.

Named inference_onnx/ to match the project's target end-state (ONNX-
exported prophets for sub-15ms inference), but today it serves predictions
directly from the trained scikit-learn/XGBoost/LightGBM objects in memory —
ONNX export is a real optimization, not a prerequisite, and adding it
before there's a trained model to export would be premature.
"""

from contextlib import asynccontextmanager

import duckdb
import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from features.engineer import build_feature_table
from models.gbm_prophet import GBMProphet
from models.markov_prophet import MarkovProphet
from orchestrator.council import Council, load_config
from orchestrator.kelly_staking import KellyStaking

DB_PATH = "data/processed/octagon.duckdb"

council: Council | None = None
staking: KellyStaking | None = None


def _train_council() -> Council:
    con = duckdb.connect(DB_PATH)
    try:
        table = build_feature_table(con)
    finally:
        con.close()

    prophets = [GBMProphet(), MarkovProphet()]
    if not table.is_empty() and "label_a_won" in table.columns:
        labels = table["label_a_won"]
        for p in prophets:
            p.fit(table, labels)
    return Council(prophets, weights={"gbm": 0.7, "markov": 0.3})


@asynccontextmanager
async def lifespan(app: FastAPI):
    global council, staking
    council = _train_council()
    cfg = load_config()
    staking = KellyStaking(**cfg["staking"])
    yield


app = FastAPI(title="THE OCTAGON — Inference API", lifespan=lifespan)


class PredictRequest(BaseModel):
    fighter_a: str
    fighter_b: str
    # Optional: decimal odds on fighter_a, if you want a suggested paper
    # stake back. Omit it for a pure probability call.
    decimal_odds_a: float | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if council is None:
        raise HTTPException(status_code=503, detail="Council not initialized yet")

    # Real feature lookup wiring is a TODO — this endpoint currently returns
    # the council's neutral-input prediction so the deploy path (Docker,
    # healthcheck, Cloudflare/Railway routing) can be exercised end-to-end
    # before real per-fighter feature retrieval is built.
    placeholder = pl.DataFrame({"reach_diff": [0.0], "age_diff": [0.0], "slpm_diff": [0.0]})
    result = council.consensus(placeholder)[0]

    response = {
        "fighter_a": req.fighter_a,
        "fighter_b": req.fighter_b,
        "win_prob_a": result.blended_prob_a,
        "prophet_probs": result.prophet_probs,
        "dissent": result.dissent,
    }

    if req.decimal_odds_a is not None and staking is not None:
        stake, reason = staking.calculate_stake(result.blended_prob_a, req.decimal_odds_a)
        response["suggested_stake"] = stake
        response["stake_reason"] = reason

    return response
