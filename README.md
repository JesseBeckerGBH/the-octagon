# THE OCTAGON

A UFC/MMA fight-outcome probability engine built by JBAnalytics LLC, under
a D.O.E. (Directives / Orchestration / Execution) architecture:

- **Directives** — `config/settings.yaml`: weights, thresholds, which
  prophets are active. Nothing tunable is hard-coded in the model code.
- **Orchestration** — `orchestrator/council.py`: "The Council of Prophets"
  blends multiple independent models into one consensus probability.
- **Execution** — ingestion, feature engineering, inference, and the
  post-event validator that scores every prediction against reality.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Torch isn't in requirements.txt — install the CPU wheel separately:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

cp .env.example .env   # fill in real API keys if you have them

python scripts/create_schema.py
python -m ingestion.ufc_scraper --limit 30
python scripts/check_db.py

pytest
uvicorn inference_onnx.predict:app --reload
```

## Architecture

```
octagon/
├── config/settings.yaml       # weights, thresholds, active/stub status per prophet
├── ingestion/                 # scraper (real) + odds feed (stub, needs a provider)
├── features/engineer.py       # Polars differential + rolling-form features
├── models/                    # one file per prophet — see status table below
├── orchestrator/council.py    # weighted blend + optional LightGBM stacker
├── validator/                 # Brier score, log loss, CLV, writes validation_log
├── inference_onnx/            # FastAPI service (ONNX export is a planned optimization)
├── backtesting/                # walk-forward backtest (not built yet)
├── rust_core/                  # speed-critical inference core (not built yet)
├── deploy/                     # docker-compose, systemd unit, Proxmox + Cloudflare Tunnel guide
└── scripts/                    # schema creation, DB diagnostics
```

## The Council of Prophets — current status

| Prophet | File | Status | Notes |
|---|---|---|---|
| GBM (XGBoost) | `models/gbm_prophet.py` | **Active** | Fully implemented, trains and predicts for real. |
| Markov (Hot/Neutral/Cold) | `models/markov_prophet.py` | **Active** | Fully implemented from win-rate state transitions. |
| Bayesian (PyMC) | `models/bayesian_prophet.py` | Experimental | Real fit/predict, excluded from the default blend until validated on real history. |
| LSTM | `models/lstm_prophet.py` | Stub | Real architecture, `NotImplementedError` on fit — blocked on ingestion parsing round-by-round stats, not on model code. |
| Sharp Money (OU process) | `models/sharp_money_prophet.py` | Stub | The OU math (`estimate_ou_params`) is real and unit-tested; the prophet itself is blocked on `ingestion/odds_ingestion.py`, which needs a live odds provider wired up. |

This table is the honest state of the system, not a wishlist — a prophet
only moves to Active in `config/settings.yaml` once it's actually trained
and validated. Investor demos and subscriber-facing claims should only
ever cite what's Active.

## Why Brier score, not just win-rate accuracy

A model that always says "60%" for favorites can look fine on raw
accuracy while being badly miscalibrated — overconfident on underdogs,
underconfident on big favorites. Brier score penalizes that:

```
Brier = (1/N) * sum((p_i - o_i)^2)
```

Target: < 0.18 on held-out predictions. See `validator/post_event_validator.py`.

## Ensemble stacking

Beyond a simple weighted average, `orchestrator/council.py` can train a
LightGBM meta-learner on out-of-fold prophet outputs once there's enough
validated history (`config.validation.min_predictions_before_stacking`).
This typically adds several points of accuracy over averaging alone, while
preserving calibration — but it needs real validated predictions to train
on, so it isn't switched on by default.

## Where this actually runs

This service is designed to run on **Railway** (see the sibling
`ufc-predictions-saas` repo/service for the current production deployment
pattern: Stripe + Postgres + Redis + custom domain). **Cloudflare is DNS
only** — it's where `thebeastufc.com` is registered and where the domain's
DNS record points at Railway. It is not a second place this code runs.

`deploy/PROXMOX.md` documents a self-hosted alternative (Proxmox LXC +
Cloudflare Tunnel) for if/when you want to run this on your own hardware
instead of Railway — treat it as an alternative, not an addition.

## Roadmap

1. Extend `ingestion/ufc_scraper.py` to parse fight-detail pages (strikes,
   takedowns, control time) — this unblocks real differential features,
   the LSTM prophet, and style clustering.
2. Wire `ingestion/odds_ingestion.py` to a real provider (The Odds API or
   SportsData.io) — this unblocks the Sharp Money prophet and real ROI/CLV
   in the validator.
3. Walk-forward backtesting (`backtesting/engine.py`) once there's enough
   real history to backtest against.
4. ONNX export + `rust_core/` once there's a trained model worth
   optimizing.
5. Promote Bayesian → Active once validated; build LSTM and Sharp Money
   for real once steps 1–2 unblock them.

## Contributing (even solo)

Branch off `main`, open a PR describing what changed and why, let CI
(`.github/workflows/ci.yml`) run lint + tests, then merge. `main` is
protected — nothing lands without a green PR. This is what keeps the
history readable six months from now, for you or anyone else who joins.
