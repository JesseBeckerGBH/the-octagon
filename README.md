# THE OCTAGON

A UFC/MMA fight-outcome probability engine built by JBAnalytics LLC, under
a D.O.E. (Directives / Orchestration / Execution) architecture:

- **Directives** — `config/settings.yaml`: weights, thresholds, which
  prophets are active. Nothing tunable is hard-coded in the model code.
- **Orchestration** — `orchestrator/council.py`: "The Council of Prophets"
  blends multiple independent models into one consensus probability.
- **Execution** — ingestion, feature engineering, inference, and the
  post-event validator that scores every prediction against reality.

## What changed: the merge with the live production engine

This repo started as a scaffold built from a Grok planning conversation
("THE OCTAGON" transcript) plus the Gemini implementation brief that came
with it — real Directives/Orchestration/Execution structure, but the
Council's actual predictive power came from a single GBM prophet trained
on three placeholder differential columns (`reach_diff`/`age_diff`/
`slpm_diff`, always 0.0), because ingestion never captured real per-fight
stats.

Since then, the actual production engine behind the live
`thebeastufc.com` — the repo `JesseBeckerGBH/ufc-predictions-saas`, and
specifically its `engine/` subfolder — was located and read in full. That
engine is real, live, and walk-forward-validated: a 4-model ensemble
(XGBoost, LightGBM, Logistic Regression, Random Forest) with 5-fold
out-of-fold isotonic calibration and a logistic-regression stacking
meta-learner, trained on ~70 leak-safe features (career/rolling stat
differentials, a fitted 4-state Markov form chain, KMeans style-cluster
matchups, method-of-victory labels) built from HuggingFace + Kaggle UFC
datasets.

**This merge ports that proven model + validation layer into Octagon's
D.O.E. architecture** — the model, feature engineering, and backtesting
engine, not the billing/auth/frontend around it (see "Merge scope"
below for why). Concretely:

- `ingestion/hf_pipeline.py` — the real HuggingFace/Kaggle data pipeline,
  replacing/supplementing the fragile `ufcstats.com` scraper as the
  source of per-fight strike/takedown/control-time stats.
- `features/leak_safe_features.py` — the real leak-safe feature
  engineering (career expanding means, recent-3 rolling form, win
  streaks, differentials), ported from `engine/ingestion/ingest.py`.
- `features/markov_form.py`, `features/ou_process.py`,
  `features/style_cluster.py`, `models/method_classifier.py` — the four
  feature/model modules ported from `engine/models/`, each upgrading an
  Octagon roadmap item from sketch to proven implementation (see the
  Council table below).
- `models/lightgbm_prophet.py`, `logistic_prophet.py`,
  `randomforest_prophet.py` — three new base learners, same
  hyperparameters as production, joining the existing `gbm_prophet.py`.
- `orchestrator/council.py::fit_calibrated_stacker()` — the real
  ensemble recipe (OOF isotonic calibration + logistic stacking),
  replacing the old plain-weighted-average-or-raw-LightGBM-stacker as
  the primary blend mode once there's enough validated history.
- `orchestrator/gating.py` — new. See "The calibration gate" below.
- `backtesting/engine.py` — the real walk-forward backtest, replacing
  the `NotImplementedError` stub.

### Merge scope: model + validation only

The live `ufc-predictions-saas` repo also has a Next.js frontend, a
FastAPI backend with JWT/Google OAuth, and Stripe + Whop billing serving
real paying subscribers. None of that was touched or ported here, on
purpose: those systems are live and working, this merge's job was fixing
the thing that was actually flagged as broken (prediction calibration),
and neither this sandbox nor this repo has the live Postgres data,
Stripe/Whop keys, or subscriber records needed to safely rebuild or test
against them. The plan is to point `thebeastufc.com` at a *new* Railway
service running this engine once it's verified, not to modify the
current live service in place.

### An honest gap worth chasing down

The production engine's own training script reports out-of-fold Brier
around 0.21 (5-fold cross-validation on historical data). But a separate
analysis of the *live* model's actual predictions measured its rolling
Brier at 0.373 — a much worse number — and found it losing money on
55-70%-confidence picks specifically, plus negative-ROI coverage drift in
Women's Flyweight and Catch Weight. That gap (good in cross-validation,
worse in live serving) usually means one of: the live feature pipeline
has drifted from the training pipeline, the model is seeing a
harder/different fight mix live than in its training history, or a
fallback heuristic (not the real ensemble) is quietly serving some
fraction of predictions. `backtesting/engine.py`'s confidence-bucket and
weight-class breakdowns are the tool for finding out which, against
Octagon's own ingested data.

## The calibration gate

`orchestrator/gating.py` is the concrete fix for the confidence-bucket
and coverage-drift findings above: `config/settings.yaml`'s
`calibration.suppressed_confidence_bands` and `.flagged_weight_classes`
let you mark a confidence range or weight class as "don't present this as
an actionable pick" without changing the model itself — the Council still
computes and logs its best estimate (so the validator can keep tracking
whether the gate is well-calibrated), it just flags `CouncilResult.gated`
so `/predict` and any subscriber-facing surface know not to act on it.

**Both are empty by default.** The temptation is to copy
thebeastufc.com's exact bad bands (55-70% confidence, Women's Flyweight,
Catch Weight) straight into this repo's config — don't, yet. Those bands
were measured against that model's specific feature set and training
history; run `backtesting/engine.py` against Octagon's own ingested data
first and gate on what you actually find here.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Torch isn't in requirements.txt — install the CPU wheel separately:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

cp .env.example .env   # fill in real API keys if you have them

python scripts/create_schema.py

# Real data path (recommended): HuggingFace + optional Kaggle ingestion,
# then the full leak-safe feature set.
python -m ingestion.hf_pipeline
python -m features.leak_safe_features

# Lighter-weight fallback: the ufcstats.com scraper alone (fewer features,
# no real per-fight stats — see features/engineer.py's fallback path).
python -m ingestion.ufc_scraper --limit 30

python scripts/check_db.py

pytest
python -m backtesting.engine          # walk-forward backtest report
uvicorn inference_onnx.predict:app --reload
```

## Architecture

```
octagon/
├── config/settings.yaml            # weights, thresholds, calibration gate, active/stub status
├── ingestion/
│   ├── hf_pipeline.py               # real: HuggingFace + Kaggle historical merge
│   ├── ufc_scraper.py               # lighter fallback: ufcstats.com scraper
│   └── odds_ingestion.py            # stub — needs a live odds provider
├── features/
│   ├── engineer.py                  # entry point: real leak-safe table, or placeholder fallback
│   ├── leak_safe_features.py        # real feature engineering (ported from production)
│   ├── markov_form.py               # fitted 4-state Markov form chain
│   ├── ou_process.py                # OU process: CLV estimate, steam detection
│   └── style_cluster.py             # KMeans style-archetype matchup features
├── models/                          # one file per prophet/model — see status table below
├── orchestrator/
│   ├── council.py                   # 3 blend modes: calibrated stacking > legacy stacker > weighted avg
│   ├── gating.py                    # confidence/coverage presentation gate
│   └── kelly_staking.py             # fractional Kelly bet sizing (ported from The Beast)
├── validator/                       # Brier score, log loss, CLV, writes validation_log
├── inference_onnx/                  # FastAPI service (ONNX export is a planned optimization)
├── backtesting/engine.py            # real walk-forward backtest (Brier/ROI/AUC by confidence & weight class)
├── rust_core/                       # speed-critical inference core (not built yet)
├── deploy/                          # docker-compose, systemd unit, Proxmox + Cloudflare Tunnel guide
└── scripts/                         # schema creation, DB diagnostics
```

## The Council of Prophets — current status

| Prophet | File | Status | Notes |
|---|---|---|---|
| GBM (XGBoost) | `models/gbm_prophet.py` | **Active** | Hyperparameters ported from the production ensemble. |
| LightGBM | `models/lightgbm_prophet.py` | **Active** | New — one of the 4 production base learners. |
| Logistic | `models/logistic_prophet.py` | **Active** | New — the calibrated-baseline base learner. |
| Random Forest | `models/randomforest_prophet.py` | **Active** | New — the 4th production base learner. |
| Markov (fitted form chain) | `models/markov_prophet.py` | **Active** | Upgraded from a hardcoded hot/neutral/cold probability table to a real fitted 4-state chain (`features/markov_form.py`) + a logistic head trained on real labels. |
| Method of Victory | `models/method_classifier.py` | **New, auxiliary** | Predicts P(KO/TKO)/P(Submission)/P(Decision) alongside the win-probability vote — attached to `CouncilResult.method_probs`, doesn't affect `blended_prob_a`. |
| Bayesian (PyMC) | `models/bayesian_prophet.py` | Experimental | Real fit/predict, excluded from the default blend until validated on real history. |
| LSTM | `models/lstm_prophet.py` | Stub | Real architecture, `NotImplementedError` on fit — blocked on ingestion parsing round-by-round stats, not on model code. |
| Sharp Money (OU process) | `models/sharp_money_prophet.py` | Stub | `features/ou_process.py` has the full production-grade OU math (CLV estimate, steam detection) ready to go; the prophet itself is still blocked on `ingestion/odds_ingestion.py`, which needs a live odds provider wired up. |

This table is the honest state of the system, not a wishlist — a prophet
only moves to Active in `config/settings.yaml` once it's actually trained
and validated. Investor demos and subscriber-facing claims should only
ever cite what's Active, and should exclude anything the calibration gate
flags as `gated=True`.

## Why Brier score, not just win-rate accuracy

A model that always says "60%" for favorites can look fine on raw
accuracy while being badly miscalibrated — overconfident on underdogs,
underconfident on big favorites. Brier score penalizes that:

```
Brier = (1/N) * sum((p_i - o_i)^2)
```

Target: < 0.18 on held-out predictions. See `validator/post_event_validator.py`
for the live-tracking version and `backtesting/engine.py` for the
walk-forward version (which is what actually caught the live-vs-training
Brier gap described above).

## Ensemble stacking: three modes

`orchestrator/council.py::consensus()` tries these in order:

1. **Calibrated stacking** (`fit_calibrated_stacker()`) — the production
   recipe: 5-fold out-of-fold predictions per prophet, isotonic
   calibration per prophet, then a logistic-regression meta-learner on
   the calibrated OOF matrix. Kicks in automatically once there's enough
   real history (`validation.min_predictions_before_stacking`) — this is
   the mode that actually produces the ~0.21 OOF Brier the production
   engine reports, and it's why per-prophet calibration matters: stacking
   raw, uncalibrated outputs (mode 3) is exactly the kind of thing that
   produces a good-looking headline accuracy with bad calibration
   underneath.
2. **Legacy stacker** (`fit_stacker()`) — the original design: a single
   LightGBM meta-learner on raw prophet outputs, no per-fold refitting.
   Lighter-weight, but superseded by mode 1 for anything that ships.
3. **Weighted average** (default, no fitting needed) — what a fresh
   install runs on before there's enough history for either stacker.

## What was ported from The Beast (tennis Council of Prophets)

Jesse uploaded `council_of_prophets.zip` believing it might be the UFC
"Prophet" engine. It isn't — there's no UFC-specific code in it (no
strikes, takedowns, or fight outcomes anywhere). It's **The Beast's**
tennis prediction stack: a `BayesianProphet` fit on ATP serve/return
stats, a `NeuralProphet` (LSTM, trained on placeholder random data — same
maturity as Octagon's own LSTM stub), a `MarkovSimulator` that computes
tennis game/set/tiebreak probabilities analytically, an `OracleEnsemble`
meta-learner, a `TennisDataLoader` pulling Jeff Sackmann's public ATP/WTA
CSVs, plus tooling for other sports (DataGolf client, a darts computer-
vision scoreboard reader) and personal-assistant tooling (an arXiv paper
scout, a podcast transcriber/sentiment scorer, a local-LLM "Chairman"
report writer) that rode along in the same folder. A "SBME's for Manus"
folder found later in Drive turned out to be the same story: Jesse's
table-tennis/darts/golf betting stacks, not a second UFC engine — "SBME"
(Sports Betting Model Engine) is a generic term he uses across sports,
not a project name.

What actually moved over, because the math is sport-agnostic:

- **`orchestrator/kelly_staking.py`** — the fractional-Kelly bet sizer.
  Octagon didn't have one at all; this is a straight, valuable port, now
  wired into `/predict`'s optional `decimal_odds_a` field.

What's a good idea but not ported yet:

- **The "Chairman" LLM report** (`ollama_agent.py`) — turns the council's
  numbers into a plain-English betting writeup. No UFC equivalent exists;
  this would be a genuinely new subscriber-facing feature, not just a
  port.
- **Podcast sentiment intel** (`podcast_feed_handler.py` +
  `podcast_listener.py`) — transcribes podcast RSS episodes with Whisper
  and scores sentiment into a "Hype Score." For UFC this could ingest
  fight-week podcasts for injury/camp chatter — a real qualitative signal
  the sharp-money/odds side doesn't capture. Also net-new, not a port.

What's not reusable: the Markov tennis math (games/sets/tiebreaks don't
map onto UFC rounds — superseded anyway by the production Markov form
chain ported below), `TennisDataLoader` (tennis-specific source), and the
darts/golf/tech-scout tooling (different projects that happened to share
the folder).

One more thing: the zip also contained a `Saved Pictures` folder of
unrelated personal photos, which came along because the whole parent
folder got zipped rather than just the project. Nothing from it went into
this repo — flagging it so it doesn't end up committed anywhere by accident.

## Where this actually runs

This service is designed to run on **Railway**. `thebeastufc.com` is
currently served by the live `JesseBeckerGBH/ufc-predictions-saas` Railway
service; the plan is to stand up a *new* Railway service running this
repo, verify it end-to-end, then cut the domain over — not to replace the
live service in place. **Cloudflare is DNS only** — it's where
`thebeastufc.com` is registered and where the domain's DNS record points
at Railway. It is not a second place any of this code runs.

`deploy/PROXMOX.md` documents a self-hosted alternative (Proxmox LXC +
Cloudflare Tunnel) for if/when you want to run this on your own hardware
instead of Railway — treat it as an alternative, not an addition.

## Roadmap

1. Run `backtesting/engine.py` against Octagon's own ingested data and
   compare its confidence-bucket / weight-class breakdown to the live
   model's — that's what decides what actually goes into
   `calibration.suppressed_confidence_bands` / `.flagged_weight_classes`.
2. Wire `ingestion/odds_ingestion.py` to a real provider (The Odds API or
   SportsData.io) — this unblocks the Sharp Money prophet (the OU math is
   already ready in `features/ou_process.py`) and real ROI/CLV in the
   validator.
3. Extend `ingestion/hf_pipeline.py`/`ufc_scraper.py` to capture
   round-by-round sequence data — this unblocks the LSTM prophet, which
   is otherwise architecturally ready.
4. ONNX export + `rust_core/` once there's a trained model worth
   optimizing.
5. Promote Bayesian → Active once validated against real history; revisit
   whether it's worth rebuilding around hierarchical shrinkage (the
   tennis `BayesianProphet`'s approach) for fighters with only 1-2 UFC
   fights.
6. Decide the production repo consolidation: point `thebeastufc.com` at a
   new Railway service running this repo once (1) is satisfied, per the
   analysis's recommended architecture (SaaS = distribution/billing,
   this engine = model layer, DuckDB = observability).

## Contributing (even solo)

Branch off `main`, open a PR describing what changed and why, let CI
(`.github/workflows/ci.yml`) run lint + tests, then merge. `main` is
protected — nothing lands without a green PR. This is what keeps the
history readable six months from now, for you or anyone else who joins.
