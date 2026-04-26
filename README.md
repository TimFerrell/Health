# T1D Glucose Prediction — Data Pipeline

End-to-end pipeline that ingests Dexcom G7 CGM exports and Tandem Mobi
pump data, merges them onto a unified 5-minute timeline, and produces
a parquet file ready for feature engineering / XGBoost training.

This repo covers **data only** — no model training yet.

## Layout

```
app.py                  # Streamlit UI (Live / Pipeline / Timeline / Stats /
                        #               Model / Predict / What-if)
docs/
  ML_PRIMER.md          # Plain-English explainer for non-ML readers
src/
  ingest_dexcom.py      # Phase 1  — Clarity CSV -> dexcom_clean.parquet (bulk)
  ingest_tandem.py      # Phase 2  — tconnectsync -> tandem_clean.parquet
                        #            (synthetic fallback if API breaks)
  merge_pipeline.py     # Phase 3  — 5-min unified timeline
  validate_pipeline.py  # Phase 4  — EDA, gap detection, time-in-range
  features.py           # Phase 5  — lag/rolling/time features + 30m label
  train.py              # Phase 6  — walk-forward split + XGBoost
  predict.py            # Phase 7  — inference (latest + backtest)
  explain.py            # Phase 8  — SHAP global + local explanations
  ingest_dexcom_share.py# Phase 9  — pydexcom live CGM (~5 min latency)
  refresh.py            # Phase 10 — orchestrator (one-shot or --loop)
  anomaly.py            # Phase 11 — threshold-based alerts on live forecasts
  drift.py              # Phase 12 — rolling MAE/RMSE on logged predictions
  counterfactual.py     # Phase 13 — "what if" simulator over the model
  treatments.py         # Phase 14 — manual carb log + Tandem-derived carbs
                        #            unioned into a single carbs_g feature
scripts/
  _smoke_test.py        # Data-pipeline canary  (Phases 1-4)
  _smoke_ml.py          # ML-pipeline canary    (Phases 5-8)
  _smoke_live.py        # Live-pipeline canary  (Phases 9-13)
data/
  raw/                  # untracked: drop Clarity CSV exports here
  processed/            # untracked: parquet outputs + live logs
  models/               # untracked: saved model + metrics + SHAP
plots/                  # untracked: validation plots
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in TCONNECT_EMAIL / TCONNECT_PASSWORD in .env
```

`tconnectsync` is pulled directly from GitHub because the upstream
repo gets faster auth fixes than the PyPI release. If you don't have
Tandem credentials yet, the Phase 2 ingester will fall back to a
synthetic generator so you can still exercise Phases 3-4.

## Running the pipeline

Each phase is a standalone CLI — run them in order:

```bash
# Phase 1 — clean a Dexcom Clarity CSV
python -m src.ingest_dexcom data/raw/clarity_export.csv \
    -o data/processed/dexcom_clean.parquet

# Phase 2 — pull pump events for a date range
python -m src.ingest_tandem \
    --start 2026-03-25 --end 2026-04-24 \
    -o data/processed/tandem_clean.parquet

# Phase 3 — merge to the 5-min grid
python -m src.merge_pipeline \
    --dexcom data/processed/dexcom_clean.parquet \
    --tandem data/processed/tandem_clean.parquet \
    -o data/processed/unified_timeline.parquet

# Phase 4 — validate + plots
python -m src.validate_pipeline \
    --input data/processed/unified_timeline.parquet \
    --plot-dir plots/
```

Smoke test (uses synthetic Dexcom + Tandem data, no credentials needed):

```bash
python scripts/_smoke_test.py
```

## UI

A light Streamlit app drives the same modules from a browser:

```bash
streamlit run app.py
```

Sections:
- **Pipeline** — upload a Clarity CSV, pick a Tandem date range, run any
  phase with a button. Shows artifact freshness in the sidebar.
- **Timeline** — interactive plotly chart with bolus pins, CIQ suspend
  markers, and a glucose vs IOB overlay. Date-range filter at the top.
- **Stats** — TIR metrics, null counts, gap detection, pump activity
  summary, plus a CSV export of the unified timeline.

## Output schema (`unified_timeline.parquet`)

| column                 | type      | notes                                       |
|------------------------|-----------|---------------------------------------------|
| `timestamp`            | datetime  | UTC, 5-minute grid, no gaps                 |
| `glucose_mg_dl`        | float32   | 39 = "Low", 401 = "High", NaN = sensor gap  |
| `trend_arrow_encoded`  | float32   | -2 (double down) ... +2 (double up)         |
| `iob_units`            | float32   | Real if pump reports it, else modeled decay |
| `bolus_units`          | float32   | Sum of boluses delivered in the 5-min bin   |
| `basal_rate`           | float32   | Forward-filled from latest basal sample     |
| `is_suspended`         | bool      | True during pump suspend windows            |
| `control_iq_active`    | bool      | True during a CIQ algorithm action          |

## Data source notes

### Dexcom G7 (Clarity export)
- Export from [clarity.dexcom.com](https://clarity.dexcom.com) → Account →
  Data Export → CSV.
- Clarity files include EGV, calibration, alert, and device-event rows
  in one CSV; we keep only `Event Type == "EGV"`.
- Reportable range is 40–400 mg/dL. Out-of-range readings export as the
  literal strings `"Low"` / `"High"`; we encode them as **39 / 401** so
  the model sees a numeric, distinguishable signal.
- Trend arrow encoding (per project spec, clamped to -2..+2):
  doubleDown=-2, singleDown=-2, fortyFiveDown=-1, flat=0,
  fortyFiveUp=+1, singleUp=+2, doubleUp=+2.
- Sensor warmup gaps (~2 hours after a new sensor) and brief
  calibration-driven dropouts surface as gaps in Phase 4's report.
- Clarity timestamps are local-without-offset — we localize to UTC. If
  your CGM data crosses a DST boundary, set the system tz on the
  ingestion machine accordingly.

### Tandem Mobi via tconnectsync
- Tandem's t:connect API is undocumented. We use
  [tconnectsync](https://github.com/jwoglom/tconnectsync) which is
  reverse-engineered and occasionally breaks when Tandem changes auth
  or the Control-IQ endpoints.
- When ingestion fails, Phase 2 falls back to a synthetic generator
  (see `synthesize_tandem` in `src/ingest_tandem.py`) producing
  realistic basal/bolus/CIQ patterns so the rest of the pipeline stays
  testable.
- Four canonical event types are normalized: `basal`, `bolus`,
  `suspend`, `control_iq_action`.

### IOB modeling
When the pump's own IOB stream is missing, Phase 3 projects each bolus
forward via:

```
IOB(t) = dose * exp(-t / 240)   # t in minutes
```

This matches the project spec literally (4-hour exponential half-life
for rapid-acting analogs like Humalog/Novolog/Fiasp). It's a
back-of-envelope substitute for a proper one- or two-compartment PK
model — good enough for a baseline feature, not good enough for a
clinical decision system.

### Why Control-IQ events matter for prediction
- A CIQ **suspend** is the closed loop reacting to a *predicted* low.
  It's a strong leading indicator of an upcoming hypo and the rebound
  that often follows.
- A CIQ **basal increase** action means the algorithm "sees" a
  predicted high — useful as a corroborating signal for the model.

## Operational notes

- All timestamps are UTC throughout the pipeline. Convert to local for
  display only (e.g. in plotting or reporting).
- `.env` is git-ignored. **Never commit credentials.**
- `data/processed/` and `plots/` are git-ignored. Treat all generated
  artifacts as recreatable.
- Each phase is independently runnable so you can re-run a single
  stage without re-doing the whole pipeline.

## ML pipeline (Phases 5-8)

If you're not an ML person, **read [`docs/ML_PRIMER.md`](docs/ML_PRIMER.md) first** —
it explains the model, features, splits, metrics, and SHAP in plain English.

```bash
# Phase 5 — engineer features (lag, rolling, IOB context, circadian)
python -m src.features --input data/processed/unified_timeline.parquet \
    -o data/processed/features.parquet

# Phase 6 — train XGBoost with walk-forward split + early stopping
python -m src.train --features data/processed/features.parquet \
    --model-dir data/models

# Phase 7 — produce predictions (--latest for current 30-min forecast)
python -m src.predict --features data/processed/features.parquet \
    --model-dir data/models --latest

# Phase 8 — SHAP feature importance + local explanation
python -m src.explain --features data/processed/features.parquet \
    --model-dir data/models
```

### What "good" looks like
For a child's 30-minute glucose forecast on real CGM + pump data, a
**test-set MAE of 12-18 mg/dL** is competitive with published baselines.
Below ~10 is suspicious (probably a leak — see ML_PRIMER §"naive splits
cheat"). Above 25 means the model is struggling — check the validation
parquet for gaps, sensor noise, or unlogged meals.

### What this model **cannot** do
- Predict spikes from un-bolused meals.
- React to exercise, illness, stress, or hormonal shifts.
- Substitute for clinical judgment. **Use for insight only — never dose
  insulin from these predictions.**

## ML smoke test
```bash
python scripts/_smoke_ml.py    # synthesizes 60 days, runs phases 5-8 end-to-end
```

## Live pipeline (Phases 9-13)

Backfill (Phases 1-4) gave us the historical timeline; the **live pipeline**
keeps it fresh and turns predictions into actionable alerts.

| Phase | Module | What it does |
|-------|--------|--------------|
| 9  | `ingest_dexcom_share.py` | Polls Dexcom Share for the latest CGM (~5 min latency). Falls back to synthetic. |
| 10 | `refresh.py`             | One refresh cycle: pull → re-merge → re-feature → predict → log → alert. `--loop` runs forever. |
| 11 | `anomaly.py`             | Hand-tuned thresholds on the latest forecast: predicted low/high/drop/rise/stale-data. |
| 12 | `drift.py`               | Rolling MAE/RMSE on logged predictions vs the actual glucose 30 min later. Flags "retrain needed" when live MAE exceeds 1.5× the trained-test MAE. |
| 13 | `counterfactual.py`      | "What if" simulator — re-runs the model with hypothetical bolus/suspend actions and reports the predicted delta. |

### Run a single refresh

```bash
python -m src.refresh -v
```

Output:
```
Merged 12 new readings (source=pydexcom); dexcom_clean now has 8640 rows up to 2026-04-25 16:25:00+00:00
as_of=2026-04-25 16:25:00+00:00  current=142  predict_30m=156  Δ=+14  alerts: info:predicted_rise
```

### Run continuously (Docker scheduler sidecar handles this)

```bash
python -m src.refresh --loop --interval 300
```

### What lands on disk per cycle
- `data/processed/dexcom_clean.parquet`  — appended with new readings, deduped on timestamp.
- `data/processed/unified_timeline.parquet`  — re-merged.
- `data/processed/features.parquet`  — re-engineered.
- `data/processed/predictions_log.parquet`  — append-only `(predicted_at, target_timestamp, current_glucose, prediction_30m)`. This is what `drift.py` reads to compute live accuracy.
- `data/processed/alerts_log.parquet`  — last 1,000 alerts.

### Live tab in the UI
- Current glucose + 30-min forecast tile with delta.
- Active alerts banner (color-coded by severity).
- Last-24h chart of predictions overlaid with realized glucose.
- Drift summary: live 24h/7d MAE vs trained MAE, with a 1.5× threshold line drawn on the rolling-MAE chart.
- Auto-refresh selector (1/5/10 min) — uses an HTML meta-refresh so it works without WebSocket sessions.

### What-if tab
Pick a set of hypothetical bolus doses and basal-suspend durations; the
model re-predicts the 30-min glucose for each scenario and you see a
side-by-side bar chart + delta-vs-baseline table. This is the
counterfactual feature — see `docs/ML_PRIMER.md` and the
`counterfactual.py` docstring for what it can and can't tell you.

### Live smoke test
```bash
python scripts/_smoke_live.py  # bootstraps with ML smoke, runs 9-14 end-to-end
```

## Carbs (Phase 14)

The model's biggest blind spot was carbs — particularly **carbs without a
bolus**, like a juice box for a low. We close it from two sides:

### Pump-derived carbs (free, automatic)
`ingest_tandem.py` now extracts the carb amount the user entered into
the bolus calculator (`carbsRequest` / `carbsAmount`), along with the
food/correction split and BG-at-bolus. These land as new optional
columns on bolus rows in `tandem_clean.parquet`.

### Manual carb log (`treatments.parquet`)
For carbs the pump never sees — low treatments, unbolused snacks — the
**Log carbs** tab in the UI logs them directly:

- Three quick-tap presets (4 g gel, 8 g juice, 15 g standard).
- A free-form form for meals/snacks/corrections with optional notes.
- Append-only parquet at `data/processed/treatments.parquet`.

Schema mirrors a Nightscout `treatments` document so a future Phase
can swap in a Nightscout client without changing the downstream
pipeline.

### Unified `carbs_g` feature
`merge_pipeline.py` unions both sources, sums by 5-min bin, and
emits a `carbs_g` column on the unified timeline. `features.py` then
derives:

- `carbs_sum_15m`, `carbs_sum_30m`, `carbs_sum_60m`, `carbs_sum_120m`
- `minutes_since_carbs` (capped at 8 h)

After your first refresh that includes carb data, **retrain the model**
so it actually learns from carbs. Until then the new features exist but
the model's `feature_columns.json` doesn't reference them, so they're
effectively no-ops (the What-if tab warns about this).

### What-if with carbs
The What-if tab now includes a third action axis — grams of carbs to
eat now (4 / 8 / 15 / 20 / 30 / 45 / 60). After retraining you can
ask "what if my kid has 8g of juice right now?" and see the model's
predicted glucose effect at the 30-min horizon.

## Deploying on Unraid (Docker)

The repo ships a `Dockerfile`, a `docker-compose.yml`, and an Unraid
template. Two paths:

### Option A — Compose Manager plugin (recommended)

1. **Community Applications → Compose Manager** (install the plugin
   if you haven't).
2. Add a new stack, name it `t1d-pipeline`, paste the contents of
   `docker-compose.yml`. Adjust the host bind path if you don't use
   `/mnt/user/appdata/t1d-pipeline`.
3. Stack → **Up**. First build takes ~3-5 min while it pulls the
   wheels for pandas / xgboost / streamlit; subsequent rebuilds are
   cached.
4. Open `http://<unraid-ip>:8501`.
5. Drop your Clarity CSV in the file uploader on the Pipeline tab,
   or `cp` it into `/mnt/user/appdata/t1d-pipeline/raw/` first.

### Option B — Manual container

1. SSH into Unraid, clone the repo somewhere persistent:
   ```bash
   mkdir -p /mnt/user/appdata/t1d-pipeline-src
   cd /mnt/user/appdata/t1d-pipeline-src
   git clone <repo-url> .
   docker build -t t1d-pipeline:latest .
   ```
2. **Docker** tab → **Add Container**. Paste the contents of
   `unraid-template.xml` into the *Template* field, or import it via
   URL.
3. Bind path: `/mnt/user/appdata/t1d-pipeline` → `/data`.
4. Apply, wait for green status, click **WebUI**.

### Persistent state

The container writes everything to `/data` (a single bind mount):

```
/data/raw/         # uploaded Clarity CSVs
/data/processed/   # parquets (Phases 1-4 + features)
/data/models/      # trained XGBoost model + metrics + SHAP
/data/plots/       # matplotlib outputs (if the CLI is run inside)
```

Survives container rebuilds, image updates, and Unraid reboots.

### Security notes

- **Streamlit has no built-in authentication.** Don't expose port 8501
  to the public internet. Either keep access on the LAN/VPN, or put it
  behind your existing reverse proxy (Nginx Proxy Manager + Authelia /
  Tailscale Funnel / Cloudflare Tunnel access policy).
- Tandem credentials should be set via the container's environment
  variables (Unraid's UI handles this), not committed to the repo.
- The `.env` file is git-ignored and is not baked into the image —
  `tconnectsync` reads creds from the container env at runtime.
- All traffic between the container and Dexcom Clarity is one-way
  (you upload a CSV); the only outbound network call is the optional
  `tconnectsync` request to Tandem.

### Updating

```bash
cd /mnt/user/appdata/t1d-pipeline-src
git pull
docker compose up -d --build         # Compose route
# or
docker build -t t1d-pipeline:latest . # manual route, then restart container
```

### CPU / memory

XGBoost training + SHAP on ~60 days of 5-min data uses < 1 GB RAM and
finishes in 10-30 seconds on a modest CPU. No GPU needed.
