# T1D Glucose Prediction — Data Pipeline

End-to-end pipeline that ingests Dexcom G7 CGM exports and Tandem Mobi
pump data, merges them onto a unified 5-minute timeline, and produces
a parquet file ready for feature engineering / XGBoost training.

This repo covers **data only** — no model training yet.

## Layout

```
app.py                 # Streamlit UI (Pipeline / Timeline / Stats / Model / Predict)
docs/
  ML_PRIMER.md         # Plain-English explainer for non-ML readers
src/
  ingest_dexcom.py     # Phase 1 — Clarity CSV -> dexcom_clean.parquet
  ingest_tandem.py     # Phase 2 — tconnectsync -> tandem_clean.parquet
                       #           (synthetic fallback if API breaks)
  merge_pipeline.py    # Phase 3 — 5-min unified timeline
  validate_pipeline.py # Phase 4 — EDA, gap detection, time-in-range
  features.py          # Phase 5 — lag/rolling/time features + 30m label
  train.py             # Phase 6 — walk-forward split + XGBoost
  predict.py           # Phase 7 — inference (latest + backtest)
  explain.py           # Phase 8 — SHAP global + local explanations
scripts/
  _smoke_test.py       # Data-pipeline canary (Phases 1-4)
  _smoke_ml.py         # ML-pipeline canary (Phases 5-8)
data/
  raw/                 # untracked: drop Clarity CSV exports here
  processed/           # untracked: parquet outputs land here
  models/              # untracked: saved model + metrics + SHAP
plots/                 # untracked: validation plots
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
