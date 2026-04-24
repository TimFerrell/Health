# ML Primer — for parents, clinicians, and other non-ML readers

This document explains what the prediction pipeline is doing, in plain
English, so you can understand the model's outputs and limitations
without a stats background.

If you've never written a line of ML code: that's fine. The goal here
is to give you enough vocabulary to read the code, ask good questions,
and trust (or distrust) the results appropriately.

---

## What problem are we solving?

> Given everything we know up to right now — the last few hours of CGM
> readings, recent boluses, basal rate, Control-IQ activity — what will
> the glucose value be **30 minutes from now**?

That's it. One number, 30 minutes ahead. We pick 30 minutes because:

1. It's far enough out to be useful (you can pre-bolus, snack, suspend).
2. It's close enough that the signal-to-noise ratio is still good.
   Predicting 6 hours out is essentially impossible — too many
   unknowns (meals, exercise, illness, stress, sleep stages).

The "60 minutes ahead" target is also produced as a stretch goal so we
can compare horizons.

---

## Why XGBoost (and not a fancy neural network)?

**XGBoost** is a gradient-boosted decision tree library. Think of it as
"a few hundred small decision trees that vote together":

- Tree 1 says "if IOB > 4, subtract 18 mg/dL".
- Tree 2 says "if it's 7am and trend is rising, add 22 mg/dL".
- Tree 3 says "if there was a bolus 30 min ago and carbs unknown, add 12 mg/dL".
- ... 200 more trees ...
- Final prediction = current glucose + sum of all tree contributions.

Why we use it instead of a deep learning model:

1. **Tabular data is its sweet spot.** XGBoost almost always beats
   neural networks on small, structured datasets like a few months of
   CGM records.
2. **It handles missing values natively.** No need to fill in NaNs.
3. **It trains in seconds, not hours.** You can iterate fast.
4. **It's interpretable.** SHAP (see below) tells you exactly which
   features pushed any prediction up or down.
5. **It doesn't overfit easily** with reasonable hyper-parameters and
   early stopping.

For a child's glucose data — typically 30-90 days of 5-minute readings,
i.e. ~25k rows — XGBoost is overwhelmingly the right call.

---

## Features (the columns the model sees)

A "feature" is one column in the table the model trains on. The model
never sees raw time-series; it sees a snapshot of *engineered* numbers
at each point in time. We build features in `src/features.py`.

### 1. Lag features

> "What was glucose 5, 10, 15, 30, 60 minutes ago?"

These are the most predictive features by far. Glucose has strong
short-term momentum — knowing where it was tells you a lot about
where it's going.

### 2. Rolling stats (windows)

> "What's the **average** glucose over the last 30 / 60 / 120 minutes?"
> "What's the **standard deviation** (volatility) over those windows?"
> "What's the **slope** — how fast is it changing?"

These smooth out noise and capture trends a single lag can't.

### 3. Insulin-on-board (IOB)

> "How many units of insulin are still active in the body right now?"

We computed this in Phase 3 as either the pump's reported IOB or our
modeled exponential decay. IOB is a *lagging-impact* feature: a bolus
delivered 30 min ago is still actively pulling glucose down right now.

### 4. Recent insulin events

> "How many bolus units in the last 30 / 60 / 120 minutes?"
> "What's the current basal rate?"
> "Is Control-IQ currently suspending or boosting?"

These tell the model what's already in flight.

### 5. Circadian / time features

> "What hour of the day is it?" (encoded as sine/cosine to be cyclic)
> "What day of week?"

The dawn phenomenon (3-7am cortisol-driven glucose rise) is a strong
recurring pattern. Meal times also cluster by hour.

### 6. The trend arrow Dexcom already gives us

A free signal worth keeping.

---

## Train / validation / test split — and why naive splits cheat

This is **the most common mistake in time-series ML**, so it deserves
a section.

If you randomly shuffle the data and put 80% in "train" and 20% in
"test", you create a leak: the model sees the future and the past
mixed together, and your "test" score looks great while real-world
performance is awful.

We use a **walk-forward** split:

```
[TRAIN..............................] [VAL.......] [TEST.......]
        first 70% of the data           next 15%     last 15%
```

The model only learns from the past; we evaluate it on data that
chronologically came *after* training. That's the only honest way to
measure how this model would have performed on a future you didn't
have when training.

---

## Metrics — what "good" means

We report two error metrics on the test set:

- **MAE (Mean Absolute Error)** — average miss in mg/dL.
  *Example: MAE = 14 means on average we're off by 14 mg/dL.*
- **RMSE (Root Mean Squared Error)** — penalizes big misses more.
  Always >= MAE.

For 30-minute glucose prediction in T1D children, **MAE in the
12-18 mg/dL range** is competitive with published research baselines.
Below 10 is suspicious (probably a leak). Above 25 means the model is
struggling with this person's data and you should look at gaps,
sensor noise, or undocumented meals.

We also report:

- **% within ±20 mg/dL** of truth (clinical "close enough" threshold).
- **Hypo recall** — of all actual lows (<70), how many did we predict
  to be <80? You want this high for safety.

---

## SHAP — why did the model say *that*?

**SHAP (SHapley Additive exPlanations)** is a method for taking any
prediction and decomposing it into per-feature contributions:

> "We predicted 175 mg/dL. SHAP says: starting baseline 110, then:
> +35 because glucose 30 min ago was 165, +25 because rising-trend,
> -8 because IOB is 1.4U, +6 because hour of day is 8am, +7 misc."

This is the closest thing in ML to a "show your work". Use it to:

1. Sanity-check that the model is using sensible signals (it should
   weight recent glucose heavily).
2. Spot when it's relying on something dumb (e.g. day-of-week being
   a top-3 feature would be suspicious).
3. Explain a specific borderline prediction to a clinician.

`src/explain.py` produces both a global feature-importance summary
and per-row local explanations.

---

## Limitations — what this model **cannot** do

Read this section twice.

1. **It does not know about meals it wasn't told about.** No carbs,
   no snacks, no juice for a low. The model will systematically
   under-predict post-meal spikes that happened without a logged
   bolus.
2. **It does not know about exercise.** A bike ride at 4pm will look
   to the model like "weird drop, must be a sensor error".
3. **It does not know about illness, stress, sleep, hormones.**
4. **It is not a clinical device.** Use it for insight; don't dose
   from it.
5. **It assumes the sensor is accurate.** Compression lows and
   warmup-period nonsense will poison the input.
6. **It will degrade over time** as the child grows, basal needs
   change, and seasons shift. Plan to retrain every 4-8 weeks.

---

## Glossary

| term            | plain meaning                                              |
|-----------------|------------------------------------------------------------|
| feature         | one column the model sees                                  |
| label / target  | the thing we're trying to predict (glucose 30 min ahead)   |
| lag             | "value N minutes ago"                                      |
| rolling         | "stat over the last N minutes"                             |
| train / val / test | three disjoint slices of data (always in time order!)   |
| MAE             | average absolute error in mg/dL                            |
| RMSE            | error metric that punishes big misses harder than MAE      |
| overfitting     | model memorized the past, doesn't generalize               |
| early stopping  | stop training when the validation score stops improving    |
| SHAP            | per-prediction breakdown of feature contributions          |
| horizon         | how far ahead we predict (we use 30 min)                   |

---

## Where to look in the code

- `src/features.py` — every feature has an inline comment explaining it.
- `src/train.py` — the train/val/test split + XGBoost config.
- `src/predict.py` — produce predictions on new data.
- `src/explain.py` — SHAP global + local explanations.

Open any of those files alongside this primer and the comments should
read like a guided tour.
