"""'What if' simulator for the trained glucose forecaster.

For non-ML readers: a counterfactual asks "what would have happened if
something had been different?" Here, we ask the model: "what would you
predict if a 2U bolus had just been delivered?" — without actually
delivering anything. We do this by:

  1. Taking the most recent feature row (current state).
  2. Making a copy of it.
  3. Mechanically modifying the copy to reflect the hypothetical action
     (e.g. adding 2U to the IOB and recent-bolus features).
  4. Re-running the model on both rows.
  5. Reporting the difference.

The output is the model's *belief* about the action's effect, not a
physiological simulation. It's only as accurate as the patterns the
model has seen. See docs/ML_PRIMER.md §"What this model cannot do"
for the honest limits.

Supported hypothetical actions:
  * bolus(units)          — pretend an immediate bolus was just delivered.
  * suspend(minutes)      — pretend basal was suspended for N minutes.
                            Only tweaks features that look at recent
                            basal / IOB; the prediction horizon is 30
                            min so suspend effects beyond that don't
                            show up in this single forecast.

Carbs are NOT modeled because we don't carry a carbs feature in the
unified timeline. A future phase could add Nightscout-style carb logging
and extend this module.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Same exponential time-constant we use in merge_pipeline.IOB modeling.
IOB_TAU_MIN = 240.0


ActionKind = Literal["bolus", "suspend"]


@dataclass
class Action:
    kind: ActionKind
    units: float = 0.0       # for bolus
    minutes: float = 0.0     # for suspend


def _decay_factor(elapsed_min: float, tau_min: float = IOB_TAU_MIN) -> float:
    """exp(-t/tau) — the fraction of a bolus still active after t minutes."""
    return float(np.exp(-elapsed_min / tau_min))


def apply_action(row: pd.DataFrame, action: Action) -> pd.DataFrame:
    """Return a copy of `row` (single-row DataFrame) with features
    mutated to reflect `action`. We work with DataFrames rather than
    Series because XGBoost is dtype-strict and Series force everything
    to a single dtype on conversion.

    Only the columns the model sees are touched. We don't touch
    glucose_mg_dl itself — we want the model to *predict* what the new
    glucose will be, not be told.
    """
    new = row.copy()
    cols = new.columns

    if action.kind == "bolus" and action.units > 0:
        # Bump every "bolus_sum_Nm" feature — they're cumulative units
        # in the trailing N-min window. The bolus we're hypothetically
        # giving "just happened", so it lands inside every window.
        for col in cols:
            if col.startswith("bolus_sum_"):
                new.loc[:, col] = new[col].astype("float32") + float(action.units)

        if "iob_units" in cols:
            new.loc[:, "iob_units"] = new["iob_units"].astype("float32") + float(action.units)
        if "minutes_since_bolus" in cols:
            new.loc[:, "minutes_since_bolus"] = 0.0

    elif action.kind == "suspend" and action.minutes > 0:
        if "basal_rate" in cols:
            new.loc[:, "basal_rate"] = 0.0
        if "is_suspended" in cols:
            new.loc[:, "is_suspended"] = 1
        # IOB doesn't change — suspending basal stops *new* insulin but
        # active IOB keeps decaying naturally. The 30-min horizon is
        # too short to see most of a suspend's downstream effect on
        # glucose; document this in the UI.

    return new


def simulate(
    model,
    feature_cols: list[str],
    current_row: pd.DataFrame,
    actions: list[Action],
) -> pd.DataFrame:
    """Run the model on the current row and on each hypothetical action.

    `current_row` is a single-row DataFrame (e.g. ``feats.iloc[[-1]]``)
    so dtypes survive. Returns one result row per scenario:
      scenario, prediction_30m, delta_vs_baseline.
    """
    if isinstance(current_row, pd.Series):
        # Convenience: callers sometimes pass a Series. Reconstruct as
        # a single-row DataFrame using the original feats columns.
        raise TypeError("simulate() requires a single-row DataFrame; use feats.iloc[[idx]]")

    rows: list[dict] = []

    baseline_features = current_row[feature_cols]
    baseline_pred = float(model.predict(baseline_features)[0])
    rows.append({
        "scenario": "baseline (no action)",
        "prediction_30m": baseline_pred,
        "delta_vs_baseline": 0.0,
    })

    for action in actions:
        modified = apply_action(current_row, action)
        pred = float(model.predict(modified[feature_cols])[0])
        if action.kind == "bolus":
            label = f"bolus +{action.units:g}U now"
        elif action.kind == "suspend":
            label = f"suspend basal for {action.minutes:g} min"
        else:  # pragma: no cover
            label = action.kind
        rows.append({
            "scenario": label,
            "prediction_30m": pred,
            "delta_vs_baseline": pred - baseline_pred,
        })

    return pd.DataFrame(rows)


def standard_action_grid() -> list[Action]:
    """A reasonable default set of actions to show side-by-side."""
    return [
        Action("bolus", units=0.5),
        Action("bolus", units=1.0),
        Action("bolus", units=2.0),
        Action("bolus", units=3.0),
        Action("suspend", minutes=30),
        Action("suspend", minutes=60),
    ]
