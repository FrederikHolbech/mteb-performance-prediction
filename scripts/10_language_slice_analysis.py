"""Step 10: Compare predictive performance across task language groups.

This script evaluates the Tier-2 Random Forest on the same sampled LOT-O folds
used elsewhere in the thesis and records per-task R2 scores. It then summarizes
performance by held-out task language group.

Reads:
  data/training_data.csv

Outputs:
  output/language_slice_per_task.csv
  output/language_slice_summary.csv
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    TIERS,
    TIER1_CATEGORICALS,
    RF_PARAMS,
    EVAL_SEED,
    EVAL_N_TASKS,
    TARGET,
    TRAINING_DATA_PATH,
    OUTPUT_DIR,
)
from src.utils import sample_eligible_tasks

ANALYSIS_TIER = "Tier 2 (+ README)"
ANALYSIS_FEATURES = TIERS[ANALYSIS_TIER]
PER_TASK_PATH = os.path.join(OUTPUT_DIR, "language_slice_per_task.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "language_slice_summary.csv")


def _prepare(df, features):
    cat_cols = [
        c for c in features if df[c].dtype == "object" or c in TIER1_CATEGORICALS
    ]
    df_work = df[features + [TARGET, "task_name"]].copy()
    for col in cat_cols:
        df_work[col] = LabelEncoder().fit_transform(df_work[col].astype(str))
    return df_work[features], df_work[TARGET], df_work["task_name"]


def _task_group(meta_row):
    is_english = bool(meta_row["is_english"])
    is_multilingual = bool(meta_row["is_multilingual"])
    if is_english and not is_multilingual:
        return "English-only"
    if is_multilingual:
        return "Multilingual"
    return "No-English"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(TRAINING_DATA_PATH)
    X, y, task_names = _prepare(df, ANALYSIS_FEATURES)

    rf_params = RF_PARAMS.copy()
    rf_params["random_state"] = EVAL_SEED

    eval_tasks = sample_eligible_tasks(task_names, EVAL_N_TASKS, EVAL_SEED)

    rows = []
    for task in eval_tasks:
        test_mask = task_names == task
        train_mask = ~test_mask
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        train_med = X_train.median()
        X_train_f = X_train.fillna(train_med).fillna(0)
        X_test_f = X_test.fillna(train_med).fillna(0)

        model = RandomForestRegressor(**rf_params)
        model.fit(X_train_f, y_train)
        r2 = r2_score(y_test, model.predict(X_test_f))

        meta = df.loc[
            df["task_name"] == task,
            ["is_english", "is_multilingual", "num_languages"],
        ].iloc[0]
        rows.append(
            {
                "task_name": task,
                "language_group": _task_group(meta),
                "num_languages": int(meta["num_languages"]),
                "n_rows": int(len(X_test)),
                "r2": float(r2),
            }
        )

    per_task_df = pd.DataFrame(rows).sort_values("r2", ascending=False)
    summary_df = (
        per_task_df.groupby("language_group")["r2"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    per_task_df.to_csv(PER_TASK_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print(f"Saved {PER_TASK_PATH}")
    print(f"Saved {SUMMARY_PATH}")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
