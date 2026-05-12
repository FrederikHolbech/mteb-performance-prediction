"""Step 9: Analyze depth versus width at comparable parameter counts.

This script aggregates the training data to one row per model and studies
whether deeper or wider architectures are associated with better mean MTEB
performance once overall model size is held roughly constant.

Reads:
  data/training_data.csv

Outputs:
  output/depth_width_model_level.csv
  output/depth_width_summary.csv
  output/depth_width_by_size_bin.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import OUTPUT_DIR, TRAINING_DATA_PATH

MODEL_LEVEL_PATH = os.path.join(OUTPUT_DIR, "depth_width_model_level.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "depth_width_summary.csv")
BY_BIN_PATH = os.path.join(OUTPUT_DIR, "depth_width_by_size_bin.csv")


def fit_linear(x, y):
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]

    y_arr = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(len(x_arr)), x_arr])
    coef, _, _, _ = np.linalg.lstsq(design, y_arr, rcond=None)
    preds = design @ coef
    return coef, preds


def build_model_level_dataframe():
    train = pd.read_csv(TRAINING_DATA_PATH)
    cols = [
        "model_name",
        "task_name",
        "norm_rank",
        "hidden_size",
        "num_layers",
        "param_count",
    ]
    model_task = train[cols].drop_duplicates(subset=["model_name", "task_name"])

    model_level = (
        model_task.groupby("model_name")
        .agg(
            {
                "norm_rank": "mean",
                "hidden_size": "first",
                "num_layers": "first",
                "param_count": "first",
                "task_name": "count",
            }
        )
        .rename(columns={"task_name": "n_tasks"})
        .reset_index()
    )

    model_level = model_level.dropna(
        subset=["norm_rank", "hidden_size", "num_layers", "param_count"]
    )
    model_level = model_level[
        (model_level["hidden_size"] > 0)
        & (model_level["num_layers"] > 0)
        & (model_level["param_count"] > 0)
    ].copy()

    model_level["log_param_count"] = np.log10(model_level["param_count"])
    return model_level


def compute_summary(model_level):
    predictors = model_level[["log_param_count", "num_layers", "hidden_size"]]
    target = model_level["norm_rank"]

    predictors_z = (predictors - predictors.mean()) / predictors.std(ddof=0)
    target_z = (target - target.mean()) / target.std(ddof=0)
    coef_std, _ = fit_linear(predictors_z, target_z)

    summary_rows = []
    for feature, coef in zip(predictors.columns, coef_std[1:]):
        summary_rows.append(
            {
                "metric": f"standardized_coef_{feature}",
                "value": float(coef),
            }
        )

    for feature in ["num_layers", "hidden_size"]:
        _, preds = fit_linear(model_level[["log_param_count"]], model_level[feature])
        residuals = model_level[feature] - preds
        corr = np.corrcoef(residuals, model_level["norm_rank"])[0, 1]
        model_level[f"{feature}_residual"] = residuals
        summary_rows.append(
            {
                "metric": f"partial_corr_like_{feature}",
                "value": float(corr),
            }
        )

    depth_mask = model_level["num_layers_residual"] > 0
    width_mask = model_level["hidden_size_residual"] > 0
    summary_rows.extend(
        [
            {
                "metric": "n_models",
                "value": float(len(model_level)),
            },
            {
                "metric": "mean_norm_rank_deeper_than_expected",
                "value": float(model_level.loc[depth_mask, "norm_rank"].mean()),
            },
            {
                "metric": "mean_norm_rank_shallower_than_expected",
                "value": float(model_level.loc[~depth_mask, "norm_rank"].mean()),
            },
            {
                "metric": "mean_norm_rank_wider_than_expected",
                "value": float(model_level.loc[width_mask, "norm_rank"].mean()),
            },
            {
                "metric": "mean_norm_rank_narrower_than_expected",
                "value": float(model_level.loc[~width_mask, "norm_rank"].mean()),
            },
        ]
    )

    summary_df = pd.DataFrame(summary_rows)
    return model_level, summary_df


def compute_by_size_bin(model_level):
    working = model_level.copy()
    working["size_bin"] = pd.qcut(working["log_param_count"], q=5, duplicates="drop")

    rows = []
    for size_bin, group in working.groupby("size_bin", observed=False):
        rows.append(
            {
                "size_bin": str(size_bin),
                "n_models": int(len(group)),
                "depth_perf_corr": float(
                    np.corrcoef(group["num_layers"], group["norm_rank"])[0, 1]
                ),
                "width_perf_corr": float(
                    np.corrcoef(group["hidden_size"], group["norm_rank"])[0, 1]
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_level = build_model_level_dataframe()
    model_level, summary_df = compute_summary(model_level)
    by_bin_df = compute_by_size_bin(model_level)

    model_level.to_csv(MODEL_LEVEL_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    by_bin_df.to_csv(BY_BIN_PATH, index=False)

    print(f"Saved {MODEL_LEVEL_PATH}")
    print(f"Saved {SUMMARY_PATH}")
    print(f"Saved {BY_BIN_PATH}")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
