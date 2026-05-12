"""Step 8: SHAP filtering combined with correlation-cluster pruning.

This analysis answers a stricter feature-selection question than Step 6:
start from the Tier-2 SHAP ranking, keep only the top-k most important
features, and within that subset remove redundant correlated features by
keeping the highest-SHAP feature from each correlation cluster.

Reads:
  data/training_data.csv
  output/shap_by_family_tier_2.csv

Outputs:
  output/shap_cluster_selection_results.csv
  output/shap_cluster_best_model.csv
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
    SHAP_TIER2_PATH,
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
CORR_THRESHOLD = 0.85
TOP_K_CANDIDATES = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]


def _prepare(df, features):
    """Encode categoricals and return X, y, task names."""
    cat_cols = [
        c for c in features if df[c].dtype == "object" or c in TIER1_CATEGORICALS
    ]
    df_work = df[features + [TARGET, "task_name"]].copy()
    for col in cat_cols:
        df_work[col] = LabelEncoder().fit_transform(df_work[col].astype(str))
    return df_work[features], df_work[TARGET], df_work["task_name"]


def _lot_o_r2(X, y, task_names, eval_tasks, rf_params):
    """Run leave-one-task-out CV and return mean R2."""
    scores = []
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
        scores.append(r2_score(y_test, model.predict(X_test_f)))
    return np.mean(scores) if scores else np.nan


def _build_corr_matrix(df, features):
    """Compute Pearson correlation matrix over encoded features."""
    X, _, _ = _prepare(df, features)
    X_filled = X.fillna(X.median()).fillna(0)
    return X_filled.corr(method="pearson")


def _find_components(features, corr, threshold):
    """Find connected components using absolute correlation threshold."""
    adjacency = {feature: set() for feature in features}
    for i, left in enumerate(features):
        for right in features[i + 1 :]:
            if abs(corr.loc[left, right]) >= threshold:
                adjacency[left].add(right)
                adjacency[right].add(left)

    visited = set()
    components = []
    for feature in features:
        if feature in visited:
            continue
        queue = [feature]
        component = []
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


def prune_by_shap_and_correlation(top_features, global_shap, corr, threshold):
    """Keep the highest-SHAP feature from each correlation component."""
    components = _find_components(top_features, corr, threshold)
    kept = []
    dropped = []
    for component in components:
        if len(component) == 1:
            kept.append(component[0])
            continue
        ranked = sorted(
            component, key=lambda feature: global_shap[feature], reverse=True
        )
        kept.append(ranked[0])
        dropped.extend(ranked[1:])

    kept_sorted = sorted(kept, key=lambda feature: global_shap[feature], reverse=True)
    dropped_sorted = sorted(
        dropped, key=lambda feature: global_shap[feature], reverse=True
    )
    return kept_sorted, dropped_sorted, components


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 8: SHAP + CLUSTER FEATURE SELECTION")
    print("=" * 60)

    df = pd.read_csv(TRAINING_DATA_PATH)
    shap_df = pd.read_csv(SHAP_TIER2_PATH, index_col=0)
    global_shap = shap_df.mean(axis=1).sort_values(ascending=False)

    rf_params = RF_PARAMS.copy()
    rf_params["random_state"] = EVAL_SEED

    corr = _build_corr_matrix(df, ANALYSIS_FEATURES)

    X_full, y, task_names = _prepare(df, ANALYSIS_FEATURES)
    eval_tasks = sample_eligible_tasks(task_names, EVAL_N_TASKS, EVAL_SEED)
    baseline_r2 = _lot_o_r2(X_full, y, task_names, eval_tasks, rf_params)
    print(f"Baseline Tier-2 R2 ({len(ANALYSIS_FEATURES)} features): {baseline_r2:.4f}")

    rows = []
    for top_k in TOP_K_CANDIDATES:
        candidate_features = [
            f for f in global_shap.head(top_k).index if f in ANALYSIS_FEATURES
        ]
        kept, dropped, components = prune_by_shap_and_correlation(
            candidate_features, global_shap, corr, CORR_THRESHOLD
        )

        X_sel, _, _ = _prepare(df, kept)
        r2 = _lot_o_r2(X_sel, y, task_names, eval_tasks, rf_params)
        delta = r2 - baseline_r2

        rows.append(
            {
                "top_k_shap": top_k,
                "features_after_shap": len(candidate_features),
                "features_after_cluster_prune": len(kept),
                "dropped_by_shap": len(ANALYSIS_FEATURES) - len(candidate_features),
                "dropped_by_correlation": len(dropped),
                "n_components": len(components),
                "r2": round(r2, 6),
                "r2_delta_vs_baseline": round(delta, 6),
                "kept_features": ", ".join(kept),
                "dropped_features": ", ".join(dropped),
            }
        )
        sign = "+" if delta >= 0 else ""
        print(
            f"top-{top_k:2d} SHAP -> {len(kept):2d} after prune | "
            f"R2={r2:.4f} ({sign}{delta:.4f})"
        )

    results_df = pd.DataFrame(rows).sort_values(
        ["r2", "features_after_cluster_prune"], ascending=[False, True]
    )
    results_path = os.path.join(OUTPUT_DIR, "shap_cluster_selection_results.csv")
    results_df.to_csv(results_path, index=False)

    best_row = results_df.iloc[0].to_dict()
    best_df = pd.DataFrame([best_row])
    best_path = os.path.join(OUTPUT_DIR, "shap_cluster_best_model.csv")
    best_df.to_csv(best_path, index=False)
    print(
        best_df[
            [
                "top_k_shap",
                "features_after_cluster_prune",
                "r2",
                "r2_delta_vs_baseline",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
