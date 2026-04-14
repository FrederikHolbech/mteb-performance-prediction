"""Step 6: Feature correlation, cluster analysis, and ablation study.

Analyses performed:
  1. Correlation matrix -- Pearson pairwise correlation heatmap for Tier 2
     features, with highly-correlated pairs (|r| > 0.85) listed explicitly.
  2. Cluster detection -- Groups of 3+ features with |r| >= 0.85 are
     identified as clusters via connected components.
  3. Ablation study -- Remove one feature at a time from Tier 2 and measure
     the R2 drop (leave-one-task-out CV with RandomForest).
  4. Lean model -- Keep one representative per cluster (lowest ablation
     impact on R2), retrain and evaluate the reduced feature set.

Reads:
  data/training_data.csv

Outputs:
  output/correlation_matrix.csv
  output/high_correlations.csv
  output/feature_clusters.csv
  output/ablation_results.csv
  output/lean_model_results.csv
  output/figures/correlation_heatmap.png
  output/figures/ablation_study.png
"""

import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

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
    FIGURES_DIR,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Use Tier 2 for all analyses (the practical working tier)
ANALYSIS_TIER = "Tier 2 (+ README)"
ANALYSIS_FEATURES = TIERS[ANALYSIS_TIER]

# Correlation threshold
CORR_THRESHOLD = 0.85

# Caching flags
RUN_CORRELATION = True  # Set True to re-run correlation analysis
RUN_ABLATION = True  # Set True to re-run ablation study
RUN_LEAN_MODEL = True  # Set True to re-run lean model evaluation


# ============================================================
# Helper: prepare features
# ============================================================


def _prepare(df, features):
    """Encode categoricals, return X, y, task_names."""
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
        if len(X_test) < 5:
            continue
        train_med = X_train.median()
        X_train_f = X_train.fillna(train_med).fillna(0)
        X_test_f = X_test.fillna(train_med).fillna(0)
        model = RandomForestRegressor(**rf_params)
        model.fit(X_train_f, y_train)
        scores.append(r2_score(y_test, model.predict(X_test_f)))
    return np.mean(scores) if scores else np.nan


# ============================================================
# 1. Feature Correlation Analysis
# ============================================================


def run_correlation_analysis(df):
    """Compute Pearson correlations and identify highly-correlated pairs."""
    print("=" * 60)
    print("1. FEATURE CORRELATION ANALYSIS")
    print("=" * 60)

    X, _, _ = _prepare(df, ANALYSIS_FEATURES)
    X_filled = X.fillna(X.median()).fillna(0)

    # Pearson correlation
    corr = X_filled.corr(method="pearson")
    corr_path = os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
    corr.to_csv(corr_path)
    print(f"Saved full correlation matrix: {corr_path}")

    # Find highly-correlated pairs
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= CORR_THRESHOLD:
                pairs.append(
                    {
                        "feature_1": cols[i],
                        "feature_2": cols[j],
                        "pearson_r": round(r, 4),
                    }
                )

    pairs_df = pd.DataFrame(pairs).sort_values("pearson_r", key=abs, ascending=False)
    pairs_path = os.path.join(OUTPUT_DIR, "high_correlations.csv")
    pairs_df.to_csv(pairs_path, index=False)
    print(f"\nHighly correlated pairs (|r| >= {CORR_THRESHOLD}): {len(pairs_df)}")
    if len(pairs_df) > 0:
        print(pairs_df.to_string(index=False))
    print(f"Saved: {pairs_path}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(20, 16))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=6)
    ax.set_yticklabels(cols, fontsize=6)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(f"Feature Correlation Heatmap ({ANALYSIS_TIER}, {len(cols)} features)")
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved: {fig_path}")

    return corr, pairs_df


# ============================================================
# 1b. Cluster Detection (connected components at |r| >= threshold)
# ============================================================


def find_feature_clusters(corr, threshold=CORR_THRESHOLD):
    """Find clusters of correlated features via connected components."""
    print("\n" + "=" * 60)
    print("1b. FEATURE CLUSTER DETECTION")
    print("=" * 60)

    features = corr.columns.tolist()
    n = len(features)

    # Build adjacency: feature i <-> feature j if |r| >= threshold
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr.iloc[i, j]) >= threshold:
                adj[i].add(j)
                adj[j].add(i)

    # Connected components via BFS
    visited = set()
    clusters = []
    for start in range(n):
        if start in visited or not adj[start]:
            continue  # skip isolated features
        component = []
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for nb in adj[node]:
                if nb not in visited:
                    queue.append(nb)
        if len(component) > 2:
            clusters.append(sorted(component))

    # Build output dataframe
    rows = []
    for cid, members in enumerate(clusters, 1):
        member_names = [features[m] for m in members]
        for feat in member_names:
            rows.append(
                {"cluster_id": cid, "feature": feat, "cluster_size": len(members)}
            )
    clusters_df = pd.DataFrame(rows)

    cluster_path = os.path.join(OUTPUT_DIR, "feature_clusters.csv")
    clusters_df.to_csv(cluster_path, index=False)

    print(f"Found {len(clusters)} clusters (threshold |r| >= {threshold}):")
    for cid, members in enumerate(clusters, 1):
        member_names = [features[m] for m in members]
        print(f"  Cluster {cid} ({len(members)} features): {', '.join(member_names)}")
    print(f"Saved: {cluster_path}")

    return clusters_df


# ============================================================
# 2. Ablation Study (leave-one-feature-out)
# ============================================================


def run_ablation_study(df, baseline_r2=None):
    """Remove one feature at a time and measure R2 change."""
    print("\n" + "=" * 60)
    print("2. ABLATION STUDY (leave-one-feature-out)")
    print("=" * 60)

    rf_params = RF_PARAMS.copy()
    rf_params["random_state"] = EVAL_SEED

    X_full, y, task_names = _prepare(df, ANALYSIS_FEATURES)

    # Pick evaluation tasks
    tasks_all = df["task_name"].unique()
    np.random.seed(EVAL_SEED)
    eval_tasks = np.random.choice(
        tasks_all, size=min(EVAL_N_TASKS, len(tasks_all)), replace=False
    )

    # Baseline: all features
    if baseline_r2 is None:
        print("Computing baseline R2 with all features...")
        baseline_r2 = _lot_o_r2(X_full, y, task_names, eval_tasks, rf_params)
    print(f"Baseline R2 ({len(ANALYSIS_FEATURES)} features): {baseline_r2:.4f}\n")

    results = []
    n_features = len(ANALYSIS_FEATURES)
    for i, feat in enumerate(ANALYSIS_FEATURES):
        remaining = [f for f in ANALYSIS_FEATURES if f != feat]
        X_reduced, _, _ = _prepare(df, remaining)
        r2 = _lot_o_r2(X_reduced, y, task_names, eval_tasks, rf_params)
        delta = r2 - baseline_r2
        results.append(
            {
                "removed_feature": feat,
                "r2_without": round(r2, 6),
                "r2_delta": round(delta, 6),
                "baseline_r2": round(baseline_r2, 6),
            }
        )
        direction = "+" if delta >= 0 else ""
        print(
            f"  [{i+1:2d}/{n_features}] Remove {feat:<35s} -> R2={r2:.4f} ({direction}{delta:.4f})"
        )

    results_df = pd.DataFrame(results).sort_values("r2_delta", ascending=True)
    results_path = os.path.join(OUTPUT_DIR, "ablation_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved: {results_path}")

    # Top features whose removal hurts most
    print("\nTop 10 features whose removal HURTS performance:")
    print(
        results_df.head(10)[["removed_feature", "r2_without", "r2_delta"]].to_string(
            index=False
        )
    )

    print("\nFeatures whose removal IMPROVES performance:")
    improves = results_df[results_df["r2_delta"] > 0]
    if len(improves) > 0:
        print(
            improves[["removed_feature", "r2_without", "r2_delta"]].to_string(
                index=False
            )
        )
    else:
        print("  (none)")

    # Plot
    _plot_ablation(results_df)

    return results_df, baseline_r2


def _plot_ablation(results_df):
    """Generate ablation bar chart."""
    results_sorted = results_df.sort_values("r2_delta")
    fig, ax = plt.subplots(figsize=(10, max(8, len(results_sorted) * 0.25)))
    colors = ["#e74c3c" if d < 0 else "#27ae60" for d in results_sorted["r2_delta"]]
    ax.barh(range(len(results_sorted)), results_sorted["r2_delta"], color=colors)
    ax.set_yticks(range(len(results_sorted)))
    ax.set_yticklabels(results_sorted["removed_feature"], fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("R2 change when feature is removed")
    ax.set_title(f"Ablation Study -- {ANALYSIS_TIER}")
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "ablation_study.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved: {fig_path}")


# ============================================================
# 3. Lean Model (one representative per cluster)
# ============================================================


def run_lean_model(df, clusters_df, ablation_df, baseline_r2):
    """Keep one feature per cluster (the most important), retrain and evaluate."""
    print("\n" + "=" * 60)
    print("3. LEAN MODEL (one representative per cluster)")
    print("=" * 60)

    rf_params = RF_PARAMS.copy()
    rf_params["random_state"] = EVAL_SEED

    # For each cluster, pick the representative = the one whose removal hurts
    # R2 the most (lowest r2_delta in ablation results)
    ablation_lookup = dict(zip(ablation_df["removed_feature"], ablation_df["r2_delta"]))

    features_to_drop = []
    cluster_ids = clusters_df["cluster_id"].unique()
    print(f"\n{len(cluster_ids)} clusters to reduce:\n")

    for cid in sorted(cluster_ids):
        members = clusters_df.loc[clusters_df["cluster_id"] == cid, "feature"].tolist()
        # Sort by ablation delta ascending — most negative = most important = keeper
        members_ranked = sorted(members, key=lambda f: ablation_lookup.get(f, 0))
        keeper = members_ranked[0]
        dropped = members_ranked[1:]
        features_to_drop.extend(dropped)
        print(
            f"  Cluster {cid}: keep '{keeper}' "
            f"(delta={ablation_lookup.get(keeper, 0):+.4f}), "
            f"drop {dropped}"
        )

    lean_features = [f for f in ANALYSIS_FEATURES if f not in features_to_drop]
    print(f"\nOriginal features: {len(ANALYSIS_FEATURES)}")
    print(f"Dropped (redundant): {len(features_to_drop)}")
    print(f"Lean features:     {len(lean_features)}")

    # Evaluate lean model
    X_lean, y, task_names = _prepare(df, lean_features)
    tasks_all = df["task_name"].unique()
    np.random.seed(EVAL_SEED)
    eval_tasks = np.random.choice(
        tasks_all, size=min(EVAL_N_TASKS, len(tasks_all)), replace=False
    )

    print("\nEvaluating lean model (LOT-O CV)...")
    lean_r2 = _lot_o_r2(X_lean, y, task_names, eval_tasks, rf_params)
    diff = lean_r2 - baseline_r2
    direction = "+" if diff >= 0 else ""
    print(
        f"Lean model R2: {lean_r2:.4f} ({direction}{diff:.4f} vs baseline {baseline_r2:.4f})"
    )

    # Save results
    results = {
        "baseline_features": len(ANALYSIS_FEATURES),
        "lean_features": len(lean_features),
        "features_dropped": len(features_to_drop),
        "baseline_r2": round(baseline_r2, 6),
        "lean_r2": round(lean_r2, 6),
        "r2_delta": round(diff, 6),
        "dropped_features": ", ".join(features_to_drop),
        "kept_features": ", ".join(lean_features),
    }
    results_df = pd.DataFrame([results])
    results_path = os.path.join(OUTPUT_DIR, "lean_model_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved: {results_path}")

    return results_df


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 6: FEATURE ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(TRAINING_DATA_PATH)
    print(f"Training data: {df.shape}")
    print(f"Analysis tier: {ANALYSIS_TIER} ({len(ANALYSIS_FEATURES)} features)\n")

    # 1. Correlation
    corr_path = os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
    pairs_path = os.path.join(OUTPUT_DIR, "high_correlations.csv")

    if RUN_CORRELATION or not os.path.exists(corr_path):
        corr, high_corr = run_correlation_analysis(df)
    else:
        corr = pd.read_csv(corr_path, index_col=0)
        high_corr = pd.read_csv(pairs_path)
        print(f"Loaded correlation data from cache ({len(high_corr)} high-corr pairs)")

    # 1b. Cluster detection
    cluster_path = os.path.join(OUTPUT_DIR, "feature_clusters.csv")

    if RUN_CORRELATION or not os.path.exists(cluster_path):
        clusters_df = find_feature_clusters(corr)
    else:
        clusters_df = pd.read_csv(cluster_path)
        n_clusters = clusters_df["cluster_id"].nunique() if len(clusters_df) > 0 else 0
        print(f"Loaded cluster data from cache ({n_clusters} clusters)")

    # 2. Ablation
    ablation_path = os.path.join(OUTPUT_DIR, "ablation_results.csv")

    if RUN_ABLATION or not os.path.exists(ablation_path):
        ablation_df, baseline_r2 = run_ablation_study(df)
    else:
        ablation_df = pd.read_csv(ablation_path)
        baseline_r2 = ablation_df["baseline_r2"].iloc[0]
        print(f"Loaded ablation results from cache (baseline R2={baseline_r2:.4f})")
        # Regenerate plot from cached data
        _plot_ablation(ablation_df)

    # 3. Lean model (one representative per cluster)
    lean_path = os.path.join(OUTPUT_DIR, "lean_model_results.csv")

    if len(clusters_df) > 0:
        if RUN_LEAN_MODEL or not os.path.exists(lean_path):
            lean_df = run_lean_model(df, clusters_df, ablation_df, baseline_r2)
        else:
            lean_df = pd.read_csv(lean_path)
            print(
                f"Loaded lean model results from cache (R2={lean_df['lean_r2'].iloc[0]:.4f})"
            )
    else:
        print("\nNo clusters found — skipping lean model.")
        lean_df = None

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline R2 (all {len(ANALYSIS_FEATURES)} features): {baseline_r2:.4f}")
    print(f"Highly correlated pairs (|r| >= {CORR_THRESHOLD}): {len(high_corr)}")
    n_clusters = clusters_df["cluster_id"].nunique() if len(clusters_df) > 0 else 0
    print(f"Feature clusters: {n_clusters}")
    n_harmful = len(ablation_df[ablation_df["r2_delta"] > 0])
    n_helpful = len(ablation_df[ablation_df["r2_delta"] < -0.001])
    print(f"Features whose removal IMPROVES R2: {n_harmful}")
    print(f"Features whose removal HURTS R2 (>0.001): {n_helpful}")
    if lean_df is not None:
        print(
            f"Lean model: {lean_df['lean_features'].iloc[0]} features, R2={lean_df['lean_r2'].iloc[0]:.4f}"
        )
    print("\nDone!")
