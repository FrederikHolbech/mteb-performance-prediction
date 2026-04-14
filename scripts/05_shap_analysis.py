"""
Step 5: SHAP analysis across three feature tiers.

Trains RandomForest per task family for each tier and computes
mean |SHAP| values to understand feature importance differences
across data-availability tiers.

Reads:
  data/training_data.csv

Outputs:
  output/shap_by_family_tier_1.csv
  output/shap_by_family_tier_2.csv
  output/shap_by_family_tier_3.csv
  output/figures/shap_tier_comparison.png
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    TIERS,
    TIER1_CATEGORICALS,
    TIER1_FEATURES,
    README_FEATURES,
    DISCARDED_FEATURES,
    RF_PARAMS,
    EVAL_SEED,
    TARGET,
    TRAINING_DATA_PATH,
    SHAP_TIER1_PATH,
    SHAP_TIER2_PATH,
    SHAP_TIER3_PATH,
    OUTPUT_DIR,
    FIGURES_DIR,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

SHAP_PATHS = {
    "Tier 1 (API-only)": SHAP_TIER1_PATH,
    "Tier 2 (+ README)": SHAP_TIER2_PATH,
    "Tier 3 (+ discarded)": SHAP_TIER3_PATH,
}


# ============================================================
# SHAP computation per tier
# ============================================================


def compute_shap_per_tier(df):
    """Compute per-family SHAP importance for each tier."""
    # Lighter params for SHAP -- unlimited depth trees make TreeExplainer
    # extremely slow.  Feature importance rankings are stable with shallower
    # trees, so we cap depth and use fewer estimators here.
    shap_rf_params = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": EVAL_SEED,
    }

    tier_shap_results = {}
    tier_global_shap = {}

    for tier_idx, (tier_name, features) in enumerate(TIERS.items(), 1):
        print(f"\n--- {tier_name} ({tier_idx}/3) ---")
        # Remove task_type for per-family analysis (we split by family)
        features_no_task = [f for f in features if f != "task_type"]
        cat_cols = [
            c
            for c in features_no_task
            if df[c].dtype == "object" or c in TIER1_CATEGORICALS
        ]

        family_importance = {}
        families = [f for f in df["task_type"].unique() if f != "Summarization"]

        for fam_idx, family in enumerate(families, 1):
            df_family = df[df["task_type"] == family].copy()
            if len(df_family) < 100:
                print(f"  [{fam_idx}/{len(families)}] {family}: skipped (<100 rows)")
                continue

            print(
                f"  [{fam_idx}/{len(families)}] {family} ({len(df_family)} rows)...",
                end=" ",
                flush=True,
            )

            df_fam = df_family[features_no_task + [TARGET]].copy()
            for col in cat_cols:
                df_fam[col] = LabelEncoder().fit_transform(df_fam[col].astype(str))

            X_fam = df_fam[features_no_task]
            y_fam = df_fam[TARGET]

            mdl = RandomForestRegressor(**shap_rf_params)
            X_filled = X_fam.fillna(X_fam.median())
            mdl.fit(X_filled, y_fam)

            # Subsample for SHAP to keep runtime manageable
            shap_max_samples = 2000
            if len(X_filled) > shap_max_samples:
                X_shap = X_filled.sample(shap_max_samples, random_state=EVAL_SEED)
            else:
                X_shap = X_filled

            explainer = shap.TreeExplainer(mdl)
            shap_vals = explainer.shap_values(X_shap)

            importance = pd.Series(
                np.abs(shap_vals).mean(axis=0), index=features_no_task
            ).sort_values(ascending=False)

            family_importance[family] = importance
            print("done")

        tier_shap_results[tier_name] = family_importance

        # Global importance = mean across families
        imp_df = pd.DataFrame(family_importance)
        global_imp = imp_df.mean(axis=1).sort_values(ascending=False)
        tier_global_shap[tier_name] = global_imp

        # Save per-tier SHAP CSV
        out_path = SHAP_PATHS[tier_name]
        imp_df.round(4).to_csv(out_path)
        print(f"Saved {out_path}")

        print(f"\n{'='*60}")
        print(f"TOP 10 GLOBAL FEATURES — {tier_name}")
        print(f"{'='*60}")
        print(global_imp.head(10).to_string())

    return tier_shap_results, tier_global_shap


# ============================================================
# Visualization
# ============================================================


def plot_shap_comparison(tier_global_shap):
    """Generate the 3-panel SHAP comparison figure."""
    readme_set = set(README_FEATURES)
    discarded_set = set(DISCARDED_FEATURES)

    fig, axes = plt.subplots(1, 3, figsize=(22, 8), sharey=False)

    for idx, (tier_name, global_imp) in enumerate(tier_global_shap.items()):
        ax = axes[idx]
        top15 = global_imp.sort_values(ascending=True).tail(15)

        # Color by origin
        bar_colors = []
        for feat in top15.index:
            if feat in discarded_set:
                bar_colors.append("#F44336")  # red for discarded
            elif feat in readme_set:
                bar_colors.append("#FF9800")  # orange for README
            else:
                bar_colors.append("#2196F3")  # blue for API-only

        ax.barh(range(len(top15)), top15.values, color=bar_colors)
        ax.set_yticks(range(len(top15)))
        ax.set_yticklabels(top15.index, fontsize=8)
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title(f"{tier_name}\n({len(TIERS[tier_name])} features)", fontsize=11)
        ax.grid(axis="x", alpha=0.3)

    legend_elements = [
        Patch(facecolor="#2196F3", label="API-only features"),
        Patch(facecolor="#FF9800", label="README-scraped features"),
        Patch(facecolor="#F44336", label="Discarded (popularity)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.suptitle("Top-15 Global SHAP Features — Three Tiers", fontsize=14, y=1.01)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "shap_tier_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {fig_path}")


def print_feature_rankings(tier_global_shap):
    """Print where README and discarded features rank in their tiers."""
    tiers_list = list(TIERS.keys())

    print("\n" + "=" * 70)
    print("RANK OF README-SCRAPED FEATURES IN TIER 2")
    print("=" * 70)
    t2_imp = tier_global_shap[tiers_list[1]]
    for feat in sorted(README_FEATURES):
        if feat in t2_imp.index:
            rank = (t2_imp > t2_imp[feat]).sum() + 1
            print(f"  {feat:<28} SHAP={t2_imp[feat]:.4f}  rank={rank}/{len(t2_imp)}")

    print(f"\n{'='*70}")
    print("RANK OF DISCARDED FEATURES IN TIER 3")
    print("=" * 70)
    t3_imp = tier_global_shap[tiers_list[2]]
    for feat in DISCARDED_FEATURES:
        if feat in t3_imp.index:
            rank = (t3_imp > t3_imp[feat]).sum() + 1
            print(f"  {feat:<28} SHAP={t3_imp[feat]:.4f}  rank={rank}/{len(t3_imp)}")


# ============================================================
# Main
# ============================================================

RUN_SHAP = False  # Set True to re-run SHAP analysis

if __name__ == "__main__":

    df = pd.read_csv(TRAINING_DATA_PATH)
    print(f"Training data: {df.shape}")

    all_cached = all(os.path.exists(p) for p in SHAP_PATHS.values())

    if RUN_SHAP or not all_cached:
        print("\nComputing SHAP per tier...")
        tier_shap_results, tier_global_shap = compute_shap_per_tier(df)
    else:
        print("\nLoaded SHAP results from cache")
        tier_global_shap = {}
        for tier_name, path in SHAP_PATHS.items():
            imp_df = pd.read_csv(path, index_col=0)
            tier_global_shap[tier_name] = imp_df.mean(axis=1).sort_values(
                ascending=False
            )

    print("\nVisualization...")
    plot_shap_comparison(tier_global_shap)

    print_feature_rankings(tier_global_shap)
