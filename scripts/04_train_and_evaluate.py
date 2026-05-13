"""
Step 4: Train and evaluate models using three feature tiers.

Runs leave-one-task-out cross-validation with Ridge and RandomForest
across three feature tiers:
  Tier 1 -- API-only features (config.json, tokenizer.json, MTEB metadata)
  Tier 2 -- + README-scraped features (training keywords, method flags)
  Tier 3 -- + Discarded popularity features (downloads, likes, model_age_days)

Also produces per-task-family breakdowns.

Reads:
  data/training_data.csv

Outputs:
  output/model_comparison.csv
  output/tier_results.csv
  output/tier_comparison_by_family.csv
  output/figures/model_comparison.png
  output/figures/tier_comparison.png
"""

import os
import sys
import warnings
from itertools import product

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    TIERS,
    TIER1_CATEGORICALS,
    RF_PARAMS,
    RF_GRID,
    EVAL_SEED,
    EVAL_N_TASKS,
    TARGET,
    TRAINING_DATA_PATH,
    TIER_COMPARISON_PATH,
    OUTPUT_DIR,
    FIGURES_DIR,
)
from src.utils import sample_eligible_tasks

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# Helper: prepare tier data (encode cats, fill NaNs)
# ============================================================


def _prepare_tier(df, features):
    """Encode categoricals, return X, y, task_name series."""
    cat_cols = [
        c for c in features if df[c].dtype == "object" or c in TIER1_CATEGORICALS
    ]

    df_tier = df[features + [TARGET, "task_name"]].copy()
    for col in cat_cols:
        df_tier[col] = LabelEncoder().fit_transform(df_tier[col].astype(str))

    X = df_tier[features]
    y = df_tier[TARGET]

    return X, y, df_tier["task_name"]


# ============================================================
# Hyperparameter grid search (RandomForest)
# ============================================================


def run_grid_search(df, tier_name="Tier 2 (+ README)"):
    """Grid search over RandomForest hyperparameters using LOT-O CV."""
    features = TIERS[tier_name]
    X, y, task_names = _prepare_tier(df, features)
    eval_tasks = sample_eligible_tasks(task_names, EVAL_N_TASKS, EVAL_SEED)

    # Build parameter combinations
    param_names = list(RF_GRID.keys())
    param_combos = list(product(*RF_GRID.values()))
    print(f"Grid search: {len(param_combos)} combinations on {tier_name}")
    print(f"  {param_names} = {[RF_GRID[k] for k in param_names]}")

    best_score = -np.inf
    best_params = {}
    results = []

    for combo in param_combos:
        params = dict(zip(param_names, combo))
        params["n_jobs"] = -1
        params["random_state"] = EVAL_SEED

        scores = []
        for task in eval_tasks:
            test_mask = task_names == task
            train_mask = ~test_mask
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            train_med = X_train.median()
            X_train_f = X_train.fillna(train_med).fillna(0)
            X_test_f = X_test.fillna(train_med).fillna(0)

            model = RandomForestRegressor(**params)
            model.fit(X_train_f, y_train)
            scores.append(r2_score(y_test, model.predict(X_test_f)))

        mean_r2 = np.mean(scores)
        results.append({**params, "mean_r2": mean_r2, "std_r2": np.std(scores)})
        print(f"  {params} -> R2={mean_r2:.4f}")

        if mean_r2 > best_score:
            best_score = mean_r2
            best_params = params

    print(f"\nBest: {best_params} -> R2={best_score:.4f}")

    # Save grid search results
    grid_df = pd.DataFrame(results).sort_values("mean_r2", ascending=False)
    grid_path = os.path.join(OUTPUT_DIR, "grid_search_results.csv")
    grid_df.to_csv(grid_path, index=False)
    print(f"Saved {grid_path}")

    return best_params


# ============================================================
# Model comparison (run once, cache)
# ============================================================


def run_model_comparison(df, rf_params=None, tier_name="Tier 2 (+ README)"):
    """Compare multiple model types on a single tier using LOT-O CV.

    Model comparison is run once and cached to model_comparison.csv.
    """
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Lasso, ElasticNet

    rf_params = rf_params or RF_PARAMS
    features = TIERS[tier_name]
    cat_cols = [
        c for c in features if df[c].dtype == "object" or c in TIER1_CATEGORICALS
    ]

    df_tier = df[features + [TARGET, "task_name"]].copy()
    for col in cat_cols:
        df_tier[col] = LabelEncoder().fit_transform(df_tier[col].astype(str))

    X = df_tier[features]
    y = df_tier[TARGET]
    cat_indices = [features.index(c) for c in cat_cols]

    eval_tasks = sample_eligible_tasks(df_tier["task_name"], EVAL_N_TASKS, EVAL_SEED)

    lgb_params = {"n_estimators": 200, "learning_rate": 0.05, "verbose": -1}
    xgb_params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "verbosity": 0,
        "enable_categorical": True,
    }

    model_scores = {
        "Ridge": [],
        "Lasso": [],
        "ElasticNet": [],
        "RandomForest": [],
        "GradientBoosting": [],
        "XGBoost": [],
        "LightGBM": [],
        "Ensemble (LGB+XGB)": [],
        "Ensemble (Ridge+LGB+XGB)": [],
    }

    for task in eval_tasks:
        test_mask = df_tier["task_name"] == task
        train_mask = ~test_mask
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        train_med = X_train.median()
        X_train_f = X_train.fillna(train_med).fillna(0)
        X_test_f = X_test.fillna(train_med).fillna(0)

        preds = {}

        # Linear models
        for name, model in [
            ("Ridge", Ridge(alpha=1.0)),
            ("Lasso", Lasso(alpha=0.001, max_iter=5000)),
            ("ElasticNet", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000)),
        ]:
            model.fit(X_train_f, y_train)
            preds[name] = model.predict(X_test_f)
            model_scores[name].append(r2_score(y_test, preds[name]))

        # Random Forest
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_train_f, y_train)
        preds["RandomForest"] = rf.predict(X_test_f)
        model_scores["RandomForest"].append(r2_score(y_test, preds["RandomForest"]))

        # Gradient Boosting (sklearn)
        gb = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5
        )
        gb.fit(X_train_f, y_train)
        preds["GradientBoosting"] = gb.predict(X_test_f)
        model_scores["GradientBoosting"].append(
            r2_score(y_test, preds["GradientBoosting"])
        )

        # XGBoost
        X_train_xgb = X_train.copy()
        X_test_xgb = X_test.copy()
        for col in cat_cols:
            X_train_xgb[col] = X_train_xgb[col].astype("category")
            X_test_xgb[col] = X_test_xgb[col].astype("category")
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train_xgb, y_train)
        preds["XGBoost"] = xgb_model.predict(X_test_xgb)
        model_scores["XGBoost"].append(r2_score(y_test, preds["XGBoost"]))

        # LightGBM
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(X_train, y_train, categorical_feature=cat_indices)
        preds["LightGBM"] = lgb_model.predict(X_test)
        model_scores["LightGBM"].append(r2_score(y_test, preds["LightGBM"]))

        # Ensembles
        ens_2 = (preds["LightGBM"] + preds["XGBoost"]) / 2
        model_scores["Ensemble (LGB+XGB)"].append(r2_score(y_test, ens_2))

        ens_3 = (preds["Ridge"] + preds["LightGBM"] + preds["XGBoost"]) / 3
        model_scores["Ensemble (Ridge+LGB+XGB)"].append(r2_score(y_test, ens_3))

    # Results table
    print(f"\n{'='*70}")
    print(
        f"MODEL COMPARISON -- {tier_name} (LOT-O, {len(model_scores['Ridge'])} tasks)"
    )
    print(f"{'Model':<28} {'Mean R2':>10} {'Std R2':>10}")

    comparison_rows = []
    for name in model_scores:
        scores = model_scores[name]
        mean_r2 = np.mean(scores)
        std_r2 = np.std(scores)
        print(f"{name:<28} {mean_r2:>10.4f} {std_r2:>10.4f}")
        comparison_rows.append({"model": name, "mean_r2": mean_r2, "std_r2": std_r2})

    comp_df = pd.DataFrame(comparison_rows).sort_values("mean_r2", ascending=False)
    comp_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"\nSaved {comp_path}")

    # Bar chart
    comp_df_sorted = comp_df.sort_values("mean_r2", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [
        (
            "#009688"
            if "Ensemble" in m
            else (
                "#4CAF50"
                if m == "RandomForest"
                else "#2196F3" if m == "LightGBM" else "#9E9E9E"
            )
        )
        for m in comp_df_sorted["model"]
    ]
    ax.barh(
        range(len(comp_df_sorted)),
        comp_df_sorted["mean_r2"],
        xerr=comp_df_sorted["std_r2"],
        color=colors,
        capsize=3,
    )
    ax.set_yticks(range(len(comp_df_sorted)))
    ax.set_yticklabels(comp_df_sorted["model"], fontsize=9)
    ax.set_xlabel("Mean R2")
    ax.set_title(f"Model Comparison -- {tier_name}", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "model_comparison.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved {fig_path}")

    return comp_df


# ============================================================
# Leave-one-task-out evaluation (Ridge + RandomForest)
# ============================================================


def run_tier_evaluation(df, rf_params=None):
    """Run leave-one-task-out CV for all three tiers."""
    rf_params = rf_params or RF_PARAMS

    tier_results = {}

    for tier_name, features in TIERS.items():
        X, y, task_names = _prepare_tier(df, features)
        eval_tasks = sample_eligible_tasks(task_names, EVAL_N_TASKS, EVAL_SEED)

        ridge_scores, rf_scores = [], []

        for task in eval_tasks:
            test_mask = task_names == task
            train_mask = ~test_mask
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            train_med = X_train.median()
            X_train_f = X_train.fillna(train_med).fillna(0)
            X_test_f = X_test.fillna(train_med).fillna(0)

            # Ridge
            ridge = Ridge()
            ridge.fit(X_train_f, y_train)
            ridge_scores.append(r2_score(y_test, ridge.predict(X_test_f)))

            # RandomForest
            model = RandomForestRegressor(**rf_params)
            model.fit(X_train_f, y_train)
            rf_scores.append(r2_score(y_test, model.predict(X_test_f)))

        tier_results[tier_name] = {
            "n_features": len(features),
            "n_tasks": len(rf_scores),
            "ridge_mean": np.mean(ridge_scores),
            "ridge_std": np.std(ridge_scores),
            "rf_mean": np.mean(rf_scores),
            "rf_std": np.std(rf_scores),
        }
        print(
            f"{tier_name}: Ridge R2={np.mean(ridge_scores):.3f} (+/-{np.std(ridge_scores):.3f}), "
            f"RF R2={np.mean(rf_scores):.3f} (+/-{np.std(rf_scores):.3f})"
        )

    # Summary table
    print("THREE-TIER COMPARISON Leave-One-Task-Out")
    print(f"\n{'Tier':<25} {'#Feats':>7} {'Ridge R2':>12} {'RF R2':>12} {'#Tasks':>8}")
    for tier_name in TIERS:
        r = tier_results[tier_name]
        print(
            f"{tier_name:<25} {r['n_features']:>7} "
            f"{r['ridge_mean']:>8.3f}+/-{r['ridge_std']:.3f} "
            f"{r['rf_mean']:>8.3f}+/-{r['rf_std']:.3f} "
            f"{r['n_tasks']:>8}"
        )

    # Marginal gains
    tiers_list = list(TIERS.keys())
    t1 = tier_results[tiers_list[0]]
    t2 = tier_results[tiers_list[1]]
    t3 = tier_results[tiers_list[2]]
    print(f"\nMarginal contributions:")
    print(
        f"  README scraping:   d Ridge = {t2['ridge_mean']-t1['ridge_mean']:+.4f},  "
        f"d RF = {t2['rf_mean']-t1['rf_mean']:+.4f}"
    )
    print(
        f"  Popularity feats:  d Ridge = {t3['ridge_mean']-t2['ridge_mean']:+.4f},  "
        f"d RF = {t3['rf_mean']-t2['rf_mean']:+.4f}"
    )

    # Save summary
    summary_df = pd.DataFrame(tier_results).T
    summary_df.index.name = "tier"
    summary_path = os.path.join(OUTPUT_DIR, "tier_results.csv")
    summary_df.to_csv(summary_path)
    print(f"\nSaved {summary_path}")

    return tier_results


# ============================================================
# Per-family breakdown (RandomForest)
# ============================================================


def run_family_breakdown(df, rf_params=None, max_tasks_per_family=30):
    """Run leave-one-task-out R2 per task family for each tier."""
    rf_params = rf_params or RF_PARAMS
    family_tier_results = {tier_name: [] for tier_name in TIERS}

    for tier_name, features in TIERS.items():
        X, y, task_names = _prepare_tier(df, features)

        for task_family in df["task_type"].unique():
            family_task_names = task_names[df["task_type"] == task_family]
            family_tasks = sample_eligible_tasks(
                family_task_names, max_tasks_per_family, EVAL_SEED
            )
            if len(family_tasks) == 0:
                continue

            family_scores = []
            for task in family_tasks:
                test_mask = task_names == task
                train_mask = ~test_mask
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]

                train_med = X_train.median()
                X_train_f = X_train.fillna(train_med).fillna(0)
                X_test_f = X_test.fillna(train_med).fillna(0)

                model = RandomForestRegressor(**rf_params)
                model.fit(X_train_f, y_train)
                family_scores.append(r2_score(y_test, model.predict(X_test_f)))

            if family_scores:
                family_tier_results[tier_name].append(
                    {
                        "task_family": task_family,
                        "n_tasks": len(family_tasks),
                        "mean_r2": np.mean(family_scores),
                        "std_r2": np.std(family_scores),
                    }
                )
            print(
                f"  {tier_name} / {task_family}: {len(family_scores)} tasks, R2={np.mean(family_scores):.3f}"
            )
        print(f"Done: {tier_name}")

    # Build comparison table
    tier_dfs = {}
    for tier_name in TIERS:
        tier_dfs[tier_name] = pd.DataFrame(family_tier_results[tier_name]).set_index(
            "task_family"
        )

    print("\n" + "=" * 90)
    print("PER-FAMILY R2 -- THREE TIERS (RandomForest)")
    print("=" * 90)
    tiers_list = list(TIERS.keys())
    print(
        f"\n{'Task Family':<22} {'Tier 1':>12} {'Tier 2':>12} {'Tier 3':>12}  "
        f"{'d(2-1)':>8} {'d(3-2)':>8}"
    )
    print("-" * 82)

    families_order = (
        tier_dfs[tiers_list[0]].sort_values("mean_r2", ascending=False).index
    )
    comparison_rows = []
    for fam in families_order:
        r1 = (
            tier_dfs[tiers_list[0]].loc[fam, "mean_r2"]
            if fam in tier_dfs[tiers_list[0]].index
            else np.nan
        )
        r2 = (
            tier_dfs[tiers_list[1]].loc[fam, "mean_r2"]
            if fam in tier_dfs[tiers_list[1]].index
            else np.nan
        )
        r3 = (
            tier_dfs[tiers_list[2]].loc[fam, "mean_r2"]
            if fam in tier_dfs[tiers_list[2]].index
            else np.nan
        )
        d21 = r2 - r1 if not (np.isnan(r1) or np.isnan(r2)) else np.nan
        d32 = r3 - r2 if not (np.isnan(r2) or np.isnan(r3)) else np.nan
        print(
            f"{fam:<22} {r1:>12.3f} {r2:>12.3f} {r3:>12.3f}  {d21:>+8.4f} {d32:>+8.4f}"
        )
        row = {"task_family": fam}
        for tier_name, short in zip(tiers_list, ["tier1", "tier2", "tier3"]):
            if fam in tier_dfs[tier_name].index:
                row[f"{short}_r2"] = tier_dfs[tier_name].loc[fam, "mean_r2"]
                row[f"{short}_std"] = tier_dfs[tier_name].loc[fam, "std_r2"]
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(TIER_COMPARISON_PATH, index=False)
    print(f"\nSaved {TIER_COMPARISON_PATH}")

    return tier_dfs, families_order


# ============================================================
# Ranking-oriented candidate analysis (Tier 2 RandomForest)
# ============================================================


def run_candidate_analysis(
    df,
    rf_params=None,
    tier_name="Tier 2 (+ README)",
    top_k_values=(1, 3, 5, 10),
):
    """Measure how well the top predicted candidates recover the true best model."""
    rf_params = rf_params or RF_PARAMS
    features = TIERS[tier_name]
    cat_cols = [
        c for c in features if df[c].dtype == "object" or c in TIER1_CATEGORICALS
    ]

    df_tier = df[["model_name", "task_name", TARGET] + features].copy()
    for col in cat_cols:
        df_tier[col] = LabelEncoder().fit_transform(df_tier[col].astype(str))

    X = df_tier[features]
    y = df_tier[TARGET]
    task_names = df_tier["task_name"]
    eval_tasks = sample_eligible_tasks(task_names, EVAL_N_TASKS, EVAL_SEED)

    per_task_rows = []
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
        preds = model.predict(X_test_f)

        task_df = df_tier.loc[test_mask, ["model_name"]].copy()
        task_df["y_true"] = y_test.to_numpy()
        task_df["y_pred"] = preds
        task_df = task_df.sort_values("y_pred", ascending=False).reset_index(drop=True)

        best_true_idx = task_df["y_true"].idxmax()
        best_true = task_df.loc[best_true_idx, "y_true"]
        top_pick_true = task_df.loc[0, "y_true"]
        top_pick_true_rank = (
            task_df["y_true"].rank(ascending=False, method="min").iloc[0]
        )
        winner_pred_rank = int(best_true_idx) + 1

        row = {
            "task_name": task,
            "n_models": len(task_df),
            "winner_model": task_df.loc[best_true_idx, "model_name"],
            "winner_pred_rank": winner_pred_rank,
            "top_pick_model": task_df.loc[0, "model_name"],
            "top_pick_true_norm_rank": float(top_pick_true),
            "top_pick_true_rank": int(top_pick_true_rank),
            "top_pick_regret": float(best_true - top_pick_true),
        }

        for k in top_k_values:
            top_candidates = task_df.head(min(k, len(task_df)))
            best_in_top_k = top_candidates["y_true"].max()
            row[f"hit_at_{k}"] = int(winner_pred_rank <= k)
            row[f"best_norm_rank_in_top_{k}"] = float(best_in_top_k)
            row[f"regret_at_{k}"] = float(best_true - best_in_top_k)

        per_task_rows.append(row)

    per_task_df = pd.DataFrame(per_task_rows)
    per_task_path = os.path.join(OUTPUT_DIR, "candidate_ranking_per_task.csv")
    per_task_df.to_csv(per_task_path, index=False)
    print(f"Saved {per_task_path}")

    summary_rows = [
        {"metric": "n_tasks", "value": float(len(per_task_df))},
        {
            "metric": "mean_top_pick_true_norm_rank",
            "value": float(per_task_df["top_pick_true_norm_rank"].mean()),
        },
        {
            "metric": "mean_top_pick_true_rank",
            "value": float(per_task_df["top_pick_true_rank"].mean()),
        },
        {
            "metric": "mean_top_pick_regret",
            "value": float(per_task_df["top_pick_regret"].mean()),
        },
    ]
    for k in top_k_values:
        summary_rows.extend(
            [
                {
                    "metric": f"hit_rate_at_{k}",
                    "value": float(per_task_df[f"hit_at_{k}"].mean()),
                },
                {
                    "metric": f"mean_regret_at_{k}",
                    "value": float(per_task_df[f"regret_at_{k}"].mean()),
                },
            ]
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "candidate_ranking_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    print("\nCandidate retrieval summary:")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    hit_rates = [per_task_df[f"hit_at_{k}"].mean() for k in top_k_values]
    ax.bar([str(k) for k in top_k_values], hit_rates, color="#4CAF50", alpha=0.85)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Top-k candidate list size")
    ax.set_ylabel("Hit rate for true best model")
    ax.set_title("How Often the True Best Model Appears in Top-k")
    ax.grid(axis="y", alpha=0.3)
    for idx, value in enumerate(hit_rates):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "candidate_hit_rate.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved {fig_path}")

    return per_task_df, summary_df


# ============================================================
# Visualization
# ============================================================


def plot_tier_comparison(tier_results, tier_dfs, families_order):
    """Generate the two-panel tier comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Panel 1: Grouped bar chart -- R2 per family per tier ---
    families_sorted = list(families_order)
    n_families = len(families_sorted)
    x = np.arange(n_families)
    width = 0.25

    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    tiers_list = list(TIERS.keys())

    for i, (tier_name, color) in enumerate(zip(tiers_list, colors)):
        vals = []
        for fam in families_sorted:
            if fam in tier_dfs[tier_name].index:
                vals.append(tier_dfs[tier_name].loc[fam, "mean_r2"])
            else:
                vals.append(0)
        axes[0].bar(
            x + i * width, vals, width, label=tier_name, color=color, alpha=0.85
        )

    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(families_sorted, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("Mean R2")
    axes[0].set_title("R2 by Task Family -- Three Feature Tiers", fontsize=12)
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, max(0.9, axes[0].get_ylim()[1]))

    # --- Panel 2: Summary bar chart -- overall R2 per tier ---
    tier_names_short = [
        "Tier 1\n(API-only)",
        "Tier 2\n(+ README)",
        "Tier 3\n(+ discarded)",
    ]
    ridge_vals = [tier_results[t]["ridge_mean"] for t in tiers_list]
    rf_vals = [tier_results[t]["rf_mean"] for t in tiers_list]
    ridge_std = [tier_results[t]["ridge_std"] for t in tiers_list]
    rf_std = [tier_results[t]["rf_std"] for t in tiers_list]

    x2 = np.arange(3)
    axes[1].bar(
        x2 - 0.15,
        ridge_vals,
        0.3,
        yerr=ridge_std,
        label="Ridge",
        color="#9C27B0",
        alpha=0.8,
        capsize=4,
    )
    axes[1].bar(
        x2 + 0.15,
        rf_vals,
        0.3,
        yerr=rf_std,
        label="RandomForest",
        color="#4CAF50",
        alpha=0.8,
        capsize=4,
    )
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(tier_names_short, fontsize=10)
    axes[1].set_ylabel("Mean R2")
    axes[1].set_title("Overall R2 Ridge vs RandomForest per Tier", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "tier_comparison.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved {fig_path}")


# ============================================================
# Configuration
# ============================================================

RUN_MODEL_COMPARISON = False  # Set True to re-run model comparison
RUN_TIER_EVALUATION = False  # Set True to re-run tier eval + family breakdown
RUN_CANDIDATE_ANALYSIS = False  # Set True to re-run candidate ranking analysis


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    df = pd.read_csv(TRAINING_DATA_PATH)

    best_params = RF_PARAMS.copy()
    best_params["random_state"] = EVAL_SEED

    comp_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")

    if RUN_MODEL_COMPARISON or not os.path.exists(comp_path):
        print("\n--- Model Comparison ---")
        comp_df = run_model_comparison(df, rf_params=best_params)
    else:
        comp_df = pd.read_csv(comp_path)
        print(f"\nLoaded model comparison from {comp_path}")

    best_model = comp_df.iloc[0]["model"]
    best_r2 = comp_df.iloc[0]["mean_r2"]
    print(f"Best model: {best_model} (R2={best_r2:.4f})")

    tier_path = os.path.join(OUTPUT_DIR, "tier_results.csv")

    if (
        RUN_TIER_EVALUATION
        or not os.path.exists(tier_path)
        or not os.path.exists(TIER_COMPARISON_PATH)
    ):
        print("\n Leave-One-Task-Out Evaluation")
        tier_results = run_tier_evaluation(df, rf_params=best_params)

        print("\n Per-Family Breakdown")
        tier_dfs, families_order = run_family_breakdown(df, rf_params=best_params)
    else:
        tier_summary = pd.read_csv(tier_path, index_col=0)
        tier_results = tier_summary.to_dict(orient="index")
        print(f"\nLoaded tier results from {tier_path}")

        comparison_df = pd.read_csv(TIER_COMPARISON_PATH)
        tier_dfs = {}
        tiers_list = list(TIERS.keys())
        for tier_name, short in zip(tiers_list, ["tier1", "tier2", "tier3"]):
            cols = {
                "task_family": "task_family",
                f"{short}_r2": "mean_r2",
                f"{short}_std": "std_r2",
            }
            sub = comparison_df[["task_family", f"{short}_r2", f"{short}_std"]].dropna()
            sub = sub.rename(
                columns={f"{short}_r2": "mean_r2", f"{short}_std": "std_r2"}
            )
            tier_dfs[tier_name] = sub.set_index("task_family")
        families_order = (
            tier_dfs[tiers_list[0]].sort_values("mean_r2", ascending=False).index
        )
        print(f"Loaded family breakdown from {TIER_COMPARISON_PATH}")

    candidate_summary_path = os.path.join(OUTPUT_DIR, "candidate_ranking_summary.csv")
    if RUN_CANDIDATE_ANALYSIS or not os.path.exists(candidate_summary_path):
        print("\n Candidate Ranking Analysis")
        _, candidate_summary = run_candidate_analysis(df, rf_params=best_params)
    else:
        candidate_summary = pd.read_csv(candidate_summary_path)
        print(f"Loaded candidate ranking summary from {candidate_summary_path}")

    print("\n Visualization")
    plot_tier_comparison(tier_results, tier_dfs, families_order)
