"""Helpers for training and using frozen demo inference models."""

import os
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config import OUTPUT_DIR, RF_PARAMS, TIER1_CATEGORICALS, TIERS
from src.utils import encode_booleans

DEMO_MODEL_DIR = os.path.join(OUTPUT_DIR, "demo_models")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_HOLDOUT_MODELS_PATH = os.path.join(REPO_ROOT, "demo_holdout_models.txt")


def slugify_name(value):
    """Convert a tier or family name into a file-safe suffix."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def get_demo_artifact_path(tier_name):
    """Build the default path for a frozen demo model artifact."""
    os.makedirs(DEMO_MODEL_DIR, exist_ok=True)
    tier_slug = slugify_name(tier_name)
    filename = f"frozen_demo_model_{tier_slug}.pkl"
    return os.path.join(DEMO_MODEL_DIR, filename)


def load_demo_holdout_models(path=DEMO_HOLDOUT_MODELS_PATH):
    """Load the default list of model IDs excluded from frozen demo training."""
    if not os.path.exists(path):
        return []

    holdout_models = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            model_id = line.strip()
            if not model_id or model_id.startswith("#"):
                continue
            holdout_models.append(model_id)
    return holdout_models


def _prepare_feature_frame(df, features):
    """Ensure feature columns exist and booleans are encoded."""
    work_df = encode_booleans(df.copy())
    for col in features:
        if col not in work_df.columns:
            work_df[col] = np.nan
    return work_df


def _build_category_mappings(df, features):
    """Build deterministic integer mappings for categorical columns."""
    mappings = {}
    cat_cols = [
        col
        for col in features
        if col in TIER1_CATEGORICALS or df[col].dtype == "object"
    ]

    for col in cat_cols:
        values = df[col].astype(str).fillna("nan")
        unique_values = sorted(pd.Index(values).unique().tolist())
        mapping = {value: idx for idx, value in enumerate(unique_values)}
        mapping["__UNK__"] = len(mapping)
        mappings[col] = mapping

    return mappings


def _apply_category_mappings(df, mappings):
    """Apply saved categorical mappings, routing unseen values to __UNK__."""
    work_df = df.copy()
    for col, mapping in mappings.items():
        unk_value = mapping["__UNK__"]
        work_df[col] = (
            work_df[col].astype(str).map(lambda value: mapping.get(value, unk_value))
        )
    return work_df


def build_training_matrix(df, features):
    """Encode and fill a dataframe, returning the matrix and preprocessing state."""
    work_df = _prepare_feature_frame(df, features)
    mappings = _build_category_mappings(work_df, features)
    work_df = _apply_category_mappings(work_df, mappings)

    matrix = work_df[features].copy()
    fill_values = matrix.median(numeric_only=True)
    matrix = matrix.fillna(fill_values).fillna(0)
    return matrix, mappings, fill_values


def transform_with_artifact(df, artifact):
    """Transform inference rows with the preprocessing state saved in an artifact."""
    features = artifact["features"]
    work_df = _prepare_feature_frame(df, features)
    work_df = _apply_category_mappings(work_df, artifact["category_mappings"])
    matrix = work_df[features].copy()
    matrix = matrix.fillna(artifact["fill_values"]).fillna(0)
    return matrix


def train_frozen_demo_artifact(
    train_df, tier_name, rf_params=None, excluded_models=None
):
    """Train one frozen RandomForest artifact for fast demo inference."""
    features = TIERS[tier_name]
    working_df = train_df.copy()
    excluded_models = sorted(set(excluded_models or []))

    if excluded_models:
        working_df = working_df[~working_df["model_name"].isin(excluded_models)].copy()

    if working_df.empty:
        raise ValueError(
            "No training rows available for the requested frozen demo model."
        )

    X_train, mappings, fill_values = build_training_matrix(working_df, features)
    y_train = working_df["norm_rank"].to_numpy()

    params = (rf_params or RF_PARAMS).copy()
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    return {
        "model": model,
        "tier_name": tier_name,
        "features": features,
        "category_mappings": mappings,
        "fill_values": fill_values,
        "n_rows": int(len(working_df)),
        "n_tasks": int(working_df["task_name"].nunique()),
        "n_models": int(working_df["model_name"].nunique()),
        "excluded_models": excluded_models,
        "trained_model_names": sorted(
            working_df["model_name"].dropna().unique().tolist()
        ),
    }


def save_demo_artifact(artifact, artifact_path):
    """Persist a frozen demo model artifact to disk."""
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    with open(artifact_path, "wb") as handle:
        pickle.dump(artifact, handle)


def load_demo_artifact(artifact_path):
    """Load a frozen demo model artifact from disk."""
    with open(artifact_path, "rb") as handle:
        return pickle.load(handle)
