"""
Step 2: Build the training data matrix.

Merges MTEB results, model metadata (enriched), task metadata, and
tokenizer features into a single training CSV with all features.

Reads:
  data/mteb_results_clean.csv
  data/model_metadata_enriched.csv
  data/task_metadata.csv
  data/tokenizer_features.csv

Outputs:
  data/training_data.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DATA_DIR,
    TEXT_TASK_FAMILIES,
    MTEB_RESULTS_PATH,
    MODEL_METADATA_ENRICHED_PATH,
    TASK_METADATA_PATH,
    TOKENIZER_FEATURES_PATH,
    TRAINING_DATA_PATH,
)
from src.utils import encode_booleans


def build_training_data():
    """Merge all data sources into a single training matrix."""

    # --- Load MTEB results ---
    results = pd.read_csv(MTEB_RESULTS_PATH)
    print(f"MTEB results: {results.shape}")

    # --- Load model metadata ---
    meta = pd.read_csv(MODEL_METADATA_ENRICHED_PATH)
    model_cols = [
        "model_name",
        # Base metadata
        "downloads",
        "likes",
        "created_at",
        "model_age_days",
        "pipeline_tag",
        "library_name",
        "model_type",
        "param_count",
        "hidden_size",
        "num_layers",
        "num_attention_heads",
        "max_position_embeddings",
        "vocab_size",
        "intermediate_size",
        # Enrichment
        "embedding_dim",
        "pooling_mode",
        "max_seq_length",
        "has_dense_layer",
        "base_model_family",
        "hidden_act",
        "position_embedding_type",
        "uses_gqa",
        "tokenizer_class",
        # README-derived
        "train_msmarco",
        "train_nli",
        "train_multilingual",
        "train_retrieval",
        "train_sts",
        "train_classification",
        "train_wikipedia",
        "train_commoncrawl",
        "train_squad",
        "is_instruction_tuned",
        "uses_matryoshka",
        "is_distilled",
        "loss_contrastive",
        "loss_triplet",
        "num_datasets_listed",
    ]
    model_cols = [c for c in model_cols if c in meta.columns]
    model_data = meta[model_cols].copy()

    # --- Load task metadata ---
    tasks = pd.read_csv(TASK_METADATA_PATH)
    print(f"Task metadata: {tasks.shape}")

    # --- Merge results + model + task ---
    df = results.merge(model_data, on="model_name", how="left")
    df = df.merge(tasks, on=["task_name", "task_type"], how="left")

    # --- Filter to text task families (exclude Summarization) ---
    df = df[df["task_type"].isin(TEXT_TASK_FAMILIES)].copy()

    # --- Compute normalized rank per task ---
    df["norm_rank"] = df.groupby("task_name")["score"].rank(pct=True)

    if "created_at" in df.columns:
        df = df.drop(columns=["created_at"])

    # --- Merge tokenizer features ---
    if os.path.exists(TOKENIZER_FEATURES_PATH):
        tok = pd.read_csv(TOKENIZER_FEATURES_PATH)
        tok_cols = [
            "model_name",
            "tokenizer_main_model",
            "pre_tokenizer_chain",
            "normalizer_type",
            "actual_vocab_size",
            "avg_subword_len_chars",
            "avg_subword_len_bytes",
            "median_subword_len_chars",
            "pct_continuation_tokens",
            "num_scripts_in_vocab",
            "pct_latin_tokens",
            "pct_cjk_tokens",
        ]
        tok_cols = [c for c in tok_cols if c in tok.columns]
        df = df.merge(tok[tok_cols], on="model_name", how="left")

        # Boolean tokenizer features
        for col in ["is_lowercased", "strips_accents"]:
            if col in tok.columns:
                df = df.merge(tok[["model_name", col]], on="model_name", how="left")
                df[col] = df[col].fillna(0).astype(int)
        print(f"Merged tokenizer features ({len(tok_cols) - 1} columns)")
    else:
        print(
            "WARNING: tokenizer_features.csv not found — run 02_extract_tokenizer_features.py first"
        )

    # --- Encode booleans ---
    df = encode_booleans(df)

    # --- Save ---
    df.to_csv(TRAINING_DATA_PATH, index=False)
    print(f"\nTraining data: {df.shape}")
    print(f"Models: {df['model_name'].nunique()}, Tasks: {df['task_name'].nunique()}")
    print(f"Task families: {sorted(df['task_type'].unique())}")
    print(f"\nNull rates (top 10):")
    print(
        (df.isnull().sum() / len(df) * 100)
        .round(1)
        .sort_values(ascending=False)
        .head(10)
    )
    return df


if __name__ == "__main__":
    build_training_data()
