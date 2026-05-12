"""
Shared utility functions for the MTEB performance prediction pipeline.
"""

import ast
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import BOOLEAN_COLUMNS, FAMILY_PATTERNS

# ============================================================
# Feature label display mapping
# ============================================================

FEATURE_DISPLAY_NAMES = {
    "pct_continuation_tokens": "cont_token_pct",
    "pre_tokenizer_chain": "pretokenizer",
    "avg_subword_len_chars": "avg_subword_chars",
    "avg_subword_len_bytes": "avg_subword_bytes",
    "median_subword_len_chars": "median_subword_chars",
    "num_scripts_in_vocab": "scripts_in_vocab",
    "tokenizer_main_model": "tokenizer_model",
    "tokenizer_class": "tokenizer_cls",
    "normalizer_type": "normalizer",
    "position_embedding_type": "pos_emb_type",
    "num_attention_heads": "attn_heads",
    "intermediate_size": "ffn_size",
    "embedding_dim": "emb_dim",
    "hidden_size": "hidden_dim",
    "num_layers": "layers",
    "num_languages": "n_languages",
    "num_domains": "n_domains",
    "num_subtypes": "n_subtypes",
    "num_samples_test": "n_test_samples",
    "avg_text_length": "avg_text_len",
    "annotations_creators": "ann_creators",
    "base_model_family": "base_family",
    "is_instruction_tuned": "instr_tuned",
    "is_multilingual": "multilingual",
    "is_english": "english_task",
    "model_type": "model_arch",
    "pooling_mode": "pooling",
    "main_score": "task_metric",
    "uses_matryoshka": "matryoshka",
    "param_count": "params",
    "max_position_embeddings": "max_pos_emb",
    "max_seq_length": "max_seq_len",
    "actual_vocab_size": "actual_vocab",
    "vocab_size": "vocab_size",
    "downloads": "downloads",
    "model_age_days": "model_age",
    "num_datasets_listed": "n_datasets",
    "loss_contrastive": "contrastive_loss",
    "loss_triplet": "triplet_loss",
    "train_commoncrawl": "train_cc",
    "train_classification": "train_clf",
    "train_multilingual": "train_multi",
    "train_retrieval": "train_retr",
    "train_wikipedia": "train_wiki",
}


def shorten_feature_name(feature_name):
    """Return a compact display label for a feature name."""
    return FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)


def sample_eligible_tasks(task_names, max_tasks, seed, min_rows=5):
    """Sample tasks after filtering to folds with enough observations."""
    task_counts = task_names.value_counts()
    eligible_tasks = task_counts[task_counts >= min_rows].index.to_numpy()

    if len(eligible_tasks) == 0:
        return eligible_tasks

    rng = np.random.default_rng(seed)
    sample_size = min(max_tasks, len(eligible_tasks))
    return rng.choice(eligible_tasks, size=sample_size, replace=False)


# ============================================================
# Data loading & encoding
# ============================================================


def encode_booleans(df, columns=None):
    """Encode boolean columns from string/bool to 0/1 integers."""
    columns = columns or BOOLEAN_COLUMNS
    for col in columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .map({"True": 1, "False": 0, True: 1, False: 0})
                .fillna(0)
                .astype(int)
            )
    return df


def encode_categoricals(df, cat_features):
    """Label-encode categorical features in-place."""
    for col in cat_features:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def parse_list(s):
    """Parse a string representation of a list to an actual list."""
    try:
        return ast.literal_eval(str(s))
    except (ValueError, SyntaxError):
        return []


# ============================================================
# Model family classification
# ============================================================


def classify_family(name_or_path, model_type, model_name):
    """Map a model to a base-model family bucket."""
    search_str = f"{name_or_path or ''} {model_type or ''} {model_name or ''}".lower()
    for pattern, family in FAMILY_PATTERNS.items():
        if pattern.lower() in search_str:
            return family
    return "other"


# ============================================================
# Keyword extraction from README
# ============================================================


def extract_keyword_flags(readme_text, keyword_dict):
    """Flag presence of keywords in README text."""
    flags = {}
    text_lower = readme_text.lower() if readme_text else ""
    for flag_name, patterns in keyword_dict.items():
        flags[flag_name] = any(re.search(p, text_lower) for p in patterns)
    return flags


def count_yaml_datasets(readme_text):
    """Count datasets listed in README YAML frontmatter."""
    if not readme_text or not readme_text.strip().startswith("---"):
        return 0
    yaml_end = readme_text.find("---", 3)
    if yaml_end < 0:
        return 0
    frontmatter = readme_text[3:yaml_end]
    match = re.search(r"^datasets:\s*\n((?:\s*-\s+.+\n)*)", frontmatter, re.MULTILINE)
    if match:
        return len(re.findall(r"^\s*-\s+", match.group(0), re.MULTILINE))
    return 0


# ============================================================
# Pooling & position embedding inference
# ============================================================


def extract_pooling_mode(pool_cfg):
    """Determine pooling strategy from 1_Pooling/config.json."""
    if not pool_cfg:
        return "unknown"
    mapping = [
        ("pooling_mode_mean_tokens", "mean"),
        ("pooling_mode_cls_token", "cls"),
        ("pooling_mode_lasttoken", "lasttoken"),
        ("pooling_mode_max_tokens", "max"),
        ("pooling_mode_weightedmean_tokens", "weightedmean"),
        ("pooling_mode_mean_sqrt_len_tokens", "mean_sqrt_len"),
    ]
    for key, mode in mapping:
        if pool_cfg.get(key):
            return mode
    return "other"


def infer_position_embedding_type(config):
    """Infer position embedding type from config.json keys."""
    pet = config.get("position_embedding_type")
    if pet:
        return pet
    if config.get("rope_theta") or config.get("rope_scaling"):
        return "rotary"
    if config.get("alibi"):
        return "alibi"
    return "unknown"


# ============================================================
# Task feature extraction
# ============================================================


def extract_task_features(row):
    """Extract task features from a task metadata row."""
    languages = (
        row["languages"]
        if isinstance(row["languages"], list)
        else parse_list(row["languages"])
    )
    domains = (
        row["domains"]
        if isinstance(row["domains"], list)
        else parse_list(row["domains"])
    )
    subtypes = (
        row["task_subtypes"]
        if isinstance(row["task_subtypes"], list)
        else parse_list(row["task_subtypes"])
    )

    return {
        "task_name": row["task_name"],
        "task_type": row["task_type"],
        "num_languages": len(languages),
        "is_multilingual": len(languages) > 1,
        "is_english": "eng" in [str(l) for l in languages],
        "num_domains": len(domains),
        "domain_web": any("Web" in str(d) for d in domains),
        "domain_academic": any("Academic" in str(d) for d in domains),
        "domain_medical": any("Medical" in str(d) for d in domains),
        "domain_legal": any("Legal" in str(d) for d in domains),
        "domain_news": any("News" in str(d) for d in domains),
        "domain_social": any("Social" in str(d) for d in domains),
        "domain_code": any("Programming" in str(d) for d in domains),
        "num_subtypes": len(subtypes),
    }


# ============================================================
# Unicode script lookup (for tokenizer analysis)
# ============================================================


def load_unicode_scripts(scripts_path):
    """Load Unicode script ranges from Scripts.txt into a lookup array."""
    script_ranges = [None] * 918000
    with open(scripts_path, encoding="utf-8") as f:
        for line in f:
            tok = line.split(";")
            if line[0] != "#" and len(tok) == 2:
                char_range_hex = tok[0].strip().split("..")
                char_range_int = [int(x, 16) for x in char_range_hex]
                script_name = tok[1].strip().split()[0]
                if len(char_range_int) == 1:
                    script_ranges[char_range_int[0]] = script_name
                else:
                    for ind in range(char_range_int[0], char_range_int[1] + 1):
                        script_ranges[ind] = script_name
    return script_ranges


def get_script(char, script_ranges):
    """Get the Unicode script for a character."""
    idx = ord(char)
    return script_ranges[idx] if idx < len(script_ranges) else None
