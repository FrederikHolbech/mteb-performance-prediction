"""
Central configuration for the MTEB performance prediction pipeline.
All feature definitions, tier structures, and shared constants live here.
"""

import os

# ============================================================
# PATHS
# ============================================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# Input data
UNICODE_SCRIPTS_PATH = os.path.join(ROOT_DIR, "inputEncodings", "data", "Scripts.txt")
SEQUENCES_PATH = os.path.join(ROOT_DIR, "inputEncodings", "data", "1.sequences.txt")

# Intermediate data (produced and consumed by pipeline)
MTEB_RESULTS_PATH = os.path.join(DATA_DIR, "mteb_results_clean.csv")
MODEL_METADATA_PATH = os.path.join(DATA_DIR, "model_metadata.csv")
MODEL_METADATA_ENRICHED_PATH = os.path.join(DATA_DIR, "model_metadata_enriched.csv")
TASK_METADATA_PATH = os.path.join(DATA_DIR, "task_metadata.csv")
TOKENIZER_FEATURES_PATH = os.path.join(DATA_DIR, "tokenizer_features.csv")
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "training_data.csv")

# Output CSVs
TIER_COMPARISON_PATH = os.path.join(OUTPUT_DIR, "tier_comparison_by_family.csv")
SHAP_TIER1_PATH = os.path.join(OUTPUT_DIR, "shap_by_family_tier_1.csv")
SHAP_TIER2_PATH = os.path.join(OUTPUT_DIR, "shap_by_family_tier_2.csv")
SHAP_TIER3_PATH = os.path.join(OUTPUT_DIR, "shap_by_family_tier_3.csv")

# ============================================================
# MTEB TASK FAMILIES
# ============================================================

TEXT_TASK_FAMILIES = [
    "Retrieval",
    "Classification",
    "Clustering",
    "STS",
    "PairClassification",
    "BitextMining",
    "Reranking",
]
# Summarization excluded — only 1 task in MTEB

# ============================================================
# MODEL FAMILY MAPPING
# ============================================================

FAMILY_PATTERNS = {
    "bert-base": "bert",
    "bert-large": "bert",
    "/bert": "bert",
    "roberta": "roberta",
    "xlm-roberta": "roberta",
    "mpnet": "mpnet",
    "all-mpnet": "mpnet",
    "t5": "t5",
    "sentence-t5": "t5",
    "gtr-t5": "t5",
    "mistral": "mistral",
    "e5-mistral": "mistral",
    "llama": "llama",
    "qwen": "qwen",
    "gemma": "gemma",
    "nomic": "nomic",
    "jina": "jina",
    "gte": "gte",
    "bge": "bge",
    "modernbert": "modernbert",
}

# ============================================================
# KEYWORD DICTIONARIES (for README scraping)
# ============================================================

TRAINING_KEYWORDS = {
    "train_msmarco": [r"msmarco", r"ms[\s-]?marco"],
    "train_nli": [
        r"\bnli\b",
        r"natural language inference",
        r"snli",
        r"multinli",
        r"allnli",
    ],
    "train_multilingual": [r"multilingual", r"multi-lingual", r"parallel.+corpus"],
    "train_retrieval": [r"retrieval", r"search", r"passage.+ranking"],
    "train_sts": [r"\bsts\b", r"semantic textual similarity"],
    "train_classification": [r"classification", r"sentiment", r"topic"],
    "train_wikipedia": [r"wikipedia", r"wiki\b"],
    "train_commoncrawl": [r"common.?crawl", r"cc[\s-]?news", r"\bc4\b"],
    "train_squad": [r"\bsquad\b", r"question.?answer"],
}

METHOD_KEYWORDS = {
    "is_instruction_tuned": [r"instruct", r"instruction[\s-]?tun"],
    "uses_matryoshka": [r"matryoshka", r"\bmrl\b"],
    "is_distilled": [r"distil", r"knowledge.+distillation", r"teacher.+student"],
    "loss_contrastive": [
        r"contrastive",
        r"infonce",
        r"multiple.+negatives",
        r"mnrl",
        r"cosine.+loss",
    ],
    "loss_triplet": [r"triplet"],
}

# ============================================================
# FEATURE TIER DEFINITIONS
# ============================================================

# Boolean columns that need True/False -> 1/0 encoding
BOOLEAN_COLUMNS = [
    "is_multilingual",
    "is_english",
    "domain_web",
    "domain_academic",
    "domain_medical",
    "domain_legal",
    "domain_news",
    "domain_social",
    "domain_code",
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
    "has_dense_layer",
    "uses_gqa",
    "is_lowercased",
    "strips_accents",
]

# --- TIER 1: API-only features (config.json, tokenizer.json, HF API, MTEB) ---
TIER1_MODEL_NUMERIC = [
    "param_count",
    "hidden_size",
    "num_layers",
    "num_attention_heads",
    "max_position_embeddings",
    "vocab_size",
    "intermediate_size",
    "embedding_dim",
    "max_seq_length",
    # Derived from architecture
    "head_dim",
    "ffn_ratio",
    "log_param_count",
    "embedding_ratio",
    # Tokenizer features (from tokenizer.json)
    "actual_vocab_size",
    "avg_subword_len_chars",
    "avg_subword_len_bytes",
    "median_subword_len_chars",
    "pct_continuation_tokens",
    "num_scripts_in_vocab",
    "pct_latin_tokens",
    "pct_cjk_tokens",
    "embed_param_ratio",
    "is_lowercased",
    "strips_accents",
]

TIER1_TASK_NUMERIC = [
    "num_languages",
    "is_multilingual",
    "is_english",
    "num_domains",
    "domain_web",
    "domain_academic",
    "domain_medical",
    "domain_legal",
    "domain_news",
    "domain_social",
    "domain_code",
    "num_subtypes",
    "num_samples_test",
    "log_num_samples",
    "avg_text_length",
    "unique_labels",
]

TIER1_CATEGORICALS = [
    "task_type",
    "pipeline_tag",
    "library_name",
    "model_type",
    "base_model_family",
    "pooling_mode",
    "hidden_act",
    "position_embedding_type",
    "tokenizer_class",
    "main_score",
    "annotations_creators",
    "tokenizer_main_model",
    "pre_tokenizer_chain",
    "normalizer_type",
]

TIER1_FEATURES = TIER1_MODEL_NUMERIC + TIER1_TASK_NUMERIC + TIER1_CATEGORICALS

# --- TIER 2: + README-scraped features ---
README_FEATURES = [
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

TIER2_FEATURES = TIER1_FEATURES + README_FEATURES

# --- TIER 3: + Discarded popularity features ---
DISCARDED_FEATURES = ["downloads", "likes", "model_age_days"]

TIER3_FEATURES = TIER2_FEATURES + DISCARDED_FEATURES

TIERS = {
    "Tier 1 (API-only)": TIER1_FEATURES,
    "Tier 2 (+ README)": TIER2_FEATURES,
    "Tier 3 (+ discarded)": TIER3_FEATURES,
}

# ============================================================
# MODEL HYPERPARAMETERS
# ============================================================

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "n_jobs": -1,
}

# Grid search space for RandomForest hyperparameter tuning
RF_GRID = {
    "n_estimators": [200, 500, 1000],
    "max_depth": [15, 30, None],
    "min_samples_leaf": [1, 2, 5],
}

EVAL_SEED = 42
EVAL_N_TASKS = 50
TARGET = "norm_rank"
