# Predicting MTEB Performance from Model & Task Features

This repository contains the code for the bachelor thesis on **predicting MTEB benchmark performance** using model architecture, tokenizer, task metadata, and training-data features.

## Project Structure

```
├── src/                  # Shared code
│   ├── config.py         # All paths, feature tiers, constants, hyperparameters
│   └── utils.py          # Utility functions (encoding, keyword extraction, etc.)
├── scripts/              # Reproducible pipeline (run in order)
│   ├── 01_collect_data.py              # Fetch MTEB results + HF model/task metadata
│   ├── 02_build_training_data.py       # Merge into single training matrix
│   ├── 03_extract_tokenizer_features.py # Extract tokenizer-level features
│   ├── 04_train_and_evaluate.py        # Three-tier leave-one-task-out evaluation
│   └── 05_shap_analysis.py            # Per-family SHAP analysis across tiers
├── inputEncodings/       # Unicode script data (for tokenizer analysis)
│   └── data/
│       ├── Scripts.txt           # Unicode script ranges
│       └── 1.sequences.txt       # Pre-computed tokenizer pipeline data (optional)
├── data/                 # Generated data (created by scripts)
├── output/               # Results, CSVs, and figures
│   └── figures/
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the Pipeline

Run the scripts in order. Each script reads from `data/` and writes its outputs there (or to `output/`).

```bash
# Step 1: Collect raw data from MTEB and HuggingFace API
#   - Downloads benchmark scores, model configs, README files, task metadata
#   - Takes ~30-60 minutes due to HF API rate limits
python scripts/01_collect_data.py

# Step 2: Build the unified training matrix
#   - Merges MTEB results + model metadata + task metadata + tokenizer features
#   - Computes derived features (head_dim, ffn_ratio, norm_rank, etc.)
python scripts/02_build_training_data.py

# Step 3: Extract tokenizer-level features
#   - Downloads tokenizer.json for each model and parses vocabulary
#   - Extracts: subword lengths, script coverage, continuation tokens, normalizer info
#   - Takes ~5 minutes; supports checkpointing for resume
python scripts/03_extract_tokenizer_features.py

# Step 4: Train and evaluate across three feature tiers
#   - Leave-one-task-out CV with Ridge and RandomForest
#   - Per-family performance breakdown
python scripts/04_train_and_evaluate.py

# Step 5: SHAP feature importance analysis
#   - Per-family SHAP for each tier (using RandomForest)
#   - Visualizes top features color-coded by data source
python scripts/05_shap_analysis.py
```

> **Note:** Steps 1 and 3 require internet access for HuggingFace API calls. Step 2 depends on Step 3's output (`tokenizer_features.csv`), so run Step 3 before Step 2 on a fresh setup, or run Step 2 twice.

## Feature Tiers

Features are organized into three tiers based on data acquisition difficulty:

| Tier | Source | # Features | Description |
|------|--------|-----------|-------------|
| **Tier 1** | API-only | 54 | Config.json, tokenizer.json, HF API, MTEB metadata |
| **Tier 2** | + README | 69 | + Training data keywords, method flags from README scraping |
| **Tier 3** | + Discarded | 72 | + Download count, likes, and model age (popularity proxies) |

## Key Results

- **Tier 1** RandomForest R² ≈ 0.610 — architecture and tokenizer features alone predict ~61% of ranking variance
- **Tier 2** RandomForest R² ≈ 0.626 — README scraping adds +0.016 R²
- **Tier 3** RandomForest R² ≈ 0.630 — popularity features contribute negligibly (+0.004)
