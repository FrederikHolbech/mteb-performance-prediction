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
│   ├── 05_shap_analysis.py            # Per-family SHAP analysis across tiers
│   └── 06_feature_analysis.py         # Correlation, clusters, ablation, lean model
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
#   - Builds the final training matrix and normalized-rank target
python scripts/02_build_training_data.py

# Step 3: Extract tokenizer-level features
#   - Downloads tokenizer.json for each model and parses vocabulary
#   - Extracts: subword lengths, script coverage, continuation tokens, normalizer info
#   - Takes ~5 minutes; supports checkpointing for resume
python scripts/03_extract_tokenizer_features.py

# Step 4: Train and evaluate across three feature tiers
#   - Leave-one-task-out CV with model comparison, tier comparison, and candidate ranking
#   - Per-family performance breakdown
python scripts/04_train_and_evaluate.py

# Step 5: SHAP feature importance analysis
#   - Per-family SHAP for each tier (using lighter RandomForest for speed)
#   - Subsamples up to 2000 rows per family for SHAP computation
#   - Visualizes top features color-coded by data source
python scripts/05_shap_analysis.py

# Step 6: Feature correlation, clusters, ablation, and lean model
#   - Pearson correlation heatmap, identifies highly-correlated pairs
#   - Detects feature clusters (>2 features with |r| >= 0.85)
#   - Leave-one-feature-out ablation to measure each feature's contribution
#   - Builds a lean model keeping one representative per cluster
python scripts/06_feature_analysis.py
```

> **Note:** Steps 1 and 3 require internet access for HuggingFace API calls. Step 2 depends on Step 3's output (`tokenizer_features.csv`), so run Step 3 before Step 2 on a fresh setup, or run Step 2 twice.

> **Caching:** Scripts 04–06 cache their expensive results to CSV. On subsequent runs, set the `RUN_*` flags at the top of each script to `False` to skip recomputation and load from cache.

## Feature Tiers

Features are organized into three tiers based on data acquisition difficulty:

| Tier | Source | # Features | Description |
|------|--------|-----------|-------------|
| **Tier 1** | API-only | 48 | Config.json, tokenizer.json, HF API, and MTEB task metadata |
| **Tier 2** | + README | 63 | + Training data keywords, method flags, and dataset-count features from README scraping |
| **Tier 3** | + Discarded | 66 | + Download count, likes, and model age (popularity proxies) |

## Key Results

- **Tier 1** RandomForest R² ≈ 0.544 — API and tokenizer metadata alone explain a substantial share of ranking variance
- **Tier 2** RandomForest R² ≈ 0.555 — README scraping adds a modest +0.010 R²
- **Tier 3** RandomForest R² ≈ 0.557 — popularity features contribute negligibly (+0.002)
- **Candidate ranking** The true best model appears in the top-10 predicted candidates for 78% of held-out tasks
