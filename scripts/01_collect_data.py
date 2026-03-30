"""
Step 1: Collect raw data from MTEB and HuggingFace.

Fetches:
  - MTEB benchmark results (model × task scores)
  - Model metadata from HuggingFace API (config.json, parameters, card data)
  - Task metadata from MTEB (languages, domains, subtypes)
  - Enriched model metadata (pooling, architecture details, README flags)
  - Enriched task metadata (descriptive statistics)

Outputs:
  data/mteb_results_clean.csv
  data/model_metadata.csv
  data/model_metadata_enriched.csv
  data/task_metadata.csv
"""

import json
import os
import re
import sys
import time
import warnings
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DATA_DIR,
    TEXT_TASK_FAMILIES,
    TRAINING_KEYWORDS,
    METHOD_KEYWORDS,
    MTEB_RESULTS_PATH,
    MODEL_METADATA_PATH,
    MODEL_METADATA_ENRICHED_PATH,
    TASK_METADATA_PATH,
)
from src.utils import (
    classify_family,
    extract_keyword_flags,
    count_yaml_datasets,
    extract_pooling_mode,
    infer_position_embedding_type,
    extract_task_features,
)

warnings.filterwarnings("ignore")
os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# 1a. Fetch MTEB results
# ============================================================


def fetch_mteb_results():
    """Download MTEB results and save as long-format CSV."""
    import mteb

    print("Loading MTEB results...")
    results = mteb.load_results()
    df = results.to_dataframe()

    # Reshape to long format
    df_long = df.melt(id_vars=["task_name"], var_name="model_name", value_name="score")
    df_long = df_long.dropna(subset=["score"])

    # Map task types
    tasks = mteb.get_tasks()
    task_info = {t.metadata.name: {"task_type": t.metadata.type} for t in tasks}
    df_long["task_type"] = df_long["task_name"].map(
        lambda x: task_info.get(x, {}).get("task_type")
    )

    # Keep only text task families (excluding Summarization — only 1 task)
    df_clean = df_long[df_long["task_type"].isin(TEXT_TASK_FAMILIES)].copy()

    print(f"Total records: {len(df_clean)}")
    print(f"Task type distribution:\n{df_clean['task_type'].value_counts()}")

    df_clean.to_csv(MTEB_RESULTS_PATH, index=False)
    print(f"Saved to {MTEB_RESULTS_PATH}")
    return df_clean


# ============================================================
# 1b. Fetch model metadata from HuggingFace API
# ============================================================


def fetch_model_metadata():
    """Fetch model metadata from HuggingFace API for all MTEB models."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import RepositoryNotFoundError

    df = pd.read_csv(MTEB_RESULTS_PATH)
    models = df["model_name"].unique().tolist()
    print(f"Fetching metadata for {len(models)} models...")

    api = HfApi()
    results = []
    failed = []

    for i, model_id in enumerate(models):
        try:
            info = api.model_info(model_id, timeout=10)

            # Parameter count from safetensors
            param_count = info.safetensors.total if info.safetensors else None

            # config.json for architecture
            config = {}
            try:
                path = hf_hub_download(repo_id=model_id, filename="config.json")
                with open(path) as f:
                    config = json.load(f)
            except Exception:
                pass

            card = info.cardData or {}
            if hasattr(card, "__dict__"):
                card = card.__dict__

            results.append(
                {
                    "model_name": model_id,
                    "downloads": info.downloads,
                    "likes": info.likes,
                    "created_at": str(info.created_at),
                    "pipeline_tag": info.pipeline_tag,
                    "library_name": getattr(info, "library_name", None),
                    "model_type": config.get("model_type")
                    or (info.config or {}).get("model_type"),
                    "param_count": param_count,
                    "hidden_size": config.get("hidden_size"),
                    "num_layers": config.get("num_hidden_layers"),
                    "num_attention_heads": config.get("num_attention_heads"),
                    "max_position_embeddings": config.get("max_position_embeddings"),
                    "vocab_size": config.get("vocab_size"),
                    "intermediate_size": config.get("intermediate_size"),
                }
            )
        except (RepositoryNotFoundError, Exception) as e:
            failed.append({"model_name": model_id, "error": str(e)})

        if (i + 1) % 50 == 0:
            print(
                f"  [{i+1}/{len(models)}] {len(results)} succeeded, {len(failed)} failed"
            )
        time.sleep(0.3)

    meta_df = pd.DataFrame(results)
    meta_df.to_csv(MODEL_METADATA_PATH, index=False)
    print(f"Saved {len(meta_df)} models to {MODEL_METADATA_PATH}")
    if failed:
        print(f"  {len(failed)} models failed")
    return meta_df


# ============================================================
# 1c. Fetch enriched model metadata
# ============================================================


def fetch_enriched_metadata():
    """Enrich model metadata with pooling, architecture, and README features."""
    from huggingface_hub import hf_hub_download

    meta = pd.read_csv(MODEL_METADATA_PATH)
    models = meta["model_name"].tolist()
    print(f"Enriching {len(models)} models...")

    enriched = []
    for i, model_id in enumerate(models):
        row = {"model_name": model_id}

        # 1. Pooling config
        pool_cfg = None
        emb_dim = None
        try:
            path = hf_hub_download(repo_id=model_id, filename="1_Pooling/config.json")
            with open(path) as f:
                pool_cfg = json.load(f)
            emb_dim = pool_cfg.get("word_embedding_dimension")
        except Exception:
            pass

        if emb_dim is None:
            try:
                path = hf_hub_download(repo_id=model_id, filename="config.json")
                with open(path) as f:
                    config = json.load(f)
                emb_dim = config.get("hidden_size")
            except Exception:
                pass

        row["embedding_dim"] = emb_dim
        row["pooling_mode"] = extract_pooling_mode(pool_cfg)

        # 2. Main config
        config = {}
        try:
            path = hf_hub_download(repo_id=model_id, filename="config.json")
            with open(path) as f:
                config = json.load(f)
        except Exception:
            pass

        name_or_path = config.get("_name_or_path") or config.get("base_model", "")
        model_type = config.get("model_type", "")
        row["base_model_family"] = classify_family(name_or_path, model_type, model_id)
        row["hidden_act"] = config.get("hidden_act", "unknown")
        row["position_embedding_type"] = infer_position_embedding_type(config)

        nkv = config.get("num_key_value_heads")
        nah = config.get("num_attention_heads")
        row["uses_gqa"] = (
            (nkv < nah) if (nkv is not None and nah is not None) else False
        )

        # 3. Sentence-BERT config
        max_seq = None
        try:
            path = hf_hub_download(
                repo_id=model_id, filename="sentence_bert_config.json"
            )
            with open(path) as f:
                max_seq = json.load(f).get("max_seq_length")
        except Exception:
            pass
        row["max_seq_length"] = max_seq

        # 4. Dense layer
        has_dense = False
        try:
            hf_hub_download(repo_id=model_id, filename="2_Dense/config.json")
            has_dense = True
        except Exception:
            pass
        row["has_dense_layer"] = has_dense

        # 5. Tokenizer class
        tok_class = "unknown"
        try:
            path = hf_hub_download(repo_id=model_id, filename="tokenizer_config.json")
            with open(path) as f:
                tok_class = json.load(f).get("tokenizer_class", "unknown") or "unknown"
        except Exception:
            pass
        row["tokenizer_class"] = tok_class

        # 6. README: training data + method keywords
        readme = ""
        try:
            path = hf_hub_download(repo_id=model_id, filename="README.md")
            with open(path, encoding="utf-8", errors="ignore") as f:
                readme = f.read()
        except Exception:
            pass

        row.update(extract_keyword_flags(readme, TRAINING_KEYWORDS))
        row.update(extract_keyword_flags(readme, METHOD_KEYWORDS))
        if re.search(r"instruct", model_id.lower()):
            row["is_instruction_tuned"] = True
        row["num_datasets_listed"] = count_yaml_datasets(readme)

        enriched.append(row)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(models)}] processed...")
        time.sleep(0.2)

    enriched_df = pd.DataFrame(enriched)

    # Model age
    meta["created_at"] = pd.to_datetime(meta["created_at"], utc=True, errors="coerce")
    reference_date = datetime(2025, 1, 1)
    meta["model_age_days"] = (
        reference_date - meta["created_at"].dt.tz_localize(None)
    ).dt.days

    meta_enriched = meta.merge(enriched_df, on="model_name", how="left")
    meta_enriched.to_csv(MODEL_METADATA_ENRICHED_PATH, index=False)
    print(
        f"Saved {len(meta_enriched)} enriched models to {MODEL_METADATA_ENRICHED_PATH}"
    )
    return meta_enriched


# ============================================================
# 1d. Fetch task metadata
# ============================================================


def fetch_task_metadata():
    """Extract task-level features from MTEB metadata."""
    import mteb

    tasks = mteb.get_tasks()
    text_tasks = [t for t in tasks if t.metadata.type in TEXT_TASK_FAMILIES]

    records = []
    for t in text_tasks:
        m = t.metadata

        # Basic features from extract_task_features
        rec = extract_task_features(
            {
                "task_name": m.name,
                "task_type": m.type,
                "languages": m.languages,
                "domains": m.domains or [],
                "task_subtypes": m.task_subtypes or [],
            }
        )

        # Enrichment: descriptive stats
        rec["main_score"] = m.main_score
        rec["annotations_creators"] = m.annotations_creators

        dsp = str(m.descriptive_stat_path)
        ds = None
        for path in [dsp, dsp + ".json"]:
            if os.path.exists(path):
                with open(path) as f:
                    ds = json.load(f)
                break

        if ds:
            test_data = ds.get("test", {})
            if isinstance(test_data, dict):
                rec["num_samples_test"] = test_data.get("num_samples")
                ts = test_data.get("text_statistics", {})
                rec["avg_text_length"] = ts.get("average_text_length")
                ls = test_data.get("label_statistics", {})
                rec["unique_labels"] = ls.get("unique_labels", 0)
            else:
                rec["num_samples_test"] = None
                rec["avg_text_length"] = None
                rec["unique_labels"] = 0
        else:
            rec["num_samples_test"] = None
            rec["avg_text_length"] = None
            rec["unique_labels"] = 0

        records.append(rec)

    task_df = pd.DataFrame(records)
    task_df.to_csv(TASK_METADATA_PATH, index=False)
    print(f"Saved {len(task_df)} tasks to {TASK_METADATA_PATH}")
    return task_df


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("\n 1a. MTEB Results")
    fetch_mteb_results()

    print("\n 1b. Model Metadata")
    fetch_model_metadata()

    print("\n 1c. Enriched Model Metadata")
    fetch_enriched_metadata()

    print("\n 1d. Task Metadata")
    fetch_task_metadata()

