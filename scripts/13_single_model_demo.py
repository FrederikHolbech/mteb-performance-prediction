"""Demo a single model against its current MTEB results.

Given a Hugging Face model ID, this script:
    1. Fetches the model's current Hugging Face metadata, README flags, and tokenizer features.
    2. Loads current MTEB results and finds the tasks where the model has actual scores.
    3. Rebuilds the per-task feature rows needed for inference.
    4. Loads a frozen demo RandomForest artifact.
    5. Compares predicted vs actual normalized rank per task and per task family.

Outputs:
  output/single_model_task_predictions_<slug>.csv
  output/single_model_family_summary_<slug>.csv
  output/figures/single_model_family_comparison_<slug>.png
"""

import argparse
import json
import logging
import os
import re
import sys
import warnings
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    FIGURES_DIR,
    METHOD_KEYWORDS,
    OUTPUT_DIR,
    TASK_METADATA_PATH,
    TEXT_TASK_FAMILIES,
    TIERS,
    TRAINING_KEYWORDS,
    UNICODE_SCRIPTS_PATH,
)
from src.demo_inference import (
    get_demo_artifact_path,
    load_demo_artifact,
    transform_with_artifact,
)
from src.utils import (
    classify_family,
    count_yaml_datasets,
    encode_booleans,
    extract_keyword_flags,
    extract_pooling_mode,
    get_script,
    infer_position_embedding_type,
    load_unicode_scripts,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict a single model on the MTEB tasks where it has actual results."
    )
    parser.add_argument(
        "model_id", help="Hugging Face model id, e.g. BAAI/bge-small-en-v1.5"
    )
    parser.add_argument(
        "--tier",
        default="Tier 2 (+ README)",
        choices=list(TIERS.keys()),
        help="Feature tier to use for inference.",
    )
    parser.add_argument(
        "--task-family",
        choices=TEXT_TASK_FAMILIES,
        help="Optionally limit the demo to one task family.",
    )
    return parser.parse_args()


def slugify_model_id(model_id):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)


def load_current_mteb_results(model_id, task_family=None):
    import mteb

    mteb_logger = logging.getLogger("mteb.results.task_result")
    previous_level = mteb_logger.level
    try:
        mteb_logger.setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*load_results.*deprecated.*",
                category=DeprecationWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*Missing subsets.*|.*Missing splits.*|.*Main score .* not found in scores.*",
                category=UserWarning,
            )
            results = mteb.load_results().to_dataframe()
    finally:
        mteb_logger.setLevel(previous_level)

    df_long = results.melt(
        id_vars=["task_name"], var_name="model_name", value_name="score"
    )
    df_long = df_long.dropna(subset=["score"])

    model_task_names = (
        df_long.loc[df_long["model_name"] == model_id, "task_name"]
        .dropna()
        .unique()
        .tolist()
    )
    if not model_task_names:
        raise ValueError(
            f"Model '{model_id}' has no current MTEB results in the requested task scope."
        )

    task_meta = pd.read_csv(TASK_METADATA_PATH)
    task_meta = task_meta[task_meta["task_name"].isin(model_task_names)].copy()
    task_meta = task_meta.drop_duplicates(subset=["task_name"])
    if task_meta.empty:
        raise ValueError(f"No local task metadata found for model '{model_id}'.")

    task_info = task_meta.set_index("task_name")["task_type"].to_dict()
    df_long = df_long[df_long["task_name"].isin(task_info)].copy()
    df_long["task_type"] = df_long["task_name"].map(task_info)
    df_long = df_long[df_long["task_type"].isin(TEXT_TASK_FAMILIES)].copy()

    if task_family is not None:
        df_long = df_long[df_long["task_type"] == task_family].copy()

    df_long["actual_norm_rank"] = df_long.groupby("task_name")["score"].rank(pct=True)

    model_df = df_long[df_long["model_name"] == model_id].copy()
    if model_df.empty:
        raise ValueError(
            f"Model '{model_id}' has no current MTEB results in the requested task scope."
        )

    return model_df


def fetch_single_model_metadata(model_id):
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.model_info(model_id, timeout=10)

    param_count = info.safetensors.total if info.safetensors else None

    config = {}
    try:
        path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception:
        config = {}

    row = {
        "model_name": model_id,
        "downloads": info.downloads,
        "likes": info.likes,
        "created_at": str(info.created_at),
        "pipeline_tag": info.pipeline_tag,
        "library_name": getattr(info, "library_name", None),
        "model_type": config.get("model_type") or (info.config or {}).get("model_type"),
        "param_count": param_count,
        "hidden_size": config.get("hidden_size"),
        "num_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "vocab_size": config.get("vocab_size"),
        "intermediate_size": config.get("intermediate_size"),
    }

    pool_cfg = None
    emb_dim = None
    try:
        path = hf_hub_download(repo_id=model_id, filename="1_Pooling/config.json")
        with open(path, encoding="utf-8") as handle:
            pool_cfg = json.load(handle)
        emb_dim = pool_cfg.get("word_embedding_dimension")
    except Exception:
        pool_cfg = None

    if emb_dim is None:
        emb_dim = config.get("hidden_size")

    row["embedding_dim"] = emb_dim
    row["pooling_mode"] = extract_pooling_mode(pool_cfg)
    name_or_path = config.get("_name_or_path") or config.get("base_model", "")
    row["base_model_family"] = classify_family(
        name_or_path, row["model_type"], model_id
    )
    row["hidden_act"] = config.get("hidden_act", "unknown")
    row["position_embedding_type"] = infer_position_embedding_type(config)

    nkv = config.get("num_key_value_heads")
    nah = config.get("num_attention_heads")
    row["uses_gqa"] = (nkv < nah) if (nkv is not None and nah is not None) else False

    max_seq = None
    try:
        path = hf_hub_download(repo_id=model_id, filename="sentence_bert_config.json")
        with open(path, encoding="utf-8") as handle:
            max_seq = json.load(handle).get("max_seq_length")
    except Exception:
        max_seq = None
    row["max_seq_length"] = max_seq

    has_dense = False
    try:
        hf_hub_download(repo_id=model_id, filename="2_Dense/config.json")
        has_dense = True
    except Exception:
        has_dense = False
    row["has_dense_layer"] = has_dense

    tok_class = "unknown"
    try:
        path = hf_hub_download(repo_id=model_id, filename="tokenizer_config.json")
        with open(path, encoding="utf-8") as handle:
            tok_class = json.load(handle).get("tokenizer_class", "unknown") or "unknown"
    except Exception:
        tok_class = "unknown"
    row["tokenizer_class"] = tok_class

    readme = ""
    try:
        path = hf_hub_download(repo_id=model_id, filename="README.md")
        with open(path, encoding="utf-8", errors="ignore") as handle:
            readme = handle.read()
    except Exception:
        readme = ""

    row.update(extract_keyword_flags(readme, TRAINING_KEYWORDS))
    row.update(extract_keyword_flags(readme, METHOD_KEYWORDS))
    if re.search(r"instruct", model_id.lower()):
        row["is_instruction_tuned"] = True
    row["num_datasets_listed"] = count_yaml_datasets(readme)

    created_at = pd.to_datetime(row["created_at"], utc=True, errors="coerce")
    reference_date = datetime(2025, 1, 1)
    if pd.notna(created_at):
        row["model_age_days"] = (reference_date - created_at.tz_localize(None)).days
    else:
        row["model_age_days"] = np.nan

    return row


def download_tokenizer_json(model_id):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import (
        EntryNotFoundError,
        GatedRepoError,
        RepositoryNotFoundError,
    )

    try:
        path = hf_hub_download(model_id, "tokenizer.json", local_files_only=False)
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (EntryNotFoundError, RepositoryNotFoundError, GatedRepoError, Exception):
        return None


def download_vocab_txt(model_id):
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(model_id, "vocab.txt", local_files_only=False)
        with open(path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    except Exception:
        return None


def extract_pre_tokenizer(pre_tok_obj):
    if pre_tok_obj is None:
        return []
    tok_type = pre_tok_obj.get("type", "")
    if tok_type == "Sequence":
        chain = []
        for step in pre_tok_obj.get("pretokenizers", []):
            chain.extend(extract_pre_tokenizer(step))
        return chain
    return [tok_type]


def extract_normalizer_info(norm_obj):
    if norm_obj is None:
        return {"normalizer_type": "None"}
    tok_type = norm_obj.get("type", "")
    info = {}
    if tok_type == "Sequence":
        parts = [n.get("type", "") for n in norm_obj.get("normalizers", [])]
        info["normalizer_type"] = "+".join(parts) if parts else "Sequence"
        all_text = json.dumps(norm_obj)
        info["is_lowercased"] = int(
            "Lowercase" in all_text or '"lowercase":true' in all_text.lower()
        )
        info["strips_accents"] = int(
            "StripAccents" in all_text or '"strip_accents":true' in all_text.lower()
        )
    elif tok_type == "BertNormalizer" or "Bert" in tok_type:
        info["normalizer_type"] = "BertNormalizer"
        info["is_lowercased"] = int(bool(norm_obj.get("lowercase", False)))
        info["strips_accents"] = int(bool(norm_obj.get("strip_accents", False)))
    elif tok_type == "Precompiled":
        info["normalizer_type"] = "Precompiled"
    else:
        info["normalizer_type"] = tok_type
        all_text = json.dumps(norm_obj)
        info["is_lowercased"] = int("Lowercase" in all_text)
        info["strips_accents"] = int("StripAccents" in all_text)
    return info


def analyze_vocab(vocab_dict, added_tokens_list, script_ranges):
    special_set = set()
    if added_tokens_list:
        for item in added_tokens_list:
            if isinstance(item, dict) and item.get("special", False):
                special_set.add(item.get("content", ""))

    regular_tokens = [token for token in vocab_dict.keys() if token not in special_set]
    clean_tokens = []
    for token in regular_tokens:
        cleaned = token.replace("##", "").replace("\u2581", "").replace("\u0120", "")
        if cleaned:
            clean_tokens.append(cleaned)

    result = {}
    if clean_tokens:
        char_lengths = [len(token) for token in clean_tokens]
        byte_lengths = [
            len(token.encode("utf-8", errors="replace")) for token in clean_tokens
        ]
        result["avg_subword_len_chars"] = np.mean(char_lengths)
        result["avg_subword_len_bytes"] = np.mean(byte_lengths)
        result["median_subword_len_chars"] = np.median(char_lengths)
    else:
        result["avg_subword_len_chars"] = np.nan
        result["avg_subword_len_bytes"] = np.nan
        result["median_subword_len_chars"] = np.nan

    result["actual_vocab_size"] = len(vocab_dict)

    sample_keys = list(vocab_dict.keys())[:500]
    sample_str = "".join(sample_keys)
    is_wordpiece = any(token.startswith("##") for token in sample_keys)
    is_sentencepiece = "\u2581" in sample_str
    is_bytelevel = "\u0120" in sample_str

    n_continuation = 0
    for token in regular_tokens:
        if is_wordpiece:
            if token.startswith("##"):
                n_continuation += 1
        elif is_sentencepiece:
            if not token.startswith("\u2581") and not token.startswith("<"):
                n_continuation += 1
        elif is_bytelevel:
            if not token.startswith("\u0120") and not token.startswith("<"):
                n_continuation += 1

    result["pct_continuation_tokens"] = n_continuation / max(len(regular_tokens), 1)

    scripts_found = set()
    script_counts = {}
    for token in regular_tokens:
        cleaned = token.replace("##", "").replace("\u2581", "").replace("\u0120", "")
        token_scripts = set()
        for char in cleaned:
            script = get_script(char, script_ranges)
            if script:
                token_scripts.add(script)
                scripts_found.add(script)
        for script in token_scripts:
            script_counts[script] = script_counts.get(script, 0) + 1

    result["num_scripts_in_vocab"] = len(scripts_found)
    if regular_tokens:
        result["pct_latin_tokens"] = script_counts.get("Latin", 0) / len(regular_tokens)
        result["pct_cjk_tokens"] = (
            script_counts.get("Han", 0)
            + script_counts.get("Hangul", 0)
            + script_counts.get("Katakana", 0)
            + script_counts.get("Hiragana", 0)
        ) / len(regular_tokens)
    else:
        result["pct_latin_tokens"] = np.nan
        result["pct_cjk_tokens"] = np.nan

    return result


def fetch_single_tokenizer_features(model_id):
    script_ranges = load_unicode_scripts(UNICODE_SCRIPTS_PATH)
    row = {"model_name": model_id}

    tokenizer_json = download_tokenizer_json(model_id)
    if tokenizer_json is not None:
        model_obj = tokenizer_json.get("model", {})
        main_model = model_obj.get("type", "unknown")
        row["tokenizer_main_model"] = main_model

        pre_tok_types = extract_pre_tokenizer(tokenizer_json.get("pre_tokenizer"))
        row["pre_tokenizer_chain"] = (
            "+".join(pre_tok_types) if pre_tok_types else "None"
        )
        row.update(extract_normalizer_info(tokenizer_json.get("normalizer")))

        vocab = model_obj.get("vocab", {})
        if isinstance(vocab, list):
            vocab = {
                item[0] if isinstance(item, (list, tuple)) else str(item): idx
                for idx, item in enumerate(vocab)
            }
        added_tokens = tokenizer_json.get("added_tokens", [])

        if vocab:
            row.update(analyze_vocab(vocab, added_tokens, script_ranges))
        else:
            row["actual_vocab_size"] = np.nan
    else:
        vocab_tokens = download_vocab_txt(model_id)
        if vocab_tokens:
            row["tokenizer_main_model"] = "WordPiece"
            row["pre_tokenizer_chain"] = "BertPreTokenizer"
            row["normalizer_type"] = "BertNormalizer"
            row["is_lowercased"] = 0
            row["strips_accents"] = 0
            row.update(
                analyze_vocab(
                    {token: idx for idx, token in enumerate(vocab_tokens)},
                    None,
                    script_ranges,
                )
            )
        else:
            row["tokenizer_main_model"] = "error"
            row["pre_tokenizer_chain"] = "None"
            row["normalizer_type"] = "unknown"

    return row


def fetch_task_metadata_for(task_names):
    task_df = pd.read_csv(TASK_METADATA_PATH)
    task_df = task_df[task_df["task_name"].isin(task_names)].copy()
    task_df = task_df.drop_duplicates(subset=["task_name", "task_type"])
    missing_tasks = set(task_names) - set(task_df["task_name"].tolist())
    if missing_tasks:
        raise ValueError(f"Missing task metadata for: {sorted(missing_tasks)}")
    return task_df


def build_prediction_rows(model_results, model_meta, tokenizer_meta):
    task_meta = fetch_task_metadata_for(model_results["task_name"].tolist())

    predict_df = model_results.merge(
        task_meta, on=["task_name", "task_type"], how="left"
    )
    for key, value in model_meta.items():
        predict_df[key] = value
    for key, value in tokenizer_meta.items():
        if key != "model_name":
            predict_df[key] = value

    predict_df = encode_booleans(predict_df)
    return predict_df


def predict_single_model_fast(model_id, tier_name, task_family=None):
    artifact_path = get_demo_artifact_path(tier_name)
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"Frozen demo artifact not found at '{artifact_path}'. Run scripts/12_train_frozen_demo_model.py first."
        )

    artifact = load_demo_artifact(artifact_path)

    model_results = load_current_mteb_results(model_id, task_family=task_family)
    model_meta = fetch_single_model_metadata(model_id)
    tokenizer_meta = fetch_single_tokenizer_features(model_id)
    predict_df = build_prediction_rows(model_results, model_meta, tokenizer_meta)

    X_predict = transform_with_artifact(predict_df, artifact)
    preds = artifact["model"].predict(X_predict)

    comparison_df = predict_df[
        ["task_name", "task_type", "score", "actual_norm_rank"]
    ].copy()
    comparison_df["predicted_norm_rank"] = preds.astype(float)
    comparison_df["abs_error"] = (
        comparison_df["predicted_norm_rank"] - comparison_df["actual_norm_rank"]
    ).abs()

    family_summary = (
        comparison_df.groupby("task_type")
        .agg(
            n_tasks=("task_name", "count"),
            actual_norm_rank=("actual_norm_rank", "mean"),
            predicted_norm_rank=("predicted_norm_rank", "mean"),
        )
        .reset_index()
        .sort_values("actual_norm_rank", ascending=False)
    )

    return (
        comparison_df.sort_values("actual_norm_rank", ascending=False),
        family_summary,
    )


def save_outputs(model_id, comparison_df, family_summary, task_family=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    slug = slugify_model_id(model_id)
    family_slug = slugify_model_id(task_family) if task_family else None
    suffix = f"_{family_slug}" if family_slug else ""
    task_path = os.path.join(
        OUTPUT_DIR, f"single_model_task_predictions_{slug}{suffix}.csv"
    )
    family_path = os.path.join(
        OUTPUT_DIR, f"single_model_family_summary_{slug}{suffix}.csv"
    )
    figure_path = os.path.join(
        FIGURES_DIR, f"single_model_family_comparison_{slug}{suffix}.png"
    )

    comparison_df.to_csv(task_path, index=False)
    family_summary.to_csv(family_path, index=False)

    plot_df = family_summary.sort_values("actual_norm_rank", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    positions = np.arange(len(plot_df))
    width = 0.38
    ax.barh(
        positions - width / 2,
        plot_df["actual_norm_rank"],
        height=width,
        color="#1F77B4",
        label="Actual",
    )
    ax.barh(
        positions + width / 2,
        plot_df["predicted_norm_rank"],
        height=width,
        color="#FF7F0E",
        label="Predicted",
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(plot_df["task_type"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean normalized rank")
    title_scope = task_family if task_family else "task family"
    ax.set_title(f"{model_id}: actual vs predicted performance by {title_scope}")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close(fig)

    return task_path, family_path, figure_path


def main():
    args = parse_args()
    comparison_df, family_summary = predict_single_model_fast(
        args.model_id,
        args.tier,
        task_family=args.task_family,
    )
    task_path, family_path, figure_path = save_outputs(
        args.model_id, comparison_df, family_summary, task_family=args.task_family
    )

    print(f"Saved {task_path}")
    print(f"Saved {family_path}")
    print(f"Saved {figure_path}")
    print("\nPer-family summary:")
    print(
        family_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
