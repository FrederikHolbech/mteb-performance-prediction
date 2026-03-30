"""
Step 3: Extract tokenizer-level features.

Downloads tokenizer.json for each model from HuggingFace and extracts:
  - Tokenizer model type (BPE / WordPiece / Unigram)
  - Pre-tokenizer chain
  - Normalizer type + properties (lowercased, strips accents)
  - Vocabulary statistics (size, subword lengths, continuation tokens)
  - Unicode script coverage (number of scripts, % Latin, % CJK)

Falls back to vocab.txt for WordPiece models without tokenizer.json,
and to pre-loaded sequence data from inputEncodings/ as a last resort.

Reads:
  data/model_metadata_enriched.csv
  inputEncodings/data/Scripts.txt
  inputEncodings/data/1.sequences.txt  (optional fallback)

Outputs:
  data/tokenizer_features.csv
"""

import ast
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DATA_DIR,
    MODEL_METADATA_ENRICHED_PATH,
    TOKENIZER_FEATURES_PATH,
    UNICODE_SCRIPTS_PATH,
    SEQUENCES_PATH,
)
from src.utils import load_unicode_scripts, get_script

warnings.filterwarnings("ignore")
os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# Tokenizer JSON helpers
# ============================================================


def download_tokenizer_json(model_id):
    """Download and parse tokenizer.json directly from HuggingFace."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import (
        EntryNotFoundError,
        RepositoryNotFoundError,
        GatedRepoError,
    )

    try:
        path = hf_hub_download(model_id, "tokenizer.json", local_files_only=False)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (EntryNotFoundError, RepositoryNotFoundError, GatedRepoError, Exception):
        return None


def download_vocab_txt(model_id):
    """Fallback: download vocab.txt for WordPiece models (BERT-style)."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(model_id, "vocab.txt", local_files_only=False)
        with open(path, "r", encoding="utf-8") as f:
            tokens = [line.strip() for line in f if line.strip()]
        return tokens
    except Exception:
        return None


def extract_pre_tokenizer(pre_tok_obj):
    """Recursively extract pre-tokenizer chain from JSON."""
    if pre_tok_obj is None:
        return []
    t = pre_tok_obj.get("type", "")
    if t == "Sequence":
        chain = []
        for step in pre_tok_obj.get("pretokenizers", []):
            chain.extend(extract_pre_tokenizer(step))
        return chain
    return [t]


def extract_normalizer_info(norm_obj):
    """Extract normalizer type and properties."""
    if norm_obj is None:
        return {"normalizer_type": "None"}
    t = norm_obj.get("type", "")
    info = {}
    if t == "Sequence":
        parts = [n.get("type", "") for n in norm_obj.get("normalizers", [])]
        info["normalizer_type"] = "+".join(parts) if parts else "Sequence"
        all_text = json.dumps(norm_obj)
        info["is_lowercased"] = int(
            "Lowercase" in all_text or '"lowercase":true' in all_text.lower()
        )
        info["strips_accents"] = int(
            "StripAccents" in all_text or '"strip_accents":true' in all_text.lower()
        )
    elif t == "BertNormalizer" or "Bert" in t:
        info["normalizer_type"] = "BertNormalizer"
        info["is_lowercased"] = int(bool(norm_obj.get("lowercase", False)))
        info["strips_accents"] = int(bool(norm_obj.get("strip_accents", False)))
    elif t == "Precompiled":
        info["normalizer_type"] = "Precompiled"
    else:
        info["normalizer_type"] = t
        all_text = json.dumps(norm_obj)
        info["is_lowercased"] = int("Lowercase" in all_text)
        info["strips_accents"] = int("StripAccents" in all_text)
    return info


def analyze_vocab(vocab_dict, added_tokens_list, script_ranges):
    """Analyze vocabulary: subword lengths, continuation tokens, script coverage."""
    # Identify special tokens
    special_set = set()
    if added_tokens_list:
        for at in added_tokens_list:
            if isinstance(at, dict) and at.get("special", False):
                special_set.add(at.get("content", ""))

    regular_tokens = [w for w in vocab_dict.keys() if w not in special_set]

    # Clean tokens (remove prefix markers)
    clean_tokens = []
    for tok_str in regular_tokens:
        cleaned = tok_str.replace("##", "").replace("\u2581", "").replace("\u0120", "")
        if cleaned:
            clean_tokens.append(cleaned)

    result = {}
    if clean_tokens:
        char_lengths = [len(t) for t in clean_tokens]
        byte_lengths = [len(t.encode("utf-8", errors="replace")) for t in clean_tokens]
        result["avg_subword_len_chars"] = np.mean(char_lengths)
        result["avg_subword_len_bytes"] = np.mean(byte_lengths)
        result["median_subword_len_chars"] = np.median(char_lengths)
    else:
        result["avg_subword_len_chars"] = np.nan
        result["avg_subword_len_bytes"] = np.nan
        result["median_subword_len_chars"] = np.nan

    result["actual_vocab_size"] = len(vocab_dict)

    # Continuation token detection
    sample_keys = list(vocab_dict.keys())[:500]
    sample_str = "".join(sample_keys)
    is_wordpiece = any(t.startswith("##") for t in sample_keys)
    is_sentencepiece = "\u2581" in sample_str
    is_bytelevel = "\u0120" in sample_str

    n_continuation = 0
    for tok_str in regular_tokens:
        if is_wordpiece:
            if tok_str.startswith("##"):
                n_continuation += 1
        elif is_sentencepiece:
            if not tok_str.startswith("\u2581") and not tok_str.startswith("<"):
                n_continuation += 1
        elif is_bytelevel:
            if not tok_str.startswith("\u0120") and not tok_str.startswith("<"):
                n_continuation += 1

    result["pct_continuation_tokens"] = n_continuation / max(len(regular_tokens), 1)

    # Script coverage
    scripts_found = set()
    script_counts = {}
    for tok_str in regular_tokens:
        cleaned = tok_str.replace("##", "").replace("\u2581", "").replace("\u0120", "")
        tok_scripts = set()
        for char in cleaned:
            s = get_script(char, script_ranges)
            if s:
                tok_scripts.add(s)
                scripts_found.add(s)
        for s in tok_scripts:
            script_counts[s] = script_counts.get(s, 0) + 1

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


# ============================================================
# Main extraction
# ============================================================


def extract_tokenizer_features():
    """Extract tokenizer features for all models."""
    from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError

    # Load Unicode script data
    print("Loading Unicode script ranges...")
    script_ranges = load_unicode_scripts(UNICODE_SCRIPTS_PATH)

    # Load fallback sequence data
    seq_data = {}
    if os.path.isfile(SEQUENCES_PATH):
        with open(SEQUENCES_PATH, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    seq_data[parts[0]] = parts[1]
        print(f"Loaded {len(seq_data)} fallback sequence entries")

    # Load model list
    meta = pd.read_csv(MODEL_METADATA_ENRICHED_PATH)
    models = meta["model_name"].tolist()
    print(f"Extracting tokenizer features for {len(models)} models...")

    # Checkpoint support for resuming
    checkpoint_file = os.path.join(DATA_DIR, "tokenizer_features_partial.csv")
    if os.path.isfile(checkpoint_file):
        done_df = pd.read_csv(checkpoint_file)
        done_models = set(done_df["model_name"].tolist())
        results = done_df.to_dict("records")
        print(f"Resuming from checkpoint: {len(done_models)} models already done")
    else:
        done_models = set()
        results = []

    t0 = time.time()

    for i, model_id in enumerate(models):
        if model_id in done_models:
            continue

        row = {"model_name": model_id}

        try:
            # Strategy 1: Download tokenizer.json directly (fast)
            tj = download_tokenizer_json(model_id)

            if tj is not None:
                model_obj = tj.get("model", {})
                main_model = model_obj.get("type", "unknown")
                row["tokenizer_main_model"] = main_model

                pre_tok_types = extract_pre_tokenizer(tj.get("pre_tokenizer"))
                row["pre_tokenizer_chain"] = (
                    "+".join(pre_tok_types) if pre_tok_types else "None"
                )
                row["tokenizer_pipeline"] = str(pre_tok_types + [main_model])

                norm_info = extract_normalizer_info(tj.get("normalizer"))
                row.update(norm_info)

                vocab = model_obj.get("vocab", {})
                if isinstance(vocab, list):
                    vocab = {
                        item[0] if isinstance(item, (list, tuple)) else str(item): idx
                        for idx, item in enumerate(vocab)
                    }
                if not vocab and "merges" in model_obj:
                    vocab = {}
                added_tokens = tj.get("added_tokens", [])

                if vocab:
                    vocab_stats = analyze_vocab(vocab, added_tokens, script_ranges)
                    row.update(vocab_stats)
                else:
                    row["actual_vocab_size"] = np.nan

            else:
                # Strategy 2: Try vocab.txt (WordPiece / BERT-style)
                vocab_tokens = download_vocab_txt(model_id)
                if vocab_tokens:
                    row["tokenizer_main_model"] = "WordPiece"
                    row["pre_tokenizer_chain"] = "BertPreTokenizer"
                    row["normalizer_type"] = "BertNormalizer"
                    vocab_dict = {t: idx for idx, t in enumerate(vocab_tokens)}
                    vocab_stats = analyze_vocab(vocab_dict, None, script_ranges)
                    row.update(vocab_stats)
                elif model_id in seq_data:
                    # Strategy 3: Use pre-loaded sequence data
                    seq = ast.literal_eval(seq_data[model_id])
                    row["tokenizer_main_model"] = seq[-1] if seq else "unknown"
                    row["pre_tokenizer_chain"] = (
                        "+".join(seq[:-1]) if len(seq) > 1 else "None"
                    )
                else:
                    row["tokenizer_main_model"] = "error"

        except (GatedRepoError, RepositoryNotFoundError):
            row["tokenizer_main_model"] = "error"
        except Exception as e:
            print(f"  [{i+1}] ERROR {model_id}: {type(e).__name__}: {e}")
            row["tokenizer_main_model"] = "error"

        results.append(row)
        done_models.add(model_id)

        # Checkpoint every 10 models
        if (i + 1) % 10 == 0:
            pd.DataFrame(results).to_csv(checkpoint_file, index=False)
            elapsed = time.time() - t0
            print(f"[{i+1}/{len(models)}] {elapsed:.0f}s — last: {model_id}")

    # Final save
    tok_df = pd.DataFrame(results)
    tok_df.to_csv(TOKENIZER_FEATURES_PATH, index=False)
    if os.path.isfile(checkpoint_file):
        os.remove(checkpoint_file)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s! Extracted features for {len(tok_df)} models")
    print(f"Columns: {tok_df.columns.tolist()}")
    print(f"\nNull rates:")
    print((tok_df.isnull().sum() / len(tok_df) * 100).round(1))
    print(f"\ntokenizer_main_model distribution:")
    print(tok_df["tokenizer_main_model"].value_counts())
    return tok_df


if __name__ == "__main__":
    extract_tokenizer_features()
