"""Step 11: Plot availability of README-derived training-data features.

This script summarizes how often selected training-data-related README features
are available at the model level. It creates:

  output/readme_training_feature_prevalence.csv
  output/readme_dataset_count_distribution.csv
  output/figures/readme_training_feature_availability.png
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import FIGURES_DIR, MODEL_METADATA_ENRICHED_PATH, OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

TRAINING_FLAGS = [
    "train_msmarco",
    "train_nli",
    "train_multilingual",
    "train_retrieval",
    "train_sts",
    "train_classification",
    "train_wikipedia",
    "train_commoncrawl",
    "train_squad",
]

DISPLAY_NAMES = {
    "train_msmarco": "MS MARCO",
    "train_nli": "NLI",
    "train_multilingual": "Multilingual",
    "train_retrieval": "Retrieval",
    "train_sts": "STS",
    "train_classification": "Classification",
    "train_wikipedia": "Wikipedia",
    "train_commoncrawl": "Common Crawl",
    "train_squad": "SQuAD",
}


def _to_binary(series):
    mapped = series.astype(str).str.lower().map({"true": 1, "false": 0})
    return mapped.fillna(pd.to_numeric(series, errors="coerce")).fillna(0).astype(int)


def main():
    df = pd.read_csv(MODEL_METADATA_ENRICHED_PATH)
    n_models = len(df)

    prevalence_rows = []
    for feature in TRAINING_FLAGS:
        values = _to_binary(df[feature])
        count = int(values.sum())
        prevalence_rows.append(
            {
                "feature": feature,
                "label": DISPLAY_NAMES[feature],
                "count": count,
                "share": count / n_models,
            }
        )

    prevalence_df = pd.DataFrame(prevalence_rows).sort_values("share", ascending=True)
    prevalence_df.to_csv(
        os.path.join(OUTPUT_DIR, "readme_training_feature_prevalence.csv"), index=False
    )

    dataset_counts = pd.to_numeric(df["num_datasets_listed"], errors="coerce").fillna(0)
    bucketed = dataset_counts.clip(upper=5).astype(int)
    bucket_labels = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5+"}
    distribution_df = (
        bucketed.map(bucket_labels)
        .value_counts()
        .reindex(["0", "1", "2", "3", "4", "5+"], fill_value=0)
        .rename_axis("datasets_listed_bucket")
        .reset_index(name="count")
    )
    distribution_df["share"] = distribution_df["count"] / n_models
    distribution_df.to_csv(
        os.path.join(OUTPUT_DIR, "readme_dataset_count_distribution.csv"), index=False
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.45, 1]}
    )

    axes[0].barh(prevalence_df["label"], prevalence_df["share"], color="#4C78A8")
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Share of models with README mention", fontsize=11)
    axes[0].set_title("README-derived training-data flags", fontsize=12)
    for y, share, count in zip(
        prevalence_df["label"], prevalence_df["share"], prevalence_df["count"]
    ):
        axes[0].text(
            min(share + 0.02, 0.98),
            y,
            f"{share:.1%} ({count})",
            va="center",
            fontsize=9,
        )

    axes[1].barh(
        distribution_df["datasets_listed_bucket"],
        distribution_df["share"],
        color="#72B7B2",
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Share of models", fontsize=11)
    axes[1].set_ylabel("Datasets listed in YAML frontmatter", fontsize=11)
    axes[1].set_title("Structured dataset listings are rare", fontsize=12)
    for y, share, count in zip(
        distribution_df["datasets_listed_bucket"],
        distribution_df["share"],
        distribution_df["count"],
    ):
        axes[1].text(
            min(share + 0.02, 0.98),
            y,
            f"{share:.1%} ({count})",
            va="center",
            fontsize=9,
        )

    fig.suptitle("Availability of README-derived training-data metadata", fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "readme_training_feature_availability.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(
        f"Saved: {os.path.join(OUTPUT_DIR, 'readme_training_feature_prevalence.csv')}"
    )
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'readme_dataset_count_distribution.csv')}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
