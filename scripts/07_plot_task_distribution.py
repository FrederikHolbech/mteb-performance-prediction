"""Step 7: Plot task-family distribution with English subset highlighted.

Reads:
  data/task_metadata.csv

Outputs:
  output/figures/task_family_distribution.png
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import TASK_METADATA_PATH, FIGURES_DIR


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = pd.read_csv(TASK_METADATA_PATH)

    # Keep the main text families used throughout the thesis.
    keep_families = [
        "Retrieval",
        "Classification",
        "Clustering",
        "STS",
        "PairClassification",
        "BitextMining",
        "Reranking",
    ]

    task_df = df[df["task_type"].isin(keep_families)].copy()

    total = task_df.groupby("task_type")["task_name"].nunique().reindex(keep_families)
    english = (
        task_df[task_df["is_english"].astype(str).str.lower().isin(["true", "1"])]
        .groupby("task_type")["task_name"]
        .nunique()
        .reindex(keep_families)
        .fillna(0)
        .astype(int)
    )

    non_english = total - english

    x = np.arange(len(keep_families))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.bar(x, non_english.values, color="#C8D6E5", label="Non-English tasks")
    ax.bar(
        x,
        english.values,
        bottom=non_english.values,
        color="#1F77B4",
        label="Tasks including English",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Retrieval",
            "Classification",
            "Clustering",
            "STS",
            "PairClassif.",
            "BitextMining",
            "Reranking",
        ],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Number of tasks")
    ax.set_title("Task-family distribution with English coverage")
    ax.grid(axis="y", alpha=0.25)

    # Annotate totals on top of each bar.
    for i, t in enumerate(total.values):
        ax.text(i, t + 3, str(int(t)), ha="center", va="bottom", fontsize=9)

    ax.legend(loc="upper right")
    fig.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "task_family_distribution.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {out_path}")
    print("Totals by family:")
    print(total.to_string())
    print("\nEnglish-including tasks by family:")
    print(english.to_string())


if __name__ == "__main__":
    main()
