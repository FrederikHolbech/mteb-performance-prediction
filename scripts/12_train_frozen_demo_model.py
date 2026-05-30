"""Train and save a frozen RandomForest artifact for fast thesis demo inference."""

import argparse
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    EVAL_SEED,
    RF_PARAMS,
    TIERS,
    TRAINING_DATA_PATH,
)
from src.demo_inference import (
    get_demo_artifact_path,
    save_demo_artifact,
    train_frozen_demo_artifact,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and save a frozen demo model for fast single-model inference."
    )
    parser.add_argument(
        "--tier",
        default="Tier 2 (+ README)",
        choices=list(TIERS.keys()),
        help="Feature tier to train the frozen demo model on.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_df = pd.read_csv(TRAINING_DATA_PATH)

    rf_params = RF_PARAMS.copy()
    rf_params["random_state"] = EVAL_SEED

    artifact = train_frozen_demo_artifact(
        train_df,
        args.tier,
        rf_params=rf_params,
    )
    output_path = get_demo_artifact_path(args.tier)
    save_demo_artifact(artifact, output_path)

    print(f"Saved {output_path}")
    print(
        f"Rows={artifact['n_rows']}, Models={artifact['n_models']}, Tasks={artifact['n_tasks']}, Tier={artifact['tier_name']}"
    )


if __name__ == "__main__":
    main()
