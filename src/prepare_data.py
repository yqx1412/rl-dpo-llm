"""Prepare preference data for DPO training.

Downloads the UltraFeedback binarized dataset (already in chosen/rejected form),
optionally subsamples it for fast iteration, and saves it to disk in the exact
format TRL's DPOTrainer expects: each row has `prompt`, `chosen`, `rejected`
as chat-formatted strings/messages.

Usage:
    python src/prepare_data.py --num-samples 4000
    python src/prepare_data.py --config configs/dpo_config.yaml --num-samples 2000
"""
import argparse
import os

import yaml
from datasets import load_dataset


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DPO preference data.")
    parser.add_argument("--config", default="configs/dpo_config.yaml")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4000,
        help="Train rows to keep (0 = keep all). Smaller = faster iteration.",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=200,
        help="Held-out rows kept for evaluation prompts.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    out_dir = data_cfg["prepared_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading dataset: {data_cfg['dataset_name']}")
    ds = load_dataset(data_cfg["dataset_name"])

    train = ds[data_cfg["train_split"]]
    test = ds[data_cfg["test_split"]] if data_cfg["test_split"] in ds else None

    # The TRL ultrafeedback_binarized set already exposes `chosen`/`rejected`
    # as lists of chat messages, which DPOTrainer accepts directly. We only
    # verify the schema and subsample.
    required = {"chosen", "rejected"}
    missing = required - set(train.column_names)
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns {missing}. "
            f"Got columns: {train.column_names}. "
            "Point --config at a dataset already in chosen/rejected form, or "
            "add a mapping step here."
        )

    if args.num_samples and args.num_samples < len(train):
        train = train.shuffle(seed=42).select(range(args.num_samples))
    print(f"Train rows kept: {len(train)}")

    if test is not None and args.num_test and args.num_test < len(test):
        test = test.shuffle(seed=42).select(range(args.num_test))
    if test is not None:
        print(f"Test rows kept: {len(test)}")

    train.save_to_disk(os.path.join(out_dir, "train"))
    if test is not None:
        test.save_to_disk(os.path.join(out_dir, "test"))

    print(f"Saved prepared data to: {out_dir}")
    print("Next: python src/train_dpo.py --config", args.config)


if __name__ == "__main__":
    main()
