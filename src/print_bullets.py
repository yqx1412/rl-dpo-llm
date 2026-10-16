"""Print ready-to-paste resume bullets from an evaluation run.

Reads the win-rate written by `src/evaluate.py` (outputs/eval_results.json) and
the training knobs from the config, then prints the exact bullet text to drop
into the README — no manual arithmetic, no guessing the number.

Usage:
    python src/print_bullets.py
    python src/print_bullets.py --results outputs/eval_results.json \
        --config configs/dpo_config.yaml
"""
import argparse
import json
import os
import sys

import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print resume bullets from eval_results.json.")
    parser.add_argument("--results", default="outputs/eval_results.json",
                        help="Path to the JSON written by evaluate.py.")
    parser.add_argument("--config", default="configs/dpo_config.yaml",
                        help="Config used for training (for model/LoRA details).")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        sys.exit(
            f"No results file at '{args.results}'.\n"
            f"Run the evaluation first:\n"
            f"    python src/evaluate.py --config {args.config} --num-eval 100\n"
            f"That writes {args.results}, then re-run this script."
        )

    with open(args.results, "r") as f:
        res = json.load(f)

    win_rate = res.get("win_rate")
    if win_rate is None:
        sys.exit(f"'{args.results}' has no 'win_rate' field — is it from evaluate.py?")
    win_pct = win_rate * 100
    num_eval = res.get("num_eval", "?")
    judge = res.get("judge", "?")

    # Pull model / LoRA details from the config for a complete bullet.
    base_model = lora_r = beta = "?"
    try:
        cfg = load_config(args.config)
        base_model = cfg.get("model", {}).get("base_model", "?")
        lora_r = cfg.get("lora", {}).get("r", "?")
        beta = cfg.get("dpo", {}).get("beta", "?")
    except FileNotFoundError:
        pass  # bullets still work without config detail

    short_model = base_model.split("/")[-1] if isinstance(base_model, str) else base_model
    delta = win_pct - 50.0
    judge_label = {"openai": "an LLM-as-judge (GPT-4o-mini)",
                   "local": "a local LLM-as-judge"}.get(judge, "an LLM-as-judge")

    print("\n" + "=" * 68)
    print(f"  Win-rate: {win_pct:.1f}%  (vs 50% base · {delta:+.1f} pts · "
          f"{num_eval} prompts · judge={judge})")
    print("=" * 68)
    print("\nPaste these into the README's resume bullets:\n")

    bullets = [
        f"- Fine-tuned {short_model} with **Direct Preference Optimization "
        f"(TRL + LoRA, r={lora_r}, β={beta})** on UltraFeedback preference "
        f"pairs, raising held-out **win-rate from 50% → {win_pct:.1f}%** vs. the "
        f"base model (judged by {judge_label} in both orderings to cancel "
        f"position bias, n={num_eval}).",
        "- Built a reproducible RLHF-style pipeline (data prep → DPO training → "
        "LLM-as-judge evaluation) runnable on a single consumer GPU / free Colab.",
        "- Ran ablations on the DPO **β** parameter and LoRA rank, documenting "
        "the reward-margin / win-rate tradeoff.",
    ]
    for b in bullets:
        print(b)

    print("\nOne-line README edit for the headline bullet:")
    print(f"  50% → {win_pct:.1f}%")
    print()


if __name__ == "__main__":
    main()
