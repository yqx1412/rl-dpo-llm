"""Generate dpo_colab.ipynb. Run once: python notebooks/_build_colab.py"""
import json
import os

REPO = "https://github.com/yqx1412/rl-dpo-llm.git"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _src(lines),
    }


def _src(lines):
    # nbformat wants a list of strings, each ending in \n except the last.
    out = []
    for i, ln in enumerate(lines):
        out.append(ln + ("\n" if i < len(lines) - 1 else ""))
    return out


cells = [
    md(
        "# DPO LLM Alignment — Colab Trainer",
        "",
        "Train a small LLM with **Direct Preference Optimization (DPO + LoRA)** end to end,",
        "right in your browser on a free Colab T4.",
        "",
        "**Before you start:** set the runtime to GPU —",
        "`Runtime → Change runtime type → T4 GPU`.",
        "",
        "Pipeline: clone repo → install → prepare data → train → evaluate (win-rate).",
    ),
    md("## 1. Check the GPU", "", "If this errors or shows no GPU, switch the runtime type (see above)."),
    code("!nvidia-smi"),
    md("## 2. Clone the project"),
    code(
        f"!git clone {REPO}",
        "%cd rl-dpo-llm",
    ),
    md(
        "## 3. Install dependencies",
        "",
        "Uses the pinned, known-compatible versions in `requirements.txt`.",
        "Takes a few minutes on first run.",
    ),
    code("!pip install -q -r requirements.txt"),
    md(
        "## 4. (Optional) Log in to Hugging Face",
        "",
        "Not required for the public Qwen model + UltraFeedback dataset, but useful",
        "if you later swap in a gated model or want to push your adapter to the Hub.",
    ),
    code(
        "# from huggingface_hub import login",
        "# login()  # paste a token from https://huggingface.co/settings/tokens",
    ),
    md(
        "## 5. Prepare the preference data",
        "",
        "Downloads UltraFeedback and keeps a 2000-row subset for a fast Colab run.",
        "Bump `--num-samples` for a stronger result if you have Colab time.",
    ),
    code("!python src/prepare_data.py --num-samples 2000 --num-test 100"),
    md(
        "## 6. Train with DPO + LoRA",
        "",
        "Reads every knob from `configs/dpo_config.yaml` (base model, LoRA rank,",
        "`beta`, learning rate). One epoch on 2000 pairs is roughly 30–60 min on a T4.",
        "",
        "Watch the `rewards/margins` value in the logs climb — that is DPO learning",
        "to prefer the chosen responses.",
    ),
    code("!python src/train_dpo.py --config configs/dpo_config.yaml"),
    md(
        "## 7. Evaluate — win-rate vs. the base model",
        "",
        "Two judge options:",
        "- **Local judge** (default below): no API key, uses a small local model.",
        "- **OpenAI judge**: set `OPENAI_API_KEY` and pass `--judge openai` for a",
        "  stronger judge (edit the cell).",
        "",
        "Each pair is judged in both orders to cancel position bias.",
    ),
    code(
        "# Local judge (no key needed):",
        "!python src/evaluate.py --config configs/dpo_config.yaml --num-eval 50 --judge local",
        "",
        "# OpenAI judge instead (stronger). Uncomment and set your key:",
        "# import os; os.environ['OPENAI_API_KEY'] = 'sk-...'",
        "# !python src/evaluate.py --config configs/dpo_config.yaml --num-eval 100 --judge openai",
    ),
    md(
        "## 8. Save your adapter (optional)",
        "",
        "The LoRA adapter is in `outputs/dpo-adapter/`. Download it, or mount Drive",
        "and copy it there so it survives the Colab session.",
    ),
    code(
        "# from google.colab import drive",
        "# drive.mount('/content/drive')",
        "# !cp -r outputs/dpo-adapter /content/drive/MyDrive/dpo-adapter",
    ),
    md(
        "---",
        "Once you have a win-rate number, put it in the README's resume bullets",
        "(`50% → __%`). That number is the headline metric for this project.",
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = os.path.join(os.path.dirname(__file__), "dpo_colab.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {out} ({len(cells)} cells)")
