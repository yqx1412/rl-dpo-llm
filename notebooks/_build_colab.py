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
    md(
        "## 🚀 One-click run",
        "",
        "**New here? Just run this one cell** (or `Runtime → Run all`). It does the",
        "whole pipeline end to end with free-T4-safe settings: clone → install →",
        "prepare data → train (DPO + LoRA) → evaluate → print your win-rate.",
        "",
        "First set the runtime to a **T4 GPU** (`Runtime → Change runtime type`),",
        "then run. Expect **~45–75 min**. Watch for `Aligned model win-rate vs base:`",
        "near the end — that is your headline number.",
        "",
        "Prefer to go step by step and understand each stage? Skip this cell and run",
        "sections 1–8 below instead.",
    ),
    code(
        "import os, subprocess, sys",
        "",
        "# T4-safe knobs. Lower TRAIN_SAMPLES if you are short on Colab time;",
        "# the per-device batch=1 + grad-accum=16 combo keeps VRAM well under a T4.",
        "TRAIN_SAMPLES = 4000   # preference pairs to train on",
        "TEST_SAMPLES  = 100    # held-out prompts available for eval",
        "NUM_EVAL      = 50     # how many to actually judge (local judge, no API key)",
        "",
        "# Absolute, fixed project location so re-runs never nest a second clone",
        "# inside the first (which split data prep and training across two dirs).",
        "BASE    = '/content' if os.path.isdir('/content') else os.getcwd()",
        "PROJECT = os.path.join(BASE, 'rl-dpo-llm')",
        "",
        "def sh(cmd):",
        "    print(f'\\n$ {cmd}\\n', flush=True)",
        "    rc = subprocess.call(cmd, shell=True)",
        "    if rc != 0:",
        "        raise SystemExit(f'Step failed (exit {rc}): {cmd}\\n'",
        "                         f'(scroll up for the actual error from that command)')",
        "",
        "# 1. Clone once to the fixed path, then always work from there.",
        f"if not os.path.isdir(PROJECT):",
        f"    sh(f'git clone {REPO} {{PROJECT}}')",
        "os.chdir(PROJECT)",
        "print('Working dir:', os.getcwd())",
        "",
        "# GPU sanity check — DPO on CPU is not viable. If this prints no GPU,",
        "# set Runtime -> Change runtime type -> T4 GPU and re-run this cell.",
        "import torch",
        "if not torch.cuda.is_available():",
        "    raise SystemExit('No GPU detected. Set Runtime -> Change runtime type '",
        "                     '-> T4 GPU, then re-run this cell. (DPO training on CPU '",
        "                     'is far too slow to finish.)')",
        "print('GPU:', torch.cuda.get_device_name(0))",
        "",
        "# 2. Install, 3. prepare data, 4. train, 5. evaluate",
        "sh('pip install -q -r requirements.txt')",
        "# Colab preinstalls an old torchao (0.10) that recent PEFT rejects during",
        "# LoRA injection. We do plain LoRA (no torchao), so removing it makes PEFT",
        "# skip the torchao path cleanly. Harmless if torchao isn't present.",
        "subprocess.call('pip uninstall -y -q torchao', shell=True)",
        "sh(f'python src/prepare_data.py --num-samples {TRAIN_SAMPLES} --num-test {TEST_SAMPLES}')",
        "sh('python src/train_dpo.py --config configs/dpo_config.yaml')",
        "sh(f'python src/evaluate.py --config configs/dpo_config.yaml --num-eval {NUM_EVAL} --judge local')",
        "",
        "# 6. Print ready-to-paste resume bullets with your real win-rate",
        "sh('python src/print_bullets.py')",
        "print('\\n✅ Done. Copy the win-rate bullet above into the README.')",
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
        "Uses the version floors in `requirements.txt` (kept compatible with the",
        "DPOTrainer API). On Colab, `torch` is already installed — these just add",
        "the HuggingFace stack on top. Takes a few minutes on first run.",
    ),
    code(
        "!pip install -q -r requirements.txt",
        "# Remove Colab's old preinstalled torchao (0.10) that recent PEFT rejects",
        "# during LoRA injection. This project uses plain LoRA, not torchao.",
        "!pip uninstall -y -q torchao",
    ),
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
        "Downloads UltraFeedback and keeps a 4000-row subset. More pairs give a",
        "stronger preference signal; lower `--num-samples` if you are short on time.",
    ),
    code("!python src/prepare_data.py --num-samples 4000 --num-test 100"),
    md(
        "## 6. Train with DPO + LoRA",
        "",
        "Reads every knob from `configs/dpo_config.yaml` (base model, LoRA rank,",
        "`beta`, learning rate). Two epochs on 4000 pairs is roughly 45–90 min on a T4.",
        "",
        "Watch the `rewards/margins` value in the logs climb — that is DPO learning",
        "to prefer the chosen responses.",
        "",
        "**If you hit a CUDA out-of-memory error on a free T4:** open",
        "`configs/dpo_config.yaml` and set `per_device_train_batch_size: 1` and",
        "`gradient_accumulation_steps: 16` (same effective batch of 16, lower peak",
        "VRAM), then re-run this cell.",
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
