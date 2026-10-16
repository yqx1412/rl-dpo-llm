# Aligning a Small LLM with Direct Preference Optimization (DPO)

End-to-end **RL-for-LLM post-training** project: take a small open base model and
fine-tune it to better match human preferences using **Direct Preference
Optimization (DPO)** with HuggingFace **TRL** + **LoRA**. Includes data prep, a
reproducible training script, and a **win-rate evaluation harness** that measures
whether the aligned model actually beats the base model on held-out prompts.

DPO is the modern, reward-model-free successor to PPO-based RLHF: instead of
training a separate reward model and running PPO, it optimizes the policy directly
on preference pairs (chosen vs. rejected) with a simple classification-style loss.
It's what most teams reach for first in 2025 because it's stable, cheap, and
single-GPU friendly.

> **On the SFT stage:** the canonical recipe is SFT → DPO. This project skips
> SFT *intentionally* by starting from `Qwen2.5-0.5B-**Instruct**`, which is
> already instruction/chat-tuned, and applies DPO on top. That keeps the project
> single-GPU-friendly while still demonstrating the full preference-optimization
> loop; a dedicated SFT stage would be the natural next step for a non-chat base.

## Why this is on my resume

- **In-demand skill:** RLHF / preference optimization is the dominant LLM
  post-training technique; DPO is the practical entry point.
- **Full pipeline:** data → training → quantitative eval, not just a notebook.
- **Measurable impact:** reports a held-out **win-rate** of the aligned model vs.
  the base model (the metric hiring managers want).
- **Reproducible:** runs on a single consumer GPU or free Colab via LoRA/QLoRA.

## Pipeline

```
raw preference data (UltraFeedback)
        │  src/prepare_data.py
        ▼
chosen / rejected pairs (chat-formatted)
        │  src/train_dpo.py   (TRL DPOTrainer + LoRA)
        ▼
DPO-aligned adapter
        │  src/evaluate.py    (LLM-as-judge win-rate)
        ▼
win-rate report vs. base model
```

## Defaults (change in `configs/dpo_config.yaml`)

| Choice | Default | Why |
|---|---|---|
| Base model | `Qwen2.5-0.5B-Instruct` | Already chat-tuned, tiny, strong for size |
| Method | DPO | Stable, reward-model-free, single-GPU |
| Dataset | `trl-lib/ultrafeedback_binarized` | Standard, clean chosen/rejected pairs |
| PEFT | LoRA (r=16) | Fits on ~8–12 GB VRAM / Colab T4 |
| Eval | Win-rate (judge model) | The resume metric |

## Quick start

**Train in your browser (no local GPU):** open the Colab notebook —
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yqx1412/rl-dpo-llm/blob/main/notebooks/dpo_colab.ipynb)
— set the runtime to a T4 GPU and run the cells top to bottom.

**Or run locally:**

```bash
# 0. Enter the project
cd /Users/yeoquanxiang/dev/RL

# 1. Setup (Python 3.10+; a CUDA GPU strongly recommended)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Prepare preference data (downloads + formats a subset)
python src/prepare_data.py --num-samples 4000

# 3. Train with DPO + LoRA
python src/train_dpo.py --config configs/dpo_config.yaml

# 4. Evaluate: aligned vs. base win-rate on held-out prompts
python src/evaluate.py --config configs/dpo_config.yaml --num-eval 100
```

On a free Colab T4, `--num-samples 2000` and 1 epoch finishes in roughly an hour.
Set `OPENAI_API_KEY` to use GPT as the eval judge, or pass `--judge local` to use a
local model.

## Try it: interactive demo

After training, launch a Gradio chat UI that loads the base model + your DPO
adapter — with a **Compare base vs aligned** toggle that shows both responses
side by side, the clearest way to *see* what preference optimization changed:

```bash
python app.py                    # chats with the aligned model (reads the config)
python app.py --base-only        # no adapter needed; try the base model
python app.py --share            # public share link (handy on Colab)
```

The demo reads the base model and adapter path from `configs/dpo_config.yaml`, so
it always points at whatever you trained. If no adapter exists yet it falls back to
the base model and tells you so.

## Resume bullets (fill in your real numbers after running)

- Fine-tuned Qwen2.5-0.5B with **Direct Preference Optimization (TRL + LoRA)** on
  4K preference pairs, raising held-out **win-rate from 50% → __%** vs. the base
  model.
- Built a reproducible RLHF-style pipeline (data prep → DPO training → LLM-as-judge
  evaluation) runnable on a single consumer GPU.
- Ran ablations on the DPO **β** parameter and LoRA rank, documenting the
  reward-margin / win-rate tradeoff.

## Learn-the-why reading list

See `docs/LEARNING.md`.

## Next steps / stretch goals

See `docs/NEXT_STEPS.md` — includes swapping DPO for **GRPO** (DeepSeek-R1 style
reasoning) as a second, flashier project.
