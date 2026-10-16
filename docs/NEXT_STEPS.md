# Next steps & stretch goals

Order them from "polish this project" to "second flashy project."

## Polish (makes the resume bullet stronger)

1. **Ablate β.** Run DPO at β ∈ {0.05, 0.1, 0.3}, plot win-rate vs. β. One figure
   + one sentence of analysis reads as rigor.
2. **Ablate LoRA rank** (r ∈ {8, 16, 32}): quality vs. trainable-parameter count.
3. **Log reward margins.** TRL logs `rewards/chosen`, `rewards/rejected`,
   `rewards/margins` — screenshot the margin climbing; it's the clearest "it's
   learning" signal.
4. **Add a tiny Gradio demo** so a recruiter can type a prompt and see base vs.
   aligned side by side.

## Swap the dataset for a domain you care about

Replace UltraFeedback with preferences from a niche (code review tone, medical
Q&A safety, concise summaries). Any dataset with `(prompt, chosen, rejected)` works
once `prepare_data.py` maps to those columns. A domain-specific result stands out
more than a generic one.

## Second project: GRPO reasoning (DeepSeek-R1 style)

Highest "wow" factor for interviews. Train a small model to solve **GSM8K** math
with **verifiable rewards** (exact-match on the final number) via TRL's
`GRPOTrainer`.

Sketch:
- Dataset: `openai/gsm8k`.
- Reward function: parse the model's final answer, `+1` if it equals the gold
  answer, else `0` (optionally a small format reward for using `\boxed{}`).
- Trainer: `trl.GRPOTrainer` — samples a group of completions per prompt and
  reinforces the above-average ones. No reward model, no preference labels.
- Metric: accuracy on held-out GSM8K before vs. after.

This pairs perfectly with the DPO project: "preference-based alignment (DPO) **and**
verifiable-reward RL for reasoning (GRPO)."

## Production-flavored add-ons (for MLOps-leaning roles)

- Push the adapter to the HuggingFace Hub with a model card reporting win-rate.
- Wrap inference in a FastAPI endpoint + Dockerfile.
- Track runs with Weights & Biases (`report_to="wandb"` in `DPOConfig`).

## API-drift note

TRL moves fast. If `DPOConfig`/`DPOTrainer` args changed in your installed version:
- `processing_class=` replaced the older `tokenizer=` arg — if you pinned an older
  TRL, swap it back.
- `max_length` / `max_prompt_length` live on `DPOConfig` in 0.12; older versions
  took them as `DPOTrainer` kwargs.
Check `python -c "import trl; print(trl.__version__)"` and the
[TRL docs](https://huggingface.co/docs/trl/dpo_trainer) for your version.
