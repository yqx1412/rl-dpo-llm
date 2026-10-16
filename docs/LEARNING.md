# Understand the "why" behind each stage

You should be able to explain all of this in an interview. Keep it tight.

## 1. The problem RLHF/DPO solves

A pretrained LLM predicts likely next tokens; it isn't inherently *helpful* or
*aligned* with what humans prefer. **Preference optimization** nudges the model
toward responses humans rated higher, using pairs of (chosen, rejected) answers.

## 2. Classic RLHF (the PPO recipe) — know it, don't build it first

Four stages:
1. **Pretraining** — the base LLM.
2. **SFT** (supervised fine-tuning) — teach the format on curated examples.
3. **Reward Model** — train a model to score responses from human rankings.
4. **PPO** — reinforcement learning that pushes the policy to maximize the reward
   model's score, with a KL penalty keeping it near the SFT model.

Downsides: three models in memory, reward-hacking, unstable PPO. This is why DPO
exists.

## 3. DPO — what this project builds

**Direct Preference Optimization** collapses stages 3+4. Key insight: the optimal
RLHF policy has a closed-form relationship to the reward, so you can skip the
explicit reward model and optimize the policy *directly* on preference pairs with a
simple binary-classification-style loss.

- Inputs: `(prompt, chosen, rejected)` triples.
- Loss: increase the policy's log-prob margin of `chosen` over `rejected`,
  measured *relative to a frozen reference model*.
- **β (beta)**: controls how far the policy may drift from the reference. Lower β =
  more aggressive change (and more drift risk); higher β = conservative.

One policy + one frozen reference. Stable, cheap, single-GPU. That's the whole pitch.

## 4. Evaluation: win-rate

Preference quality isn't captured by loss alone. The standard metric is
**win-rate**: on held-out prompts, how often does an impartial judge prefer the
aligned model's answer over the base model's? 50% = no improvement.
We judge each pair in both orders to cancel position bias.

## 5. Where GRPO fits (your stretch goal)

**GRPO** (Group Relative Policy Optimization, used by DeepSeek-R1) is RL with
*verifiable* rewards (e.g. "is the math answer correct?") instead of a learned
preference. It samples a group of answers per prompt and reinforces the
above-average ones. Great for reasoning tasks. See `docs/NEXT_STEPS.md`.

## Reading list

- DPO paper — "Direct Preference Optimization: Your Language Model is Secretly a
  Reward Model" (Rafailov et al., 2023): [arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)
- HuggingFace TRL DPO docs: [huggingface.co/docs/trl/dpo_trainer](https://huggingface.co/docs/trl/dpo_trainer)
- InstructGPT (the original RLHF-for-LLMs paper): [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
- DeepSeek-R1 (GRPO for reasoning): [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)
- HuggingFace RLHF blog: [huggingface.co/blog/rlhf](https://huggingface.co/blog/rlhf)
