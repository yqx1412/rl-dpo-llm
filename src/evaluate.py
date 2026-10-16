"""Evaluate the DPO-aligned model vs. the base model via win-rate.

For each held-out prompt we generate a response from (a) the base model and
(b) the DPO-aligned model, then ask a judge which response is better. The
headline metric is the aligned model's **win-rate** against the base.

To avoid position bias, each pair is judged in BOTH orders and ties are counted
as half a win for each side.

Usage:
    python src/evaluate.py --config configs/dpo_config.yaml --num-eval 100
    python src/evaluate.py --config configs/dpo_config.yaml --judge local
"""
import argparse
import json
import os
import re

import torch
import yaml
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_prompt(row) -> str:
    """Pull the user prompt text from an ultrafeedback-style chosen message."""
    chosen = row.get("chosen")
    if isinstance(chosen, list):
        for msg in chosen:
            if msg.get("role") == "user":
                return msg["content"]
    if isinstance(row.get("prompt"), str):
        return row["prompt"]
    return str(chosen)


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


JUDGE_PROMPT = """You are comparing two AI assistant responses to the same user prompt.
Pick the response that is more helpful, correct, and clear.

[User prompt]
{prompt}

[Response A]
{a}

[Response B]
{b}

Reply with EXACTLY one token: "A", "B", or "TIE". No explanation."""


def judge_openai(client, model_name, prompt, a, b) -> str:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user",
                   "content": JUDGE_PROMPT.format(prompt=prompt, a=a, b=b)}],
        temperature=0,
        max_tokens=4,
    )
    return resp.choices[0].message.content.strip().upper()


def judge_local(judge_model, judge_tok, prompt, a, b) -> str:
    text = JUDGE_PROMPT.format(prompt=prompt, a=a, b=b)
    messages = [{"role": "user", "content": text}]
    chat = judge_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = judge_tok(chat, return_tensors="pt").to(judge_model.device)
    with torch.no_grad():
        out = judge_model.generate(**inputs, max_new_tokens=4, do_sample=False)
    ans = judge_tok.decode(out[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True).upper()
    return ans


def parse_verdict(raw: str) -> str:
    m = re.search(r"\b(A|B|TIE)\b", raw.upper())
    return m.group(1) if m else "TIE"


def main() -> None:
    parser = argparse.ArgumentParser(description="Win-rate eval: DPO vs base.")
    parser.add_argument("--config", default="configs/dpo_config.yaml")
    parser.add_argument("--num-eval", type=int, default=100)
    parser.add_argument("--judge", default=None,
                        help="Override config judge: 'openai' or 'local'.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg, data_cfg = cfg["model"], cfg["data"]
    out_cfg, eval_cfg = cfg["output"], cfg["eval"]
    judge_kind = args.judge or eval_cfg["judge"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"], torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading DPO-aligned model (base + adapter)...")
    aligned_base = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"], torch_dtype=dtype).to(device)
    aligned = PeftModel.from_pretrained(aligned_base, out_cfg["adapter_dir"]).to(device)

    # Judge setup
    oai_client = local_judge = local_judge_tok = None
    if judge_kind == "openai":
        from openai import OpenAI
        oai_client = OpenAI()  # reads OPENAI_API_KEY
    else:
        print("Loading local judge model...")
        local_judge = AutoModelForCausalLM.from_pretrained(
            eval_cfg["local_judge_model"], torch_dtype=dtype).to(device)
        local_judge_tok = AutoTokenizer.from_pretrained(eval_cfg["local_judge_model"])

    test_ds = load_from_disk(f"{data_cfg['prepared_dir']}/test")
    n = min(args.num_eval, len(test_ds))
    max_new = eval_cfg["max_new_tokens"]

    aligned_score = 0.0
    records = []

    for i in range(n):
        prompt = extract_prompt(test_ds[i])
        base_resp = generate(base, tokenizer, prompt, max_new)
        aligned_resp = generate(aligned, tokenizer, prompt, max_new)

        # Judge in both orders to cancel position bias.
        # Order 1: A=aligned, B=base
        if judge_kind == "openai":
            v1 = parse_verdict(judge_openai(
                oai_client, eval_cfg["judge_model"], prompt, aligned_resp, base_resp))
            v2 = parse_verdict(judge_openai(
                oai_client, eval_cfg["judge_model"], prompt, base_resp, aligned_resp))
        else:
            v1 = parse_verdict(judge_local(
                local_judge, local_judge_tok, prompt, aligned_resp, base_resp))
            v2 = parse_verdict(judge_local(
                local_judge, local_judge_tok, prompt, base_resp, aligned_resp))

        # v1: "A" means aligned won. v2: "B" means aligned won.
        round_score = 0.0
        for v, aligned_is in ((v1, "A"), (v2, "B")):
            if v == aligned_is:
                round_score += 0.5
            elif v == "TIE":
                round_score += 0.25
        aligned_score += round_score

        records.append({"prompt": prompt, "base": base_resp,
                        "aligned": aligned_resp, "v1": v1, "v2": v2,
                        "round_score": round_score})
        print(f"[{i+1}/{n}] round_score={round_score:.2f} "
              f"running_winrate={aligned_score/(i+1):.3f}")

    win_rate = aligned_score / n if n else 0.0
    print("\n==================== RESULT ====================")
    print(f"Evaluated prompts: {n}")
    print(f"Aligned model win-rate vs base: {win_rate*100:.1f}%")
    print("(50% = no improvement; higher = DPO helped)")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/eval_results.json", "w") as f:
        json.dump({"num_eval": n, "win_rate": win_rate,
                   "judge": judge_kind, "records": records}, f, indent=2)
    print("Wrote outputs/eval_results.json")


if __name__ == "__main__":
    main()
