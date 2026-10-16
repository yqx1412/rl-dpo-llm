"""Gradio demo for the DPO-aligned model.

Loads the base model and the DPO LoRA adapter, then serves a chat UI. A toggle
lets you compare the aligned model against the untouched base side by side --
the quickest way to *see* what preference optimization changed.

Paths (base model, adapter dir) are read from the same config the training and
eval scripts use, so the demo always points at whatever you trained.

Usage:
    python app.py                                   # uses configs/dpo_config.yaml
    python app.py --config configs/dpo_config.yaml --share
    python app.py --base-only                       # demo without a trained adapter

The adapter is loaded from `output.adapter_dir` in the config. If it does not
exist yet (you haven't trained), start with --base-only to try the base model,
or train first (see README).
"""
import argparse
import os

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import gradio as gr
except ImportError as e:  # pragma: no cover - friendlier message than a traceback
    raise SystemExit(
        "gradio is not installed. Run: pip install gradio\n"
        "(it is in requirements.txt)."
    ) from e


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pick_device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def build_prompt(tokenizer, history, user_msg: str) -> str:
    """Render chat history + new turn with the model's chat template."""
    messages = []
    for turn in history:
        # Gradio 'messages' format: list of {role, content} dicts.
        if isinstance(turn, dict):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_msg})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def generate(model, tokenizer, text: str, max_new_tokens: int,
             temperature: float, top_p: float) -> str:
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-4),
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio demo for the DPO model.")
    parser.add_argument("--config", default="configs/dpo_config.yaml")
    parser.add_argument("--base-only", action="store_true",
                        help="Skip the adapter; serve only the base model.")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link (useful on Colab).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg, out_cfg, eval_cfg = cfg["model"], cfg["output"], cfg["eval"]
    base_name = model_cfg["base_model"]
    adapter_dir = out_cfg["adapter_dir"]
    default_max_new = eval_cfg.get("max_new_tokens", 256)

    device, dtype = pick_device_dtype()
    print(f"Device: {device}  dtype: {dtype}")

    print(f"Loading base model: {base_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    aligned_model = None
    have_adapter = (not args.base_only) and os.path.isdir(adapter_dir)
    if args.base_only:
        print("Running in --base-only mode (no adapter).")
    elif have_adapter:
        from peft import PeftModel
        print(f"Loading DPO adapter from: {adapter_dir}")
        aligned_base = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=dtype).to(device)
        aligned_model = PeftModel.from_pretrained(aligned_base, adapter_dir).to(device)
    else:
        print(f"No adapter found at '{adapter_dir}'. Serving the base model only.\n"
              f"Train first (see README) or pass --base-only to silence this.")

    def respond(user_msg, history, compare, max_new, temperature, top_p):
        prompt_text = build_prompt(tokenizer, history, user_msg)
        model_for_chat = aligned_model if aligned_model is not None else base_model
        aligned_reply = generate(
            model_for_chat, tokenizer, prompt_text, max_new, temperature, top_p)

        if compare and aligned_model is not None:
            base_reply = generate(
                base_model, tokenizer, prompt_text, max_new, temperature, top_p)
            reply = (
                f"**🟢 Aligned (DPO)**\n\n{aligned_reply}\n\n"
                f"---\n\n**⚪ Base (no DPO)**\n\n{base_reply}"
            )
        else:
            reply = aligned_reply

        history = history + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": reply},
        ]
        return history, history

    adapter_status = (
        "✅ DPO adapter loaded — chatting with the **aligned** model."
        if aligned_model is not None
        else "⚠️ No adapter loaded — chatting with the **base** model only. "
             "Train first, then restart."
    )

    with gr.Blocks(title="DPO-aligned LLM demo") as demo:
        gr.Markdown(
            f"# DPO-aligned LLM demo\n"
            f"Base model: `{base_name}` · {adapter_status}\n\n"
            f"Toggle **Compare base vs aligned** to see both responses "
            f"side by side — the clearest view of what DPO changed."
        )
        chatbot = gr.Chatbot(type="messages", height=440)
        state = gr.State([])
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask something…", scale=8, show_label=False,
                container=False, autofocus=True)
            send = gr.Button("Send", variant="primary", scale=1)
        with gr.Accordion("Options", open=False):
            compare = gr.Checkbox(
                label="Compare base vs aligned",
                value=aligned_model is not None,
                interactive=aligned_model is not None)
            max_new = gr.Slider(32, 512, value=default_max_new, step=16,
                                label="Max new tokens")
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05,
                                    label="Temperature (0 = greedy)")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
        clear = gr.Button("Clear")

        inputs = [msg, state, compare, max_new, temperature, top_p]
        send.click(respond, inputs, [chatbot, state]).then(
            lambda: "", None, msg)
        msg.submit(respond, inputs, [chatbot, state]).then(
            lambda: "", None, msg)
        clear.click(lambda: ([], []), None, [chatbot, state])

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
