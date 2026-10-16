"""Train a small LLM with Direct Preference Optimization (DPO) + LoRA.

Uses HuggingFace TRL's DPOTrainer. DPO needs no separate reward model: it
optimizes the policy directly on (prompt, chosen, rejected) triples against a
frozen reference copy of the base model, with `beta` controlling how far the
policy may drift from that reference.

Usage:
    python src/train_dpo.py --config configs/dpo_config.yaml
"""
import argparse

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pick_dtype(cfg_dpo: dict):
    if cfg_dpo.get("bf16") and torch.cuda.is_available():
        return torch.bfloat16
    if cfg_dpo.get("fp16") and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO + LoRA training.")
    parser.add_argument("--config", default="configs/dpo_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    model_cfg, data_cfg = cfg["model"], cfg["data"]
    lora_cfg, dpo_cfg, out_cfg = cfg["lora"], cfg["dpo"], cfg["output"]

    print(f"Loading base model: {model_cfg['base_model']}")
    dtype = pick_dtype(dpo_cfg)

    model_kwargs = {"torch_dtype": dtype}
    if dpo_cfg.get("use_qlora"):
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"], **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    print(f"Loading prepared data from: {data_cfg['prepared_dir']}")
    train_ds = load_from_disk(f"{data_cfg['prepared_dir']}/train")

    training_args = DPOConfig(
        output_dir=out_cfg["adapter_dir"],
        beta=dpo_cfg["beta"],
        learning_rate=dpo_cfg["learning_rate"],
        num_train_epochs=dpo_cfg["num_train_epochs"],
        per_device_train_batch_size=dpo_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=dpo_cfg["gradient_accumulation_steps"],
        warmup_ratio=dpo_cfg["warmup_ratio"],
        logging_steps=dpo_cfg["logging_steps"],
        save_steps=dpo_cfg["save_steps"],
        max_length=model_cfg["max_length"],
        max_prompt_length=model_cfg["max_prompt_length"],
        bf16=bool(dpo_cfg.get("bf16")) and torch.cuda.is_available(),
        fp16=bool(dpo_cfg.get("fp16")) and torch.cuda.is_available(),
        report_to="none",
    )

    # ref_model=None -> TRL builds the frozen reference by disabling the
    # LoRA adapters on the same base model (memory-efficient).
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting DPO training...")
    trainer.train()

    print(f"Saving LoRA adapter to: {out_cfg['adapter_dir']}")
    trainer.save_model(out_cfg["adapter_dir"])
    tokenizer.save_pretrained(out_cfg["adapter_dir"])

    print("Done. Next: python src/evaluate.py --config", args.config)


if __name__ == "__main__":
    main()
