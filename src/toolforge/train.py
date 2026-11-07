"""QLoRA supervised fine-tuning entry point (intended for a CUDA GPU)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def to_prompt_completion(example: dict[str, Any]) -> dict[str, Any]:
    """Convert one conversational row to TRL's completion-loss format."""
    messages = example.get("messages", [])
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("Each training example must end with an assistant message")
    return {
        "prompt": messages[:-1],
        "completion": [messages[-1]],
    }


def train(model_config: Path, training_config: Path) -> None:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("Install GPU training dependencies: pip install -e '.[training]'") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a supported GPU. Use the Colab notebook for this project.")

    model_cfg = load_yaml(model_config)
    train_cfg = load_yaml(training_config)
    set_seed(int(train_cfg["seed"]))
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_id"],
        quantization_config=quantization,
        device_map="auto",
        dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = load_dataset(
        "json",
        data_files={"train": "data/train.jsonl", "validation": "data/validation.jsonl"},
    )
    data = data.map(
        to_prompt_completion,
        remove_columns=data["train"].column_names,
        desc="Converting conversations to prompt/completion examples",
    )
    lora = LoraConfig(
        r=int(train_cfg["lora_rank"]),
        lora_alpha=int(train_cfg["lora_alpha"]),
        lora_dropout=float(train_cfg["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    output_dir = model_cfg["adapter_path"]
    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=float(train_cfg["epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        max_length=int(train_cfg["max_length"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        logging_steps=int(train_cfg["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(train_cfg["eval_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        gradient_checkpointing=True,
        assistant_only_loss=False,
        completion_only_loss=True,
        packing=False,
        report_to="none",
        seed=int(train_cfg["seed"]),
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved adapter and tokenizer to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/training.yaml"))
    args = parser.parse_args()
    train(args.model_config, args.training_config)


if __name__ == "__main__":
    main()
