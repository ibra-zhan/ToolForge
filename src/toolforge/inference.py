"""Base and adapter-backed inference with identical generation settings."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .dataset_validator import load_jsonl
from .prompts import build_messages


class ToolCallingModel:
    def __init__(self, model_id: str, adapter_path: str | None = None, device_map: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install inference dependencies: pip install -e '.[inference]'") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            resolved_device_map: str | dict[str, str] = device_map
        elif torch.backends.mps.is_available():
            dtype = torch.float16
            resolved_device_map = {"": "mps"}
        else:
            dtype = torch.float32
            resolved_device_map = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=resolved_device_map)
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("PEFT is required to load an adapter") from exc
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def generate(self, request: str, few_shot: bool = False, max_new_tokens: int = 256) -> tuple[str, float]:
        messages = build_messages(request, few_shot=few_shot)
        chat_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if "qwen3" in self.tokenizer.name_or_path.casefold():
            chat_kwargs["enable_thinking"] = False
        prompt = self.tokenizer.apply_chat_template(messages, **chat_kwargs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        elapsed = time.perf_counter() - started
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), elapsed


def run_dataset(model: ToolCallingModel, dataset_path: Path, output_path: Path, few_shot: bool = False) -> None:
    rows = []
    for example in load_jsonl(dataset_path):
        request = example["messages"][1]["content"]
        prediction, latency = model.generate(request, few_shot=few_shot)
        rows.append({"id": example["id"], "prediction": prediction, "latency_seconds": latency})
        print(f"{len(rows)}: {example['id']} -> {prediction[:100]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter")
    parser.add_argument("--dataset", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--few-shot", action="store_true")
    args = parser.parse_args()
    run_dataset(ToolCallingModel(args.model_id, args.adapter), args.dataset, args.output, args.few_shot)


if __name__ == "__main__":
    main()
