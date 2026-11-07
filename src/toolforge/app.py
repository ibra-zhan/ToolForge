"""Minimal Gradio demo for comparing base and fine-tuned routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .inference import ToolCallingModel
from .schemas import normalized_output
from .tools import execute


class DemoModels:
    def __init__(self, model_id: str, adapter_path: str) -> None:
        self.model_id = model_id
        self.adapter_path = adapter_path
        self._models: dict[str, ToolCallingModel] = {}

    def get(self, variant: str) -> ToolCallingModel:
        if variant not in self._models:
            adapter = self.adapter_path if variant == "Fine-tuned" else None
            if adapter and not Path(adapter).exists():
                raise FileNotFoundError(f"Adapter not found: {adapter}. Train it first.")
            self._models[variant] = ToolCallingModel(self.model_id, adapter)
        return self._models[variant]


def run_request(models: DemoModels, request: str, variant: str) -> tuple[str, str, str]:
    raw, elapsed = models.get(variant).generate(request, few_shot=variant == "Base (few-shot)")
    try:
        parsed = normalized_output(raw)
        validation = f"Valid schema ({elapsed:.2f}s)"
        result = execute(parsed)
        return raw, validation, json.dumps(result, indent=2)
    except Exception as exc:
        return raw, f"Invalid: {exc} ({elapsed:.2f}s)", "Not executed"


def build_demo(models: DemoModels) -> Any:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install demo dependencies: pip install -e '.[demo,inference]'") from exc
    with gr.Blocks(title="ToolForge") as demo:
        gr.Markdown("# ToolForge\nCompare structured tool calls. Every result is simulated.")
        request = gr.Textbox(label="Natural-language request", placeholder="What's on my calendar tomorrow?")
        variant = gr.Radio(["Base (zero-shot)", "Base (few-shot)", "Fine-tuned"], value="Base (zero-shot)", label="Model")
        run = gr.Button("Generate tool call", variant="primary")
        raw = gr.Code(label="Raw model output", language="json")
        validation = gr.Textbox(label="Validation")
        result = gr.Code(label="Mock execution result", language="json")
        run.click(lambda req, var: run_request(models, req, var), [request, variant], [raw, validation, result])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    build_demo(DemoModels(config["model_id"], config["adapter_path"])).launch(share=args.share)


if __name__ == "__main__":
    main()

