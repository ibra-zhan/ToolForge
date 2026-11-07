# ToolForge

ToolForge is a reproducible experiment that asks one focused question:

> Does supervised QLoRA fine-tuning make a small language model more reliable at converting natural-language requests into valid tool calls?

This is an AI engineering experiment, not a production assistant. All six tools operate on local mock data, and every returned result says that it is simulated.

## What the system does

Given a request such as `What's on my calendar tomorrow?`, the model must return exactly one of:

```json
{"type":"tool_call","tool":"get_calendar_events","arguments":{"date":"2026-11-08"}}
```

```json
{"type":"clarification","question":"Which date should I check?"}
```

```json
{"type":"unsupported","reason":"No available tool can perform this request."}
```

The model only proposes an action. Pydantic validation and a fixed Python allow-list decide whether a mock tool may run. No model output is passed to `eval` or arbitrary code execution.

## Architecture

```text
request -> prompt -> base/fine-tuned model -> strict JSON validation
                                              |
                                 valid tool call only
                                              v
                                    allow-listed mock tool
```

The first experiment compares the same Qwen3-0.6B model in three configurations:

1. Base model, zero-shot prompt
2. Base model, few-shot prompt
3. QLoRA adapter, zero-shot prompt

All configurations use the same tool definitions, held-out examples, deterministic decoding, and scoring rules.

## Quick start: local foundation

The local dataset and tests do not require a GPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
toolforge-generate --total 420
toolforge-validate
pytest
```

The generator uses a fixed reference date of `2026-11-07` and timezone `America/Indiana/Indianapolis`. It creates approximately 65% tool calls, 20% clarifications, and 15% unsupported requests. Template families—not individual rows—are assigned to train, validation, or test sets to reduce phrasing leakage.

## Train the QLoRA adapter

Training is intended for Google Colab or another machine with a compatible NVIDIA GPU. In a Colab terminal or notebook:

1. Create an upload bundle on your Mac with `git archive --format=zip --output=toolforge-colab.zip HEAD`.
2. Open `notebooks/train_colab.ipynb` in Google Colab.
3. Select **Runtime → Change runtime type → T4 GPU**.
4. Run the cells in order. When prompted, upload `toolforge-colab.zip`.

The notebook unpacks the project, installs its training dependencies, regenerates and validates the dataset, then starts QLoRA training. Its final cell downloads the trained adapter as `toolforge-qwen3-0.6b-adapter.zip`. The equivalent commands are:

```bash
pip install -e '.[training]'
toolforge-generate --total 420
toolforge-train
```

The training dependency ranges are intentionally pinned, including TRL 0.21.0 and Transformers 4.x. This keeps the documented `SFTConfig` interface reproducible instead of silently adopting breaking API changes from future releases.

The important defaults live in [`configs/training.yaml`](configs/training.yaml): 4-bit NF4 weights, LoRA rank 16, three epochs, assistant-response-only loss, and fixed seed 42. The resulting adapter is written under `artifacts/adapters/` and is intentionally ignored by Git.

The 0.6B model is deliberately chosen so the first end-to-end experiment finishes cheaply. If the run is stable and more GPU memory and disk space are available, repeat the experiment with Qwen3-1.7B or 4B; never compare a base model and adapter built from different model sizes.

## Run the three evaluations

Install inference dependencies on a machine able to load the model:

```bash
pip install -e '.[inference]'

python -m toolforge.inference \
  --output results/baseline_zero_shot_predictions.jsonl

python -m toolforge.inference \
  --few-shot \
  --output results/baseline_few_shot_predictions.jsonl

python -m toolforge.inference \
  --adapter artifacts/adapters/toolforge-qwen3-0.6b \
  --output results/fine_tuned_predictions.jsonl

toolforge-evaluate \
  --predictions results/baseline_zero_shot_predictions.jsonl \
  --output results/baseline_zero_shot.json
```

Repeat the final scoring command for the other two prediction files. The evaluator measures schema validity, output-type accuracy, exact arguments, full-decision accuracy, hallucinated tool rate, clarification accuracy, unsupported-request accuracy, latency, and per-category results. It never repairs malformed output before scoring.

## Run the demo

```bash
pip install -e '.[demo,inference]'
toolforge-demo
```

The first request can take time because model weights must be downloaded and loaded. The fine-tuned option becomes available after its adapter directory exists.

## Repository map

- `src/toolforge/schemas.py`: strict output and tool-argument contracts
- `src/toolforge/tools.py`: fixed registry and mock execution
- `src/toolforge/dataset_generator.py`: deterministic synthetic examples
- `src/toolforge/dataset_validator.py`: schema, duplicate, and leakage checks
- `src/toolforge/inference.py`: identical generation path for each configuration
- `src/toolforge/train.py`: 4-bit QLoRA training
- `src/toolforge/evaluate.py`: quantitative metrics and error labels
- `src/toolforge/app.py`: minimal Gradio interface
- `tests/`: CPU-only unit tests

## Experiment integrity

- Do not tune prompts against `data/test.jsonl` or `data/challenge.jsonl`.
- Preserve raw predictions, including malformed ones.
- Record GPU type, package versions, model ID, and training duration.
- Report a negative or inconclusive result honestly.
- Model-generated calls are untrusted until validation succeeds.

## Current limitations

- The synthetic language is intentionally controlled; real user language will be messier.
- Clarification and unsupported scoring checks the decision type, not exact wording.
- The mock tools do not access real email, calendar, contacts, or weather services.
- Zero-shot and few-shot baseline metrics are included; fine-tuned metrics remain pending until GPU training completes.
