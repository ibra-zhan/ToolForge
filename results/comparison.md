# ToolForge experiment results

Status: **both base-model baselines completed locally; QLoRA training and fine-tuned evaluation are pending**.

Do not fill pending values with estimates. After QLoRA training, run the fine-tuned configuration on the identical `data/test.jsonl` set.

| Metric | Zero-shot base | Few-shot base | Fine-tuned QLoRA |
|---|---:|---:|---:|
| Schema validity | 72.34% | 100.00% | — |
| Tool selection accuracy | 6.25% | 37.50% | — |
| Argument exact match | 0.00% | 0.00% | — |
| Full decision accuracy | 4.26% | 17.02% | — |
| Hallucinated tool rate | 0.00% | 0.00% | — |
| Clarification accuracy | 0.00% | 12.50% | — |
| Unsupported-request accuracy | 28.57% | 100.00% | — |
| Mean latency (seconds) | 0.875 | 0.784 | — |

## Run metadata

- Base model: `Qwen/Qwen3-0.6B`
- Dataset seed: `42`
- Reference date: `2026-11-07`
- Test examples: `47`
- Baseline hardware: Apple M4, 16 GB unified memory (MPS)
- Baseline packages: Python 3.14.6, PyTorch 2.15.0, Transformers 5.22.2
- Decoding: deterministic (`do_sample=False`), maximum 256 new tokens
- GPU used for training: —
- Adapter checkpoint: —

## Initial error analysis

Few-shot prompting fixed schema adherence and unsupported-request classification, but it did not solve exact argument generation. The most common few-shot problems were wrong output type (13), incorrect arguments (12), incorrect tool (7), and failure to clarify (7). Zero-shot also produced 13 malformed or schema-invalid responses, including Markdown code fences that were intentionally not repaired during scoring.

One representative zero-shot error selected `get_calendar_events` for “tomorrow” but emitted the reference date (`2026-11-07`) rather than `2026-11-08`. The fine-tuning dataset explicitly contains deterministic relative-date examples intended to address this class of failure.

The raw predictions and detailed per-example scores are stored beside this report. Fine-tuned values must remain blank until the adapter is trained and evaluated on the identical held-out set.
