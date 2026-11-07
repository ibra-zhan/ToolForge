# ToolForge experiment results

Status: **complete — both baselines and the fine-tuned QLoRA adapter were evaluated on the same held-out test set**.

The results below are measured values from 47 template-family-held-out examples. Raw predictions and per-example score records are stored beside this report.

| Metric | Zero-shot base | Few-shot base | Fine-tuned QLoRA |
|---|---:|---:|---:|
| Schema validity | 72.34% | 100.00% | 100.00% |
| Tool selection accuracy | 6.25% | 37.50% | 37.50% |
| Argument exact match | 0.00% | 0.00% | 37.50% |
| Full decision accuracy | 4.26% | 17.02% | **57.45%** |
| Hallucinated tool rate | 0.00% | 0.00% | 0.00% |
| Clarification accuracy | 0.00% | 12.50% | 100.00% |
| Unsupported-request accuracy | 28.57% | 100.00% | 100.00% |
| Mean latency (seconds) | 0.875 | 0.784 | 2.794 |

## Run metadata

- Base model: `Qwen/Qwen3-0.6B`
- Dataset seed: `42`
- Reference date: `2026-11-07`
- Test examples: `47`
- Baseline hardware: Apple M4, 16 GB unified memory (MPS)
- Baseline packages: Python 3.14.6, PyTorch 2.15.0, Transformers 5.22.2
- Decoding: deterministic (`do_sample=False`), maximum 256 new tokens
- Training hardware: Google Colab NVIDIA T4
- Training method: 4-bit NF4 QLoRA, rank 16, alpha 32, dropout 0.05
- Training duration: 3 epochs, 63 optimizer steps
- Best checkpoint: step 25, validation loss 0.09597
- Selected adapter SHA-256: `88c65ac6dc731ddfaa5a7b203f88c8ed0d9ee858aec174c37eb7bc643a57884a`
- Fine-tuned inference hardware: Apple M4, 16 GB unified memory (MPS)

## Error analysis

Few-shot prompting fixed schema adherence and unsupported-request classification, but it did not solve exact argument generation. The most common few-shot problems were wrong output type (13), incorrect arguments (12), incorrect tool (7), and failure to clarify (7). Zero-shot also produced 13 malformed or schema-invalid responses, including Markdown code fences that were intentionally not repaired during scoring.

Fine-tuning produced a clear overall improvement: full-decision accuracy increased by 40.43 percentage points over few-shot prompting and by 53.19 points over zero-shot. It also raised exact argument accuracy from 0% to 37.50% while preserving perfect schema validity and eliminating failures on held-out clarification and unsupported examples.

The dominant remaining error is over-clarification. The adapter made all 8 required clarification decisions correctly, but unnecessarily requested clarification on 20 of the 32 valid tool requests. It correctly completed 12 valid tool calls. This suggests the small model learned the safe output classes more readily than the full mapping from diverse phrasing to tool arguments.

One representative zero-shot error selected `get_calendar_events` for “tomorrow” but emitted the reference date (`2026-11-07`) rather than `2026-11-08`. The fine-tuned model instead over-corrected on that example and asked which date to check. A larger and more diverse valid-call dataset, reduced clarification proportion, and evaluation of the 1.7B model are the strongest next experiments.

The fine-tuned model was also slower on the local MPS evaluation. Adapter inference averaged 2.794 seconds versus 0.784 seconds for the few-shot base model. Runtime comparisons should be repeated after warm-up and on identical GPU hardware before drawing deployment conclusions.
