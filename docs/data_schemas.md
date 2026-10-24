# Prepared data schemas

All formats are JSONL, UTF-8, one JSON object per line. Core dependencies remain optional; tokenizer commands resolve a Transformers tokenizer only when explicitly requested. The preparation API preserves row-level metadata when it can and never writes raw examples into reports.

## CPT raw text

Input records contain a non-empty string at `text` (or a selected `text_field`); other metadata is allowed. `prepare_cpt_dataset` writes text records with the same fields, enforcing a per-document sequence limit and optional total token budget. Overlong records use `error`, `drop`, or `truncate`. Records that do not fit the remaining budget are rejected whole. The report records input/output hashes, counts, rejected reasons, tokenizer identity and length distributions.

```python
from transformers import AutoTokenizer
from llm_data_lab.adapters.cpt import prepare_cpt_dataset

tokenizer = AutoTokenizer.from_pretrained("local/tokenizer")
report = prepare_cpt_dataset(
    "raw.jsonl", "prepared.jsonl", tokenizer=tokenizer,
    max_sequence_length=256, token_budget=200_000,
    overflow_policy="truncate", report_path="prepared.report.json",
)
```

The exported JSONL can be read by `datasets.load_dataset("json", data_files=...)`; sequence packing and training remain with the caller.

## SFT conversations

Each input object contains a non-empty `messages` list. Each message has `role` in `system`, `user`, or `assistant` and string `content`; at least one assistant response must be non-empty. The tokenizer must provide a compatible chat template that returns `input_ids` and assistant-span masks when called with `return_assistant_tokens_mask=True`.

Prepared rows preserve `messages` and add `input_ids` plus integer `assistant_mask` of the same length. The mask marks assistant target positions and is framework-neutral. It is not converted to trainer-specific `labels`. Overlength handling can error, drop, or truncate from a chosen side. Any result without at least one supervised token is rejected.

## DPO preference pairs

Each input object contains non-empty string fields `prompt`, `chosen`, and `rejected`. Equal answers are rejected. Prepared rows keep these fields and add `token_counts` for the prompt, each answer, and each prompt/answer sequence. Length limits apply to both prompt-plus-answer sequences. Truncation preserves the prompt and shortens answers; empty answers and pairs that become identical are rejected.

Repeated prompts and reversed chosen/rejected preferences are counted. The default conflict policy reports and preserves contradictory rows; `error` stops before writing the output. Use `check_dpo_split_overlap` or `dpo-overlap` to check prompt overlap across pair splits.

## Report stability and limits

The CPT, SFT, and DPO adapter reports use `schema_version: 1`, input/output SHA-256 checksums, counts, rejection reasons, configuration, tokenizer identity and software metadata. These adapters prepare data only. They do not claim semantic leakage detection or model-quality improvement.
