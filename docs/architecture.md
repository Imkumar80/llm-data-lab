# Data preparation architecture

The maintained package is intentionally small. `pipeline.py` owns streaming JSONL validation, cleaning, exact deduplication, filtering, deterministic splitting, corpus statistics, checksums, and run manifests. `token_stats.py` adds tokenizer-aware corpus diagnostics without importing Transformers unless the caller asks to load a tokenizer by name. `leakage.py` compares exact text across named splits and does not return document contents. `adapters/cpt.py`, `adapters/sft.py`, and `adapters/dpo.py` validate and export task-specific data but do not own training. `cli.py` exposes the same operations. GPU training and TRL integrations live only in examples.

## Optional tokenizer support

The mandatory package has no runtime dependencies. Install `llm-data-lab[transformers]` only when using the CLI's `stats --tokenizer NAME_OR_PATH` form. Python callers may pass a tokenizer object directly and avoid Transformers entirely. Token counts omit special tokens by default. Empty or whitespace-only records count as zero and are not passed to the tokenizer. A tokenizer exception is raised as a record-numbered `RuntimeError`; malformed text fields are rejected with a `TypeError`.

`token_stats` returns a versioned report with the input file checksum when given a path, tokenizer identifier/revision where available, document and empty-document counts, total tokens, min/max/mean, p50/p90/p95 and an exact token-length histogram. Token-length aggregation stores counts by distinct length rather than one token count per document.

## Split overlap contract

`check_split_overlap` accepts a mapping of split names to JSONL paths or iterables of records. It reports documents per split, affected split names, pairwise unique overlapping texts and overlapping record counts. It uses SHA-256 fingerprints internally and never includes document text or per-document hashes in reports; path inputs include file-level SHA-256 checksums for audit.

Normalization policies are stable API values:

- `exact` compares decoded text as supplied.
- `nfc_casefold_whitespace` applies Unicode NFC, Unicode case folding, and collapses whitespace runs before comparison.

These checks find exact overlap under the selected normalization only. They are not semantic deduplication and cannot determine whether training data appeared in model pretraining.

## Audit report fields

Existing operation report keys remain available. Core processing reports add input/output SHA-256 values, row counts, operation configuration and software metadata. Filters include retained/rejected counts and a stable `rejected_by_reason` mapping. Split reports include input and both output checksums. Manifests retain their existing top-level structure and now record the input row count alongside its checksum.

New tokenizer and overlap report objects include `schema_version: 1`. Token length histogram keys are decimal strings to remain valid JSON object keys. Tokenizer reports for iterable inputs use a null input checksum because no canonical serialized file is available. Reports contain aggregate values, file-level checksums and configuration, not raw corpus text. Overlap keeps one fixed-size fingerprint per unique document in memory, so memory use still grows with the number of unique records.

## Validation and scope

The unit tests are offline and use tiny synthetic records and fake tokenizers. A user must explicitly opt in to tokenizer resolution, dataset downloads, or example training. The adapter APIs do not train or evaluate a model, and data diagnostics must not be presented as model-quality evidence. See [data schemas](data_schemas.md) for the task-specific input/output contracts.
