# LLM Data Lab

**Reproducible data preparation and experiment utilities for language-model research.**

LLM Data Lab turns raw text datasets into auditable training corpora. It is being developed from lessons in [KannadaSLM](https://github.com/Imkumar80/KannadaSLM), with language-specific rules kept configurable rather than hard-coded.

## Current scope

- Streaming JSONL validation and conservative Unicode/text cleanup
- Exact deduplication with SQLite-backed fingerprints
- Configurable document-length and repetition filters
- Deterministic train/validation splitting without loading the whole corpus into RAM
- Corpus statistics, optional tokenizer-aware token-length profiles, and machine-readable reports
- Exact text overlap checks across supplied splits, with explicit normalization policies
- Preparation-only CPT, SFT, and DPO adapters with token limits and auditable outputs
- Dataset/run manifest generation with checksums and environment metadata
- A real-data four-condition ablation with pinned public sample provenance and a generated report
- Unit tests and CI

Transformers remains optional: install `pip install -e ".[transformers]"` to load a tokenizer by name in the CLI or use the adapter commands. Install `pip install -e ".[training]"` for the CPT smoke example and `pip install -e ".[trl]"` for the optional SFT/DPO trainer examples. Training remains in examples and uses Transformers/TRL; the core does **not** provide a trainer or orchestration framework.

## Quickstart

Requires Python 3.10+.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -e ".[dev]"
```

Input data is JSON Lines (one JSON object per line) with a text field, by default `text`. Extra fields are preserved.

```bash
llm-data-lab clean --input data/raw.jsonl --output data/cleaned.jsonl
llm-data-lab dedupe --input data/cleaned.jsonl --output data/deduped.jsonl
llm-data-lab filter --input data/deduped.jsonl --output data/filtered.jsonl \
  --min-chars 200 --max-chars 100000 --min-unique-word-ratio 0.30
llm-data-lab split --input data/filtered.jsonl --train data/train.jsonl \
  --validation data/validation.jsonl --val-fraction 0.02 --seed 42
llm-data-lab stats --input data/filtered.jsonl --output reports/corpus_stats.json
llm-data-lab stats --input data/filtered.jsonl --tokenizer /path/to/tokenizer \
  --tokenizer-revision abc123 --output reports/token_stats.json
llm-data-lab overlap --split train=data/train.jsonl --split validation=data/validation.jsonl \
  --normalization exact --output reports/leakage.json
llm-data-lab cpt-prepare --input data/raw.jsonl --output data/cpt.jsonl \
  --tokenizer /path/to/tokenizer --max-sequence-length 256 --token-budget 200000 \
  --overflow-policy truncate --report reports/cpt.json
llm-data-lab sft-prepare --input data/conversations.jsonl --output data/sft.jsonl \
  --tokenizer /path/to/chat-model --max-sequence-length 1024 --overflow-policy drop
llm-data-lab dpo-prepare --input data/preferences.jsonl --output data/dpo.jsonl \
  --tokenizer /path/to/model --max-sequence-length 1024 --conflict-policy report
llm-data-lab dpo-overlap --split train=data/dpo_train.jsonl \
  --split validation=data/dpo_validation.jsonl
llm-data-lab manifest --input data/raw.jsonl --output reports/run_manifest.json \
  --run-name raw-ablation --seed 42 --pipeline-config '{"condition":"raw"}'
```

All commands support `--help`. Use `--text-field` to select a different text column. `stats --tokenizer` uses a local Transformers tokenizer path or model ID and records its name and available revision; it avoids adding special tokens to counts. The Python API also accepts any object with `encode(text, add_special_tokens=False)` or a compatible callable. Empty/whitespace-only records count as zero tokens; tokenizer failures identify the record number.

`overlap` requires at least two `--split NAME=PATH` arguments. The default `exact` policy compares text exactly. `nfc_casefold_whitespace` applies Unicode NFC, case folding, and whitespace collapsing before comparing. Reports contain counts and split names only, never document text. These checks detect exact text overlap under the selected policy; they do not detect semantic similarity or model-pretraining exposure.

The `cpt-prepare`, `sft-prepare`, and `dpo-prepare` commands use the schemas and report contracts in [data schemas](docs/data_schemas.md). SFT input uses `messages`; DPO input uses `prompt`, `chosen`, and `rejected`. These commands validate and prepare data; they do not train a model.

Clean, dedupe, filter, split, stats, and manifest reports include checksums and configuration/software metadata as applicable. Report dictionaries use `schema_version: 1` for the new tokenizer and overlap report formats; existing report keys remain available, with additional audit fields added.

Filter defaults are general-purpose length/repetition heuristics; they do not detect language. Add language-specific checks only when appropriate for the corpus.

## Data format

Example `sample.jsonl`:

```json
{"text":"ನಮಸ್ಕಾರ, ಇದು ಒಂದು ಉದಾಹರಣೆ.","source":"example"}
{"text":"A second example document.","source":"example"}
```

Do not commit private, licensed, or large raw datasets, model weights, credentials, or generated checkpoints. Record dataset provenance and comply with the source dataset's license and terms.

## Reproducibility notes

- Cleaning is deterministic and preserves non-text metadata.
- Deduplication uses SHA-256 of cleaned text; it is exact deduplication, not semantic deduplication.
- Splits use a stable hash of the document text and seed, so the same input documents and seed produce the same assignment independent of input order. Exact duplicate documents should be removed before splitting to avoid train/validation leakage.
- Filter reports include rejection counts. Always inspect rejected examples before using aggressive thresholds.
- Corpus statistics are descriptive, not a substitute for a held-out model evaluation.
- Split-overlap checks are exact-text checks; run them on each pair of splits you need to validate.

## Development

```bash
pytest
```

## Roadmap

1. [x] General-purpose cleaning, exact deduplication, filtering, splitting, and statistics
2. [x] Unit tests and continuous integration
3. [x] Dataset manifests with source/version/hash and pipeline configuration
4. [x] Optional tokenizer-aware corpus token counts and exact train/validation overlap checks
5. [ ] Config-driven CPT/SFT experiment runner
6. [ ] Standardized evaluation adapters and machine-readable reports
7. [x] Data-quality ablation example using a small public dataset

For implemented core boundaries and adapter report fields, see [architecture](docs/architecture.md) and [data schemas](docs/data_schemas.md). The pinned [CPT smoke example](examples/cpt_smoke/README.md) is an opt-in GPU integration run; [TRL examples](examples/trl_integrations/README.md) show how to hand prepared SFT/DPO datasets to existing trainers. For the planned evaluation format, see [evaluation design](docs/evaluation_design.md).
The completed [AG News data-quality ablation report](docs/ag_news_ablation_report.md) documents one small, reproducible corpus-level run.
Long-lived architecture and research-workflow choices are recorded in the [decision log](docs/adr/README.md).

## Origin

The initial design is informed by data curation and evaluation work in KannadaSLM. Kannada-specific script-ratio rules remain project-specific and should be implemented as an optional plugin rather than a universal default.

## License

MIT. See [LICENSE](LICENSE).
