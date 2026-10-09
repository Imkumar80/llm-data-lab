# LLM Data Lab

**Reproducible data preparation and experiment utilities for language-model research.**

LLM Data Lab turns raw text datasets into auditable training corpora. It is being developed from lessons in [KannadaSLM](https://github.com/Imkumar80/KannadaSLM), with language-specific rules kept configurable rather than hard-coded.

## Current scope

- Streaming JSONL validation and conservative Unicode/text cleanup
- Exact deduplication with SQLite-backed fingerprints
- Configurable document-length and repetition filters
- Deterministic train/validation splitting without loading the whole corpus into RAM
- Corpus statistics and machine-readable reports
- Unit tests and CI

Training runners, tokenizer-aware token counts, experiment manifests, and model evaluation are planned next. This initial release does **not** claim to provide a complete training/evaluation framework yet.

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
```

All commands support `--help`. Use `--text-field` to select a different text column. Filter defaults are general-purpose length/repetition heuristics; they do not detect language. Add language-specific checks only when appropriate for the corpus.

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

## Development

```bash
pytest
```

## Roadmap

1. [x] General-purpose cleaning, exact deduplication, filtering, splitting, and statistics
2. [x] Unit tests and continuous integration
3. [ ] Dataset manifests with source/version/hash and pipeline configuration
4. [ ] Tokenizer-aware corpus token counts and train/validation leakage checks
5. [ ] Config-driven CPT/SFT experiment runner
6. [ ] Standardized evaluation adapters and machine-readable reports
7. [ ] Data-quality ablation example using a small public dataset

## Origin

The initial design is informed by data curation and evaluation work in KannadaSLM. Kannada-specific script-ratio rules remain project-specific and should be implemented as an optional plugin rather than a universal default.

## License

MIT. See [LICENSE](LICENSE).
