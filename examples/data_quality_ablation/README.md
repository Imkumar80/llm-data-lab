# Data-quality ablation example

This example shows how to compare preprocessing choices with the same input
corpus and record the data-side consequences reproducibly.

## Conditions

| Condition | Steps |
|---|---|
| `raw` | Validate JSONL schema and copy without text transformations |
| `clean` | Conservative Unicode/text cleanup |
| `clean_dedupe` | Clean, then remove exact duplicate text |
| `full_filter` | Clean, exact deduplication, then configurable length/repetition filtering |

## Run

From the repository root after installing the package:

```bash
pip install -e ".[dev]"
python examples/data_quality_ablation/run_ablation.py \
  --input examples/data_quality_ablation/sample.jsonl \
  --output-dir runs/data_quality_ablation \
  --min-chars 30 \
  --max-chars 1000 \
  --min-unique-word-ratio 0.30 \
  --seed 42
```

Outputs include one `corpus.jsonl` and `manifest.json` per condition, a
filter rejection report for `full_filter`, and `ablation_summary.json` with
corpus statistics and SHA-256 checksums.

## Interpretation and limitations

This is a **data-preparation ablation**, not evidence that a filter improves
model quality. The script does not train or evaluate a language model. The
summary explicitly marks model evaluation as not run.

For a downstream model-quality experiment:

1. Create a fixed, independently held-out validation/test set before training.
   Do not run the ablation filters on that held-out set.
2. Keep the base model, tokenizer, training token budget, sequence length,
   optimizer, learning-rate schedule, training steps, and evaluation protocol
   fixed across conditions where possible.
3. Run each condition with the same held-out examples; record model and
   tokenizer revisions, training config, seed, hardware, and metrics.
4. Compare both a fixed-token-budget setting and an all-retained-data setting;
   they answer different questions.
5. Inspect rejected examples and report negative or inconclusive results.

The included sample data is synthetic and only demonstrates the pipeline. It
must not be used to draw conclusions about real-world model performance.

## Real public-data run

To fetch the pinned 300-row AG News sample, run all four conditions, and write
the report at `docs/ag_news_ablation_report.md`:

```bash
python examples/data_quality_ablation/run_public_dataset.py
```

The script checks the Hugging Face repository revision and sample SHA-256 before
running. The downloaded sample and condition corpora are stored under the
gitignored `data/ag_news_ablation/` directory. The committed
[report](../../docs/ag_news_ablation_report.md) contains corpus statistics and
provenance, not article text. Hugging Face lists the license as unknown, while
the dataset card describes research and other non-commercial uses; check the
source card and upstream terms before reuse.
