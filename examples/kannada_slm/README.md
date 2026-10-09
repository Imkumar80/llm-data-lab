# KannadaSLM example

This folder contains only synthetic example records, not a copy of the KannadaSLM training corpus.

From the repository root after installing the package:

```bash
llm-data-lab clean --input examples/kannada_slm/sample.jsonl --output /tmp/clean.jsonl
llm-data-lab dedupe --input /tmp/clean.jsonl --output /tmp/deduped.jsonl
llm-data-lab filter --input /tmp/deduped.jsonl --output /tmp/filtered.jsonl \
  --min-chars 1 --min-unique-word-ratio 0
llm-data-lab split --input /tmp/filtered.jsonl --train /tmp/train.jsonl \
  --validation /tmp/validation.jsonl --val-fraction 0.34 --seed 42
llm-data-lab stats --input /tmp/filtered.jsonl --output /tmp/stats.json
```

The generic filter intentionally does not impose a Kannada-script ratio. A language-specific ratio filter should be implemented as an explicit, tested option, not applied to all datasets by default.
