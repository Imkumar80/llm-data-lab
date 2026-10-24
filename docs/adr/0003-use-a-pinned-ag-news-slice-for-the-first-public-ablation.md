# 0003. Use a pinned AG News slice for the first public-data ablation

- **Status:** Accepted
- **Date:** 2026-10-24
- **Decision makers:** Project maintainer
- **Supersedes:** None

## Context

The four-condition ablation had only been exercised with synthetic examples. A small public corpus run was needed to demonstrate the workflow and produce a concrete report without training a model or adding a large dataset to the repository. The [AG News dataset card](https://huggingface.co/datasets/fancyzhx/ag_news) exposes a training split with four labels through the Hugging Face Dataset Viewer.

## Decision

Use the first 300 rows of `fancyzhx/ag_news`, config `default`, split `train`, for the initial public-data demonstration. Record the Hub revision and exact downloaded JSONL checksum, and stop the acquisition script if either changes. Keep downloaded rows and generated corpora under gitignored `data/`; publish only aggregate statistics, provenance, and limitations in the [ablation report](../ag_news_ablation_report.md).

The initial comparison uses the existing defaults: raw, cleaned, cleaned plus exact deduplication, and cleaned plus exact deduplication and filtering with a 200-character minimum, 100,000-character maximum, and 0.30 minimum unique-word ratio.

## Consequences

The run exercises the workflow on real labeled text and can be reproduced while the pinned source remains available. It does not measure downstream model quality. The first 300 rows are a source-order prefix, not a representative sample; this run contains no World or Sports records, so its retention figures must not be generalized to the full dataset. The Hub lists the license as unknown, and the dataset card describes research and non-commercial use; verify upstream terms before reusing the data. A model-quality study requires a separately designed, fixed held-out set and matched training runs.
