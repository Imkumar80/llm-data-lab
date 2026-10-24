# Designing reproducible data-quality ablations

This protocol is a starting point for studying how corpus preparation affects language-model adaptation. It is a plan, not a claim that any filter improves model quality.

The staged corpus-mix and language-composition ideas are informed by [Indic-LLM](https://github.com/adithya-s-k/Indic-llm); here they are treated as optional, measurable experiment metadata rather than a prescribed Kannada training recipe.

## Research question

How do conservative cleaning, exact deduplication, and quality filtering affect the usable corpus and downstream model quality?

## Conditions

| Condition | Preparation |
|---|---|
| raw | Validate schema only; no text transformations |
| clean | Unicode/text cleanup |
| clean_dedupe | Cleaning followed by exact deduplication |
| full_filter | Cleaning, deduplication, then configured quality filters |

For multilingual or staged adaptation, describe the corpus mix for each training phase separately. Record each source's identifier, revision, license, language or script metadata when known, and intended share of the phase's token budget. Keep those labels descriptive: the data tools should report measured composition and must not infer language or apply script-specific rejection rules by default. Token shares should be computed with the tokenizer used by the experiment, not estimated from document or character counts.

## Controls

Keep the base model and tokenizer, training token budget, sequence length, optimizer, learning-rate schedule, number of updates, validation set, and evaluation code fixed. Record any unavoidable deviations. Use multiple seeds for final conclusions when compute permits.

Two complementary comparisons are useful:
1. **Fixed token budget:** isolates data quality while keeping training exposure similar.
2. **All retained data:** measures the practical trade-off when filtering changes corpus size.

Do not conflate these comparisons.

## Metrics

- Corpus: documents and characters retained, rejection reasons, exact duplicate count, source distribution.
- Language modeling: held-out loss/perplexity, evaluated with the same tokenizer and loss protocol.
- Generation: task accuracy where ground truth exists, empty-output rate, repetition measures, and manual quality review.
- Efficiency: actual token count, preprocessing time, training throughput, and GPU memory.

For multilingual corpora, also report token and document counts by source and by any supplied language/script fields. These are diagnostics for mix review, not quality scores or automatic filtering criteria. Tokenizer efficiency comparisons should identify the tokenizer and revision and use the same text sample.

Perplexity is tokenizer-dependent. Compare it directly only when tokenization and evaluation protocol are compatible.

## Leakage and data hygiene

- Deduplicate before splitting.
- Keep a fixed held-out validation set across conditions.
- Record source dataset names, versions, licenses, download dates, and checksums.
- Do not publish raw data unless its license and terms allow redistribution.
- Inspect examples rejected by each filter; script-ratio and repetition heuristics can discard legitimate content.
- Report negative or inconclusive results as well as improvements.

## Run manifest

Each experiment should save a JSON manifest containing:
- run name and timestamp
- code commit
- input dataset identifier and checksum
- pipeline configuration and seed
- model/tokenizer identifiers and revisions
- training configuration
- evaluation dataset/version and metrics
- hardware/software environment

The first implementation of LLM Data Lab covers data preparation. Model training and evaluation orchestration are roadmap items.

## Evaluation workflow

The planned text evaluation format and result artifact are described in [evaluation design](evaluation_design.md). Keep a fixed benchmark split across data-preparation conditions and record the benchmark revision, model/tokenizer revisions, generation settings, seed, and metric definition with every result.
