# 0002. Keep model runtimes out of the data-preparation core

- **Status:** Accepted
- **Date:** 2026-10-24
- **Decision makers:** Project maintainer
- **Supersedes:** None

## Context

The repository currently provides general-purpose corpus preparation and reproducibility utilities. Model training and evaluation need additional frameworks, hardware, and runtime configuration. Making those dependencies part of the core would make common cleaning and filtering workflows heavier to install and harder to use independently.

## Decision

Keep the core focused on dependency-light data preparation, manifests, and corpus diagnostics. Add training and model evaluation as optional, separately scoped research workflows that consume prepared corpora and their manifests.

## Consequences

Users can prepare data without installing a model stack, and a corpus artifact remains usable by different training systems. Future model workflows must record their model, tokenizer, data, and evaluation configuration and should not duplicate the core preprocessing logic. The current boundary and evaluation direction are described in the [experiment design](../experiment_design.md) and [evaluation design](../evaluation_design.md) documents.
