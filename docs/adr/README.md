# Architecture and research decision records

This directory keeps the reasoning behind choices that affect the project structure, reproducibility, or research workflow. Each record describes one decision in Markdown so it can be reviewed and versioned with the code.

## Using the log

- Create one file per decision, named `NNNN-short-imperative-title.md`; take the next available number.
- Use the [template](template.md) and write for someone who was not part of the original discussion.
- Mark a decision `Proposed`, `Accepted`, `Rejected`, or `Superseded` and include its date.
- Keep accepted records as historical context. If a decision changes, create a new record that links to the record it supersedes.
- Record significant architecture, data-provenance, and experiment-method decisions. Routine implementation details belong in code comments or project documentation.

## Records

| ID | Decision | Status | Date |
|---|---|---|---|
| [0001](0001-record-decisions-in-the-repository.md) | Keep decisions in version-controlled Markdown | Accepted | 2026-10-24 |
| [0002](0002-keep-model-runtimes-out-of-the-data-core.md) | Keep model runtimes out of the data-preparation core | Accepted | 2026-10-24 |
| [0003](0003-use-a-pinned-ag-news-slice-for-the-first-public-ablation.md) | Use a pinned AG News slice for the first public-data ablation | Accepted | 2026-10-24 |

The record format follows the small, per-decision Markdown approach described by the [Architecture Decision Record project](https://github.com/architecture-decision-record/architecture-decision-record). This repository uses its own concise template and conventions.
