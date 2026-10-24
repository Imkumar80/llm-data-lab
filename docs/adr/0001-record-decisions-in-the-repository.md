# 0001. Record decisions in the repository

- **Status:** Accepted
- **Date:** 2026-10-24
- **Decision makers:** Project maintainer
- **Supersedes:** None

## Context

LLM Data Lab is adding research workflows alongside its data-preparation code. Their rationale can otherwise become scattered across implementation notes, experiment plans, and code history. The project needs a small, durable place to record why consequential choices were made.

## Decision

Keep one Markdown file per significant architecture or research-workflow decision in `docs/adr/`. Maintain a linked index and use a consistent context, decision, and consequences format. Use sequential numeric IDs and create a new record when an accepted decision changes.

## Consequences

The reason for a choice can be reviewed with the code and linked to the implementation or report it affects. The process adds a small amount of writing and is reserved for choices with lasting impact; routine changes stay in the usual docs, commits, and code reviews.
