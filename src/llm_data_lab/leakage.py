"""Exact text overlap checks across supplied dataset splits."""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .pipeline import _software_environment, read_jsonl, sha256_file

NORMALIZATION_POLICIES = ("exact", "nfc_casefold_whitespace")


def _normalize(text: str, policy: str) -> str:
    if policy == "exact":
        return text
    if policy == "nfc_casefold_whitespace":
        return " ".join(unicodedata.normalize("NFC", text).casefold().split())
    raise ValueError(
        f"Unknown normalization policy {policy!r}; choose from {', '.join(NORMALIZATION_POLICIES)}"
    )


def check_split_overlap(
    splits: Mapping[str, str | Path | Iterable[Mapping[str, Any]]],
    *,
    text_field: str = "text",
    normalization: str = "exact",
) -> dict[str, Any]:
    """Report exact normalized-text overlap without returning document contents.

    ``exact`` compares the text field byte-for-codepoint as decoded from JSON.
    ``nfc_casefold_whitespace`` applies Unicode NFC, case folding, and collapses
    all whitespace runs to one space before comparison. This detects exact text
    overlap under that policy only; it does not detect semantic or pretraining leakage.
    """
    if normalization not in NORMALIZATION_POLICIES:
        _normalize("", normalization)
    if len(splits) < 2:
        raise ValueError("At least two named splits are required")
    if any(not isinstance(name, str) or not name for name in splits):
        raise ValueError("Split names must be non-empty strings")
    contents: dict[str, dict[str, int]] = {}
    row_counts: dict[str, int] = {}
    checksums: dict[str, str | None] = {}
    for name, source in splits.items():
        records = read_jsonl(source) if isinstance(source, (str, Path)) else source
        checksums[name] = sha256_file(source) if isinstance(source, (str, Path)) else None
        seen: dict[str, int] = {}
        for record_number, row in enumerate(records, 1):
            if not isinstance(row, Mapping):
                raise TypeError(f"Split {name!r}, record {record_number}: expected an object")
            text = row.get(text_field)
            if not isinstance(text, str):
                raise TypeError(
                    f"Split {name!r}, record {record_number}: {text_field!r} must be a string"
                )
            normalized = _normalize(text, normalization)
            key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            seen[key] = seen.get(key, 0) + 1
        contents[name] = seen
        row_counts[name] = sum(seen.values())

    pairs: list[dict[str, Any]] = []
    affected_texts = {name: set() for name in contents}
    names = list(contents)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            common = contents[left].keys() & contents[right].keys()
            left_records = sum(contents[left][text] for text in common)
            right_records = sum(contents[right][text] for text in common)
            pairs.append(
                {
                    "splits": [left, right],
                    "unique_overlapping_texts": len(common),
                    "overlapping_documents": {left: left_records, right: right_records},
                }
            )
            if common:
                affected_texts[left].update(common)
                affected_texts[right].update(common)
    affected_counts = {
        name: sum(contents[name][text] for text in overlapping)
        for name, overlapping in affected_texts.items()
    }
    affected = [name for name, count in affected_counts.items() if count]
    return {
        "schema_version": 1,
        "normalization": normalization,
        "text_field": text_field,
        "split_documents": row_counts,
        "split_sha256": checksums,
        "configuration": {"text_field": text_field, "normalization": normalization},
        "software": _software_environment(),
        "affected_splits": affected,
        "overlapping_documents_by_split": affected_counts,
        "pairs": pairs,
        "has_overlap": bool(affected),
    }
