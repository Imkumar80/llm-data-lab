"""Tokenizer-aware corpus diagnostics with no mandatory tokenizer dependency."""

from __future__ import annotations

import importlib
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .pipeline import _software_environment, read_jsonl, sha256_file


def _load_tokenizer(identifier: str, revision: str | None) -> Any:
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise ImportError(
            "Loading a tokenizer by name requires the optional Transformers dependency. "
            'Install it with `pip install "llm-data-lab[transformers]"`, or pass a tokenizer object.'
        ) from exc
    return transformers.AutoTokenizer.from_pretrained(identifier, revision=revision)


def _token_ids(tokenizer: Any, text: str) -> list[Any]:
    if hasattr(tokenizer, "encode"):
        result = tokenizer.encode(text, add_special_tokens=False)
    else:
        result = tokenizer(text, add_special_tokens=False)
        if isinstance(result, Mapping):
            result = result.get("input_ids")
    if hasattr(result, "tolist"):
        result = result.tolist()
    if result is None:
        raise TypeError("tokenizer did not return token IDs")
    if isinstance(result, (str, bytes)):
        raise TypeError("tokenizer output must be a sequence of token IDs")
    try:
        result = list(result)
    except TypeError as exc:
        raise TypeError("tokenizer output must be a sequence of token IDs") from exc
    if result and isinstance(result[0], list):
        if len(result) != 1:
            raise ValueError("tokenizer returned multiple rows for one document")
        result = result[0]
    return result


def _identity(tokenizer: Any, identifier: str | None, revision: str | None) -> dict[str, Any]:
    name = identifier or getattr(tokenizer, "name_or_path", None)
    actual_revision = revision
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    if actual_revision is None and isinstance(init_kwargs, Mapping):
        actual_revision = init_kwargs.get("_commit_hash") or init_kwargs.get("revision")
    return {"identifier": name or "caller_supplied", "revision": actual_revision}


def _quantile(length_counts: Counter[int], total: int, fraction: float) -> int:
    if not total:
        return 0
    target = max(1, math.ceil(fraction * total))
    seen = 0
    for length, count in sorted(length_counts.items()):
        seen += count
        if seen >= target:
            return length
    raise AssertionError("quantile target exceeded token-length count")


def token_stats(
    corpus: str | Path | Iterable[Mapping[str, Any]],
    *,
    tokenizer: Any | str | None = None,
    tokenizer_identifier: str | None = None,
    revision: str | None = None,
    text_field: str = "text",
) -> dict[str, Any]:
    """Count tokens and report document-length distribution.

    Blank/whitespace-only text is defined to have zero tokens and is not passed to
    the tokenizer. Other tokenizer errors include the 1-based record number.
    Pass a tokenizer object or a Transformers model/tokenizer name.
    """
    if tokenizer is None:
        raise ValueError("Provide a tokenizer object or tokenizer identifier")
    identifier = tokenizer_identifier
    if isinstance(tokenizer, str):
        identifier = identifier or tokenizer
        tokenizer = _load_tokenizer(tokenizer, revision)
    docs: Iterable[Mapping[str, Any]]
    docs = read_jsonl(corpus) if isinstance(corpus, (str, Path)) else corpus
    lengths: Counter[int] = Counter()
    total = empty = 0
    for record_number, row in enumerate(docs, 1):
        if not isinstance(row, Mapping):
            raise TypeError(f"Record {record_number}: each corpus record must be an object")
        text = row.get(text_field)
        if not isinstance(text, str):
            raise TypeError(f"Record {record_number}: {text_field!r} must be a string")
        total += 1
        if not text.strip():
            empty += 1
            lengths[0] += 1
            continue
        try:
            lengths[len(_token_ids(tokenizer, text))] += 1
        except Exception as exc:
            raise RuntimeError(
                f"Tokenizer failed for record {record_number} ({type(exc).__name__})"
            ) from None
    total_tokens = sum(length * count for length, count in lengths.items())
    return {
        "schema_version": 1,
        "input_sha256": sha256_file(corpus) if isinstance(corpus, (str, Path)) else None,
        "configuration": {"text_field": text_field, "add_special_tokens": False},
        "software": _software_environment(),
        "tokenizer": _identity(tokenizer, identifier, revision),
        "documents": total,
        "empty_documents": empty,
        "total_tokens": total_tokens,
        "token_length": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": total_tokens / total if total else 0.0,
            "p50": _quantile(lengths, total, 0.50),
            "p90": _quantile(lengths, total, 0.90),
            "p95": _quantile(lengths, total, 0.95),
            "histogram": {str(size): count for size, count in sorted(lengths.items())},
        },
    }
