"""Preparation helpers for raw-text causal pretraining corpora.

This module validates and exports token-budgeted JSONL records. It does not
construct training examples, pack sequences, or run a model trainer.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from llm_data_lab.pipeline import (
    _ensure_distinct_paths,
    _software_environment,
    read_jsonl,
    sha256_file,
)
from llm_data_lab.token_stats import _identity, _token_ids

OverflowPolicy = Literal["error", "drop", "truncate"]


def _decode(tokenizer: Any, token_ids: list[Any]) -> str:
    if not hasattr(tokenizer, "decode"):
        raise TypeError("overflow_policy='truncate' requires tokenizer.decode(token_ids)")
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _truncate(tokenizer: Any, text: str, ids: list[Any], max_length: int) -> tuple[str, int]:
    candidate = ids[:max_length]
    while candidate:
        text = _decode(tokenizer, candidate)
        if not isinstance(text, str):
            raise TypeError("tokenizer.decode(token_ids) must return a string")
        result_ids = _token_ids(tokenizer, text)
        if len(result_ids) <= max_length:
            return text, len(result_ids)
        candidate = candidate[:-1]
    return "", 0


def _length_summary(lengths: Counter[int]) -> dict[str, Any]:
    total = sum(lengths.values())
    tokens = sum(length * count for length, count in lengths.items())
    ordered = sorted(lengths.items())

    def percentile(fraction: float) -> int:
        if not total:
            return 0
        target = max(1, int(fraction * total + 0.999999))
        seen = 0
        for length, count in ordered:
            seen += count
            if seen >= target:
                return length
        return 0

    return {
        "documents": total,
        "total_tokens": tokens,
        "min": ordered[0][0] if ordered else 0,
        "max": ordered[-1][0] if ordered else 0,
        "mean": tokens / total if total else 0.0,
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "histogram": {str(length): count for length, count in ordered},
    }


def prepare_cpt_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    tokenizer: Any,
    tokenizer_identifier: str | None = None,
    tokenizer_revision: str | None = None,
    max_sequence_length: int,
    token_budget: int | None = None,
    overflow_policy: OverflowPolicy = "drop",
    text_field: str = "text",
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and export raw-text JSONL under document and token limits.

    Documents over ``max_sequence_length`` are rejected, truncated through the
    tokenizer's ``decode`` method, or raise an error according to
    ``overflow_policy``. A document that does not fit the remaining token budget
    is rejected as a whole. All source records are inspected so rejection counts
    and input token statistics describe the complete input.
    """
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    if token_budget is not None and token_budget <= 0:
        raise ValueError("token_budget must be positive when provided")
    if overflow_policy not in {"error", "drop", "truncate"}:
        raise ValueError("overflow_policy must be 'error', 'drop', or 'truncate'")
    outputs = [output_path]
    if report_path is not None:
        outputs.append(report_path)
    _ensure_distinct_paths(input_path, *outputs)

    source = Path(input_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    input_checksum = sha256_file(source)
    input_lengths: Counter[int] = Counter()
    output_lengths: Counter[int] = Counter()
    rejected: Counter[str] = Counter()
    input_documents = output_documents = output_tokens = 0
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for record_number, row in enumerate(read_jsonl(source), 1):
                input_documents += 1
                if not isinstance(row, Mapping):
                    raise TypeError(f"Record {record_number}: expected a JSON object")
                text = row.get(text_field)
                if not isinstance(text, str):
                    raise TypeError(f"Record {record_number}: {text_field!r} must be a string")
                if not text.strip():
                    rejected["empty_text"] += 1
                    continue
                try:
                    token_ids = _token_ids(tokenizer, text)
                except Exception as exc:
                    raise RuntimeError(
                        f"Tokenizer failed for record {record_number} ({type(exc).__name__})"
                    ) from None
                input_lengths[len(token_ids)] += 1
                if len(token_ids) > max_sequence_length:
                    if overflow_policy == "error":
                        raise ValueError(
                            f"Record {record_number} has {len(token_ids)} tokens; "
                            f"limit is {max_sequence_length}"
                        )
                    if overflow_policy == "drop":
                        rejected["sequence_too_long"] += 1
                        continue
                    try:
                        text, token_count = _truncate(
                            tokenizer, text, token_ids, max_sequence_length
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"Could not truncate record {record_number} "
                            f"({type(exc).__name__})"
                        ) from None
                    if not text.strip():
                        rejected["empty_after_truncation"] += 1
                        continue
                    row = dict(row)
                    row[text_field] = text
                else:
                    token_count = len(token_ids)
                if token_budget is not None and output_tokens + token_count > token_budget:
                    rejected["token_budget_exceeded"] += 1
                    continue
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_documents += 1
                output_tokens += token_count
                output_lengths[token_count] += 1
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    report = {
        "schema_version": 1,
        "input": {
            "path": str(source),
            "sha256": input_checksum,
            "documents": input_documents,
            "token_lengths": _length_summary(input_lengths),
        },
        "output": {
            "path": str(destination),
            "sha256": sha256_file(destination),
            "documents": output_documents,
            "retained_documents": output_documents,
            "rejected_documents": input_documents - output_documents,
            "rejected_by_reason": dict(rejected),
            "token_lengths": _length_summary(output_lengths),
        },
        "tokenizer": _identity(tokenizer, tokenizer_identifier, tokenizer_revision),
        "configuration": {
            "text_field": text_field,
            "max_sequence_length": max_sequence_length,
            "token_budget": token_budget,
            "overflow_policy": overflow_policy,
            "add_special_tokens": False,
        },
        "software": _software_environment(),
        "scope_note": (
            "Preparation and validation only. This adapter does not pack sequences "
            "or train/evaluate a model."
        ),
    }
    if report_path is not None:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return report
