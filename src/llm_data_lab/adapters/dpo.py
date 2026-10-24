"""Preparation and integrity checks for direct-preference optimization pairs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from llm_data_lab.leakage import _normalize, check_split_overlap
from llm_data_lab.pipeline import (
    _ensure_distinct_paths,
    _software_environment,
    read_jsonl,
    sha256_file,
)
from llm_data_lab.token_stats import _identity, _token_ids

from .cpt import _decode, _length_summary

DPOOverflowPolicy = Literal["error", "drop", "truncate"]
ConflictPolicy = Literal["report", "error"]


def _fingerprint(text: str, normalization: str) -> str:
    return hashlib.sha256(_normalize(text, normalization).encode("utf-8")).hexdigest()


def _truncate_answer(
    tokenizer: Any, prompt: str, answer: str, max_length: int, prompt_ids: list[Any]
) -> tuple[str, int, int]:
    answer_ids = _token_ids(tokenizer, answer)
    candidate = answer_ids[: max(0, max_length - len(prompt_ids))]
    while candidate:
        truncated = _decode(tokenizer, candidate)
        if not isinstance(truncated, str):
            raise TypeError("tokenizer.decode(token_ids) must return a string")
        sequence_length = len(_token_ids(tokenizer, prompt + truncated))
        answer_length = len(_token_ids(tokenizer, truncated))
        if truncated.strip() and sequence_length <= max_length:
            return truncated, answer_length, sequence_length
        candidate = candidate[:-1]
    return "", 0, len(prompt_ids)


def _read_pair_fields(row: Any, record_number: int) -> tuple[str, str, str]:
    if not isinstance(row, Mapping):
        raise TypeError(f"Record {record_number}: expected a JSON object")
    values = []
    for field in ("prompt", "chosen", "rejected"):
        value = row.get(field)
        if not isinstance(value, str):
            raise TypeError(f"Record {record_number}: {field!r} must be a string")
        if not value.strip():
            raise ValueError(f"Record {record_number}: {field!r} must not be empty")
        values.append(value)
    if values[1] == values[2]:
        raise ValueError(f"Record {record_number}: chosen and rejected answers are identical")
    return values[0], values[1], values[2]


def check_dpo_split_overlap(
    splits: Mapping[str, str | Path | Any], *, normalization: str = "exact"
) -> dict[str, Any]:
    """Check whether any exact prompt appears in more than one pair split."""
    report = check_split_overlap(splits, text_field="prompt", normalization=normalization)
    report["integrity_scope"] = "DPO prompt-level pair leakage"
    return report


def prepare_dpo_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    tokenizer: Any,
    tokenizer_identifier: str | None = None,
    tokenizer_revision: str | None = None,
    max_sequence_length: int,
    overflow_policy: DPOOverflowPolicy = "drop",
    conflict_policy: ConflictPolicy = "report",
    prompt_normalization: str = "exact",
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate preference pairs and export token-counted JSONL.

    Length limits apply separately to prompt+chosen and prompt+rejected. When
    truncating, the prompt is preserved and each answer is shortened to fit.
    Contradictory preferences are preserved and counted by default; choose
    ``conflict_policy='error'`` to stop before writing any output.
    """
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    if overflow_policy not in {"error", "drop", "truncate"}:
        raise ValueError("overflow_policy must be 'error', 'drop', or 'truncate'")
    if conflict_policy not in {"report", "error"}:
        raise ValueError("conflict_policy must be 'report' or 'error'")
    if prompt_normalization not in {"exact", "nfc_casefold_whitespace"}:
        raise ValueError("prompt_normalization must be 'exact' or 'nfc_casefold_whitespace'")
    outputs = [output_path] + ([report_path] if report_path is not None else [])
    _ensure_distinct_paths(input_path, *outputs)
    source, destination = Path(input_path), Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    input_checksum = sha256_file(source)

    prompt_counts: Counter[str] = Counter()
    observed_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for record_number, row in enumerate(read_jsonl(source), 1):
        try:
            prompt, chosen, rejected = _read_pair_fields(row, record_number)
        except (TypeError, ValueError):
            continue
        prompt_key = _fingerprint(prompt, prompt_normalization)
        prompt_counts[prompt_key] += 1
        observed_pairs[prompt_key].add(
            (_fingerprint(chosen, "exact"), _fingerprint(rejected, "exact"))
        )
    conflicting_prompts = {
        prompt
        for prompt, pairs in observed_pairs.items()
        if any((rejected, chosen) in pairs for chosen, rejected in pairs)
    }
    if conflicting_prompts and conflict_policy == "error":
        raise ValueError(
            f"Found {len(conflicting_prompts)} prompts with contradictory preferences"
        )

    lengths: dict[str, Counter[int]] = {
        "prompt": Counter(),
        "chosen": Counter(),
        "rejected": Counter(),
    }
    output_lengths: dict[str, Counter[int]] = {"chosen": Counter(), "rejected": Counter()}
    rejected_counts: Counter[str] = Counter()
    input_documents = output_documents = 0
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for record_number, row in enumerate(read_jsonl(source), 1):
                input_documents += 1
                try:
                    prompt, chosen, rejected = _read_pair_fields(row, record_number)
                except TypeError as exc:
                    reason = "missing_or_invalid_field"
                    if "expected a JSON object" in str(exc):
                        reason = "invalid_record"
                    rejected_counts[reason] += 1
                    continue
                except ValueError as exc:
                    reason = (
                        "identical_answers"
                        if "answers are identical" in str(exc)
                        else "empty_field"
                    )
                    rejected_counts[reason] += 1
                    continue
                try:
                    prompt_ids = _token_ids(tokenizer, prompt)
                    chosen_ids = _token_ids(tokenizer, chosen)
                    rejected_ids = _token_ids(tokenizer, rejected)
                    chosen_sequence_ids = _token_ids(tokenizer, prompt + chosen)
                    rejected_sequence_ids = _token_ids(tokenizer, prompt + rejected)
                except Exception as exc:
                    raise RuntimeError(
                        f"Tokenizer failed for record {record_number} ({type(exc).__name__})"
                    ) from None
                lengths["prompt"][len(prompt_ids)] += 1
                lengths["chosen"][len(chosen_ids)] += 1
                lengths["rejected"][len(rejected_ids)] += 1
                if not chosen_ids or not rejected_ids:
                    rejected_counts["empty_tokenized_answer"] += 1
                    continue
                chosen_total = len(chosen_sequence_ids)
                rejected_total = len(rejected_sequence_ids)
                if max(chosen_total, rejected_total) > max_sequence_length:
                    if overflow_policy == "error":
                        raise ValueError(
                            f"Record {record_number} exceeds {max_sequence_length} tokens "
                            "for at least one preference sequence"
                        )
                    if overflow_policy == "drop":
                        rejected_counts["sequence_too_long"] += 1
                        continue
                    if len(prompt_ids) >= max_sequence_length:
                        rejected_counts["prompt_too_long"] += 1
                        continue
                    try:
                        if chosen_total > max_sequence_length:
                            chosen, chosen_count, chosen_total = _truncate_answer(
                                tokenizer, prompt, chosen, max_sequence_length, prompt_ids
                            )
                        else:
                            chosen_count = len(chosen_ids)
                        if rejected_total > max_sequence_length:
                            rejected, rejected_count, rejected_total = _truncate_answer(
                                tokenizer, prompt, rejected, max_sequence_length, prompt_ids
                            )
                        else:
                            rejected_count = len(rejected_ids)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Could not truncate record {record_number} "
                            f"({type(exc).__name__})"
                        ) from None
                    if not chosen.strip() or not rejected.strip():
                        rejected_counts["empty_after_truncation"] += 1
                        continue
                    if chosen == rejected:
                        rejected_counts["identical_after_truncation"] += 1
                        continue
                else:
                    chosen_count, rejected_count = len(chosen_ids), len(rejected_ids)
                result = dict(row)
                result.update(
                    {
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "token_counts": {
                            "prompt": len(prompt_ids),
                            "chosen": chosen_count,
                            "rejected": rejected_count,
                            "prompt_plus_chosen": chosen_total,
                            "prompt_plus_rejected": rejected_total,
                        },
                    }
                )
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_documents += 1
                output_lengths["chosen"][chosen_count] += 1
                output_lengths["rejected"][rejected_count] += 1
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    duplicate_prompt_count = sum(count > 1 for count in prompt_counts.values())
    report = {
        "schema_version": 1,
        "input": {"path": str(source), "sha256": input_checksum, "documents": input_documents},
        "output": {
            "path": str(destination),
            "sha256": sha256_file(destination),
            "documents": output_documents,
            "retained_documents": output_documents,
            "rejected_documents": input_documents - output_documents,
            "rejected_by_reason": dict(rejected_counts),
            "input_token_lengths": {key: _length_summary(value) for key, value in lengths.items()},
            "token_lengths": {
                key: _length_summary(value) for key, value in output_lengths.items()
            },
        },
        "integrity": {
            "duplicate_prompt_groups": duplicate_prompt_count,
            "duplicate_prompt_records": sum(max(count - 1, 0) for count in prompt_counts.values()),
            "conflicting_prompt_count": len(conflicting_prompts),
            "conflict_policy": conflict_policy,
        },
        "tokenizer": _identity(tokenizer, tokenizer_identifier, tokenizer_revision),
        "configuration": {
            "max_sequence_length": max_sequence_length,
            "overflow_policy": overflow_policy,
            "prompt_normalization": prompt_normalization,
            "conflict_policy": conflict_policy,
        },
        "software": _software_environment(),
        "scope_note": "Pair preparation and integrity checks only. No preference trainer is run.",
    }
    if report_path is not None:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return report
