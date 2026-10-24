"""Preparation-only validation and export for supervised chat conversations."""

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
from llm_data_lab.token_stats import _identity

from .cpt import _length_summary

SFTOverflowPolicy = Literal["error", "drop", "truncate"]
TRUNCATION_SIDES = ("left", "right")
ALLOWED_ROLES = ("system", "user", "assistant")


def _as_list(value: Any, *, field: str, record_number: int) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or value is None:
        raise TypeError(f"Record {record_number}: tokenizer returned invalid {field}")
    try:
        values = list(value)
    except TypeError as exc:
        raise TypeError(f"Record {record_number}: tokenizer returned invalid {field}") from exc
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError(f"Record {record_number}: tokenizer returned multiple rows")
        values = values[0]
    return values


def _validate_messages(row: Mapping[str, Any], record_number: int) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Record {record_number}: 'messages' must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    assistant_count = 0
    for message_number, message in enumerate(messages, 1):
        if not isinstance(message, Mapping):
            raise TypeError(f"Record {record_number}, message {message_number}: expected an object")
        role = message.get("role")
        content = message.get("content")
        if role not in ALLOWED_ROLES:
            raise ValueError(
                f"Record {record_number}, message {message_number}: role must be one of "
                f"{', '.join(ALLOWED_ROLES)}"
            )
        if not isinstance(content, str):
            raise TypeError(
                f"Record {record_number}, message {message_number}: content must be a string"
            )
        if role == "assistant":
            assistant_count += 1
            if not content.strip():
                raise ValueError(
                    f"Record {record_number}, message {message_number}: assistant response is empty"
                )
        normalized.append(dict(message))
    if not assistant_count:
        raise ValueError(f"Record {record_number}: conversation has no assistant response")
    return normalized


def prepare_sft_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    tokenizer: Any,
    tokenizer_identifier: str | None = None,
    tokenizer_revision: str | None = None,
    max_sequence_length: int,
    overflow_policy: SFTOverflowPolicy = "drop",
    truncation_side: Literal["left", "right"] = "right",
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate chat records and export token IDs with an assistant loss mask.

    The tokenizer must provide a compatible chat template that supports
    ``return_assistant_tokens_mask=True``. The exported ``assistant_mask`` is
    framework-neutral: downstream trainers may turn it into labels as needed.
    """
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    if overflow_policy not in {"error", "drop", "truncate"}:
        raise ValueError("overflow_policy must be 'error', 'drop', or 'truncate'")
    if truncation_side not in TRUNCATION_SIDES:
        raise ValueError(f"truncation_side must be one of {TRUNCATION_SIDES}")
    outputs = [output_path] + ([report_path] if report_path is not None else [])
    _ensure_distinct_paths(input_path, *outputs)
    source, destination = Path(input_path), Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    input_checksum = sha256_file(source)
    lengths: Counter[int] = Counter()
    output_lengths: Counter[int] = Counter()
    rejected: Counter[str] = Counter()
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
                    messages = _validate_messages(row, record_number)
                except (TypeError, ValueError) as exc:
                    error_text = str(exc)
                    if "assistant response is empty" in error_text:
                        reason = "empty_assistant_response"
                    elif "no assistant response" in error_text:
                        reason = "missing_assistant_response"
                    elif "role must be one of" in error_text:
                        reason = "invalid_role"
                    elif "content must be a string" in error_text:
                        reason = "invalid_content"
                    else:
                        reason = "invalid_conversation"
                    rejected[reason] += 1
                    continue
                try:
                    encoded = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=False,
                        return_dict=True,
                        return_assistant_tokens_mask=True,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Chat template failed for record {record_number} "
                        f"({type(exc).__name__}); it must support assistant span masks "
                        "and return_assistant_tokens_mask=True"
                    ) from None
                if not isinstance(encoded, Mapping):
                    raise TypeError(
                        f"Record {record_number}: chat template must return a mapping "
                        "with input_ids and assistant_masks"
                    )
                input_ids = _as_list(encoded.get("input_ids"), field="input_ids", record_number=record_number)
                mask_value = encoded.get("assistant_masks", encoded.get("assistant_mask"))
                if mask_value is None:
                    raise ValueError(
                        f"Record {record_number}: chat template did not return assistant_masks; "
                        "the template must mark assistant generation spans"
                    )
                assistant_mask = _as_list(mask_value, field="assistant_masks", record_number=record_number)
                if len(input_ids) != len(assistant_mask):
                    raise ValueError(
                        f"Record {record_number}: input_ids and assistant mask lengths differ"
                    )
                if any(value not in (0, 1, False, True) for value in assistant_mask):
                    raise ValueError(f"Record {record_number}: assistant mask must contain only 0 or 1")
                assistant_mask = [int(value) for value in assistant_mask]
                lengths[len(input_ids)] += 1
                if len(input_ids) > max_sequence_length:
                    if overflow_policy == "error":
                        raise ValueError(
                            f"Record {record_number} has {len(input_ids)} tokens; "
                            f"limit is {max_sequence_length}"
                        )
                    if overflow_policy == "drop":
                        rejected["sequence_too_long"] += 1
                        continue
                    if truncation_side == "left":
                        input_ids = input_ids[-max_sequence_length:]
                        assistant_mask = assistant_mask[-max_sequence_length:]
                    else:
                        input_ids = input_ids[:max_sequence_length]
                        assistant_mask = assistant_mask[:max_sequence_length]
                if not any(assistant_mask):
                    rejected["no_supervised_tokens"] += 1
                    continue
                result = dict(row)
                result["messages"] = messages
                result["input_ids"] = input_ids
                result["assistant_mask"] = assistant_mask
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_documents += 1
                output_lengths[len(input_ids)] += 1
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    report = {
        "schema_version": 1,
        "input": {"path": str(source), "sha256": input_checksum, "documents": input_documents},
        "output": {
            "path": str(destination),
            "sha256": sha256_file(destination),
            "documents": output_documents,
            "retained_documents": output_documents,
            "rejected_documents": input_documents - output_documents,
            "rejected_by_reason": dict(rejected),
            "input_token_lengths": _length_summary(lengths),
            "token_lengths": _length_summary(output_lengths),
        },
        "tokenizer": _identity(tokenizer, tokenizer_identifier, tokenizer_revision),
        "configuration": {
            "max_sequence_length": max_sequence_length,
            "overflow_policy": overflow_policy,
            "truncation_side": truncation_side,
            "assistant_mask_field": "assistant_mask",
        },
        "software": _software_environment(),
        "scope_note": "Preparation and schema checks only. No trainer is selected or run.",
    }
    if report_path is not None:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return report
