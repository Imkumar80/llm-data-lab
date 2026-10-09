"""Streaming, deterministic JSONL data preparation primitives."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sqlite3
import subprocess
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF\u00AD]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
HTML_ENTITY_RE = re.compile(r"&[a-zA-Z#0-9]+;")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
REPEATED_PUNCT_RE = re.compile(r"([.!?,;:\-_=~*])\1{2,}")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from JSONL, with line numbers in parse errors."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            yield item


def write_jsonl(path: str | Path, rows: Iterator[dict[str, Any]]) -> int:
    """Write JSONL incrementally and return the number of rows written."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def clean_text(text: str, *, strip_html: bool = True, collapse_punctuation: bool = True) -> str:
    """Conservatively normalize text; no language-specific transformations."""
    text = unicodedata.normalize("NFC", text)
    text = ZERO_WIDTH_RE.sub("", text)
    if strip_html:
        text = HTML_TAG_RE.sub(" ", text)
        text = HTML_ENTITY_RE.sub(" ", text)
    if collapse_punctuation:
        text = REPEATED_PUNCT_RE.sub(lambda match: match.group(1) * 2, text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def clean_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    text_field: str = "text",
    strip_html: bool = True,
) -> dict[str, int]:
    read_count = written_count = empty_count = 0

    def rows() -> Iterator[dict[str, Any]]:
        nonlocal read_count, written_count, empty_count
        for row in read_jsonl(input_path):
            read_count += 1
            value = row.get(text_field)
            if not isinstance(value, str):
                raise ValueError(
                    f"Record {read_count}: expected string field {text_field!r}, "
                    f"got {type(value).__name__}"
                )
            value = clean_text(value, strip_html=strip_html)
            if not value:
                empty_count += 1
                continue
            row[text_field] = value
            written_count += 1
            yield row

    write_jsonl(output_path, rows())
    return {"input_documents": read_count, "output_documents": written_count,
            "empty_after_cleaning": empty_count}


def deduplicate_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    text_field: str = "text",
    database_path: str | Path | None = None,
) -> dict[str, int]:
    """Exact deduplication backed by SQLite to avoid a large in-memory hash set."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    db_path = Path(database_path) if database_path else output.with_suffix(".dedupe.sqlite3")
    if db_path.resolve() == output.resolve():
        raise ValueError("SQLite database path must differ from output path")
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    read_count = written_count = duplicate_count = 0
    try:
        connection = sqlite3.connect(db_path)
        try:
            with output.open("w", encoding="utf-8") as out:
                connection.execute("CREATE TABLE seen (digest TEXT PRIMARY KEY)")
                for row in read_jsonl(input_path):
                    read_count += 1
                    value = row.get(text_field)
                    if not isinstance(value, str):
                        raise ValueError(f"Record {read_count}: {text_field!r} must be a string")
                    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
                    cursor = connection.execute(
                        "INSERT OR IGNORE INTO seen(digest) VALUES (?)",
                        (digest,),
                    )
                    if cursor.rowcount == 0:
                        duplicate_count += 1
                        continue
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written_count += 1
                connection.commit()
        finally:
            connection.close()
    finally:
        if database_path is None:
            try:
                db_path.unlink(missing_ok=True)
            except PermissionError:
                pass
    return {"input_documents": read_count, "output_documents": written_count,
            "exact_duplicates_removed": duplicate_count}


def unique_word_ratio(text: str) -> float:
    words = text.casefold().split()
    return len(set(words)) / len(words) if words else 0.0


def filter_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    text_field: str = "text",
    min_chars: int = 1,
    max_chars: int = 1_000_000,
    min_unique_word_ratio: float = 0.0,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    if min_chars < 0 or max_chars < min_chars:
        raise ValueError("Require 0 <= min_chars <= max_chars")
    if not 0.0 <= min_unique_word_ratio <= 1.0:
        raise ValueError("min_unique_word_ratio must be between 0 and 1")

    read_count = written_count = 0
    rejected: Counter[str] = Counter()
    kept_characters = 0
    def rows() -> Iterator[dict[str, Any]]:
        nonlocal read_count, written_count, kept_characters
        for row in read_jsonl(input_path):
            read_count += 1
            text = row.get(text_field)
            if not isinstance(text, str):
                raise ValueError(f"Record {read_count}: {text_field!r} must be a string")
            length = len(text)
            reason = None
            if length < min_chars:
                reason = "too_short"
            elif length > max_chars:
                reason = "too_long"
            elif unique_word_ratio(text) < min_unique_word_ratio:
                reason = "low_unique_word_ratio"
            if reason:
                rejected[reason] += 1
                continue
            written_count += 1
            kept_characters += length
            yield row

    write_jsonl(output_path, rows())
    report: dict[str, Any] = {
        "input_documents": read_count,
        "output_documents": written_count,
        "retention_rate": written_count / read_count if read_count else 0.0,
        "rejected_by_reason": dict(rejected),
        "min_chars": min_chars,
        "max_chars": max_chars,
        "min_unique_word_ratio": min_unique_word_ratio,
    }
    if written_count:
        report["kept_character_count"] = kept_characters
        report["mean_document_chars"] = kept_characters / written_count
    if report_path:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def split_jsonl(
    input_path: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
    *,
    text_field: str = "text",
    val_fraction: float = 0.02,
    seed: int = 42,
) -> dict[str, Any]:
    """Assign rows to splits using stable hashes; deterministic and order-independent."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    train_file, val_file = Path(train_path), Path(validation_path)
    if train_file.resolve() == val_file.resolve():
        raise ValueError("train and validation paths must be different")
    train_file.parent.mkdir(parents=True, exist_ok=True)
    val_file.parent.mkdir(parents=True, exist_ok=True)
    train_count = val_count = 0
    threshold = int(val_fraction * (2**256 - 1))
    with train_file.open("w", encoding="utf-8") as train, val_file.open(
        "w", encoding="utf-8"
    ) as validation:
        for index, row in enumerate(read_jsonl(input_path), 1):
            value = row.get(text_field)
            if not isinstance(value, str):
                raise ValueError(f"Record {index}: {text_field!r} must be a string")
            digest = hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()
            if int(digest, 16) <= threshold:
                validation.write(json.dumps(row, ensure_ascii=False) + "\n")
                val_count += 1
            else:
                train.write(json.dumps(row, ensure_ascii=False) + "\n")
                train_count += 1
    return {"train_documents": train_count, "validation_documents": val_count,
            "validation_fraction_requested": val_fraction, "seed": seed,
            "note": "Hash-based assignment is deterministic; realized fraction varies slightly."}


def corpus_stats(input_path: str | Path, *, text_field: str = "text") -> dict[str, Any]:
    count = total_chars = total_words = empty = 0
    source_counts: Counter[str] = Counter()
    for index, row in enumerate(read_jsonl(input_path), 1):
        text = row.get(text_field)
        if not isinstance(text, str):
            raise ValueError(f"Record {index}: {text_field!r} must be a string")
        count += 1
        total_chars += len(text)
        total_words += len(text.split())
        empty += not bool(text.strip())
        source_counts[str(row.get("source", "unknown"))] += 1
    return {
        "documents": count,
        "characters": total_chars,
        "whitespace_words": total_words,
        "empty_documents": empty,
        "mean_chars_per_document": total_chars / count if count else 0.0,
        "mean_words_per_document": total_words / count if count else 0.0,
        "source_document_counts": dict(source_counts),
    }


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest for the file at *path*."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _load_json_blob(value: Any, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label} configuration file is not valid JSON: {exc}") from exc
            if not isinstance(loaded, dict):
                raise ValueError(f"{label} configuration must decode to an object")
            return loaded
        text = str(value)
        if text.startswith("{"):
            try:
                loaded = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label} configuration is not valid JSON: {exc}") from exc
            if not isinstance(loaded, dict):
                raise ValueError(f"{label} configuration must decode to an object")
            return loaded
        raise FileNotFoundError(f"{label} configuration path does not exist: {path}")
    raise TypeError(f"{label} configuration must be a dictionary or JSON string/path")


def _software_environment() -> dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
    }


def build_dataset_manifest(
    input_path: str | Path,
    *,
    dataset_identifier: str | None = None,
    checksum: str | None = None,
    pipeline_config: dict[str, Any] | str | Path | None = None,
    seed: int | None = None,
    run_name: str | None = None,
    timestamp: str | None = None,
    code_commit: str | None = None,
    model_identifier: str | None = None,
    tokenizer_identifier: str | None = None,
    training_config: dict[str, Any] | str | Path | None = None,
    evaluation_config: dict[str, Any] | str | Path | None = None,
    hardware: dict[str, Any] | None = None,
    software: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON manifest describing the dataset and a single run."""
    input_file = Path(input_path)
    checksum_value = checksum or sha256_file(input_file)
    dataset_id = dataset_identifier or str(input_file)
    manifest = {
        "run_name": run_name or input_file.stem,
        "timestamp": timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_commit": code_commit or _git_commit(),
        "input_dataset": {
            "identifier": dataset_id,
            "path": str(input_file),
            "checksum_sha256": checksum_value,
        },
        "pipeline_configuration": _load_json_blob(pipeline_config, label="pipeline"),
        "seed": seed,
        "model": {
            "identifier": model_identifier or "unknown",
            "revision": None,
        },
        "tokenizer": {
            "identifier": tokenizer_identifier or "unknown",
            "revision": None,
        },
        "training_configuration": _load_json_blob(training_config, label="training"),
        "evaluation": _load_json_blob(evaluation_config, label="evaluation"),
        "hardware": hardware or {},
        "software": software or _software_environment(),
    }
    return manifest


def build_run_manifest(
    input_path: str | Path,
    *,
    dataset_identifier: str | None = None,
    checksum: str | None = None,
    pipeline_config: dict[str, Any] | str | Path | None = None,
    seed: int | None = None,
    run_name: str | None = None,
    timestamp: str | None = None,
    code_commit: str | None = None,
    model_identifier: str | None = None,
    tokenizer_identifier: str | None = None,
    model_revision: str | None = None,
    tokenizer_revision: str | None = None,
    training_config: dict[str, Any] | str | Path | None = None,
    evaluation_config: dict[str, Any] | str | Path | None = None,
    hardware: dict[str, Any] | None = None,
    software: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Backward-compatible alias for the run-manifest builder."""
    manifest = build_dataset_manifest(
        input_path,
        dataset_identifier=dataset_identifier,
        checksum=checksum,
        pipeline_config=pipeline_config,
        seed=seed,
        run_name=run_name,
        timestamp=timestamp,
        code_commit=code_commit,
        model_identifier=model_identifier,
        tokenizer_identifier=tokenizer_identifier,
        training_config=training_config,
        evaluation_config=evaluation_config,
        hardware=hardware,
        software=software,
    )
    manifest["model"]["revision"] = model_revision
    manifest["tokenizer"]["revision"] = tokenizer_revision
    return manifest


def write_run_manifest(
    manifest_path: str | Path,
    input_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a run manifest and write it to JSON."""
    manifest = build_run_manifest(input_path, **kwargs)
    out_path = Path(manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def write_manifest(manifest_path: str | Path, input_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Alias for write_run_manifest."""
    return write_run_manifest(manifest_path, input_path, **kwargs)
