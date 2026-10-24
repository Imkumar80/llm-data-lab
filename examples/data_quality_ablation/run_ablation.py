"""Run a reproducible data-preparation ablation without training a model.

This script compares raw, cleaned, cleaned+deduplicated, and fully filtered
corpora. It reports corpus-level effects only; it does not claim that any
condition improves model quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from llm_data_lab.pipeline import (
    build_run_manifest,
    clean_jsonl,
    corpus_stats,
    deduplicate_jsonl,
    filter_jsonl,
    read_jsonl,
    sha256_file,
    write_jsonl,
)


def _copy_validated(source: Path, destination: Path, text_field: str) -> dict[str, int]:
    """Validate JSONL rows and copy them without changing text."""
    count = 0

    def rows() -> Iterator[dict[str, Any]]:
        nonlocal count
        for index, row in enumerate(read_jsonl(source), 1):
            if not isinstance(row.get(text_field), str):
                raise TypeError(f"Record {index}: {text_field!r} must be a string")
            count += 1
            yield row

    written = write_jsonl(destination, rows())
    return {"input_documents": count, "output_documents": written}


def run_ablation(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    text_field: str = "text",
    min_chars: int = 200,
    max_chars: int = 100_000,
    min_unique_word_ratio: float = 0.30,
    seed: int = 42,
) -> dict[str, Any]:
    source = Path(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    input_documents = sum(1 for _ in read_jsonl(source))

    conditions = {
        "raw": {"description": "Schema validation only; no text transformations."},
        "clean": {"description": "Conservative Unicode/text cleanup."},
        "clean_dedupe": {"description": "Cleaning followed by exact deduplication."},
        "full_filter": {
            "description": "Cleaning, exact deduplication, then configured quality filters."
        },
    }
    summary: dict[str, Any] = {
        "experiment": "data_quality_ablation",
        "input": {
            "path": str(source),
            "sha256": source_hash,
            "documents": input_documents,
        },
        "seed": seed,
        "text_field": text_field,
        "filter_config": {
            "min_chars": min_chars,
            "max_chars": max_chars,
            "min_unique_word_ratio": min_unique_word_ratio,
        },
        "scope_note": (
            "Corpus-level ablation only. No model was trained or evaluated by this script. "
            "Use the same independently held-out validation set and matched training settings "
            "for downstream model comparisons."
        ),
        "conditions": {},
    }

    for name, config in conditions.items():
        condition_dir = out / name
        condition_dir.mkdir(parents=True, exist_ok=True)
        corpus = condition_dir / "corpus.jsonl"
        stages: dict[str, Any] = {}

        if name == "raw":
            stages["copy"] = _copy_validated(source, corpus, text_field)
        else:
            cleaned = condition_dir / "cleaned.jsonl"
            stages["clean"] = clean_jsonl(source, cleaned, text_field=text_field)
            if name == "clean":
                cleaned.replace(corpus)
            else:
                deduped = condition_dir / "deduplicated.jsonl"
                stages["dedupe"] = deduplicate_jsonl(
                    cleaned, deduped, text_field=text_field
                )
                if name == "clean_dedupe":
                    deduped.replace(corpus)
                else:
                    stages["filter"] = filter_jsonl(
                        deduped,
                        corpus,
                        text_field=text_field,
                        min_chars=min_chars,
                        max_chars=max_chars,
                        min_unique_word_ratio=min_unique_word_ratio,
                        report_path=condition_dir / "filter_report.json",
                    )

        stats = corpus_stats(corpus, text_field=text_field)
        manifest = build_run_manifest(
            corpus,
            dataset_identifier=f"{source} [{name}]",
            pipeline_config={
                "condition": name,
                "description": config["description"],
                "text_field": text_field,
                "filter_config": summary["filter_config"] if name == "full_filter" else {},
                "stages": stages,
            },
            seed=seed,
            run_name=f"data-quality-{name}",
            evaluation_config={
                "status": "not_run",
                "note": "This data-preparation example does not train or evaluate a model.",
            },
        )
        manifest_path = condition_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary["conditions"][name] = {
            "description": config["description"],
            "corpus_path": str(corpus),
            "corpus_sha256": sha256_file(corpus),
            "documents_retained": stats["documents"],
            "retention_rate_vs_input": (
                stats["documents"] / input_documents if input_documents else 0.0
            ),
            "corpus_stats": stats,
            "stages": stages,
            "manifest_path": str(manifest_path),
        }

    summary_path = out / "ablation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL corpus.")
    parser.add_argument("--output-dir", required=True, help="Directory for conditions and reports.")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=100_000)
    parser.add_argument("--min-unique-word-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = run_ablation(
        args.input,
        args.output_dir,
        text_field=args.text_field,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_unique_word_ratio=args.min_unique_word_ratio,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
