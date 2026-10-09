"""Command-line interface for LLM Data Lab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm_data_lab.pipeline import (
    clean_jsonl, corpus_stats, deduplicate_jsonl, filter_jsonl, split_jsonl,
    write_run_manifest,
)


def _print_report(report: dict[str, Any], output: str | None = None) -> None:
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-data-lab",
        description="Reproducible JSONL data preparation utilities for LLM research.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    for name, help_text in [
        ("clean", "Conservatively normalize text and remove empty records."),
        ("dedupe", "Remove exact duplicate text records."),
        ("filter", "Filter documents by length and repetition ratio."),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--input", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--text-field", default="text")

    commands.choices["clean"].add_argument("--keep-html", action="store_true")
    commands.choices["dedupe"].add_argument("--database", default=None,
                                             help="Optional temporary SQLite path.")
    filt = commands.choices["filter"]
    filt.add_argument("--min-chars", type=int, default=1)
    filt.add_argument("--max-chars", type=int, default=1_000_000)
    filt.add_argument("--min-unique-word-ratio", type=float, default=0.0)
    filt.add_argument("--report", default=None, help="Optional JSON report path.")

    split = commands.add_parser("split", help="Create deterministic train/validation splits.")
    split.add_argument("--input", required=True)
    split.add_argument("--train", required=True)
    split.add_argument("--validation", required=True)
    split.add_argument("--text-field", default="text")
    split.add_argument("--val-fraction", type=float, default=0.02)
    split.add_argument("--seed", type=int, default=42)

    stats = commands.add_parser("stats", help="Summarize corpus size and source counts.")
    stats.add_argument("--input", required=True)
    stats.add_argument("--text-field", default="text")
    stats.add_argument("--output", default=None)

    manifest = commands.add_parser(
        "manifest",
        aliases=["run-manifest", "experiment-manifest"],
        help="Write a JSON run manifest with dataset, pipeline, and environment metadata.",
    )
    manifest.add_argument("--input", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--run-name", default=None)
    manifest.add_argument("--dataset-id", default=None)
    manifest.add_argument("--seed", type=int, default=42)
    manifest.add_argument("--pipeline-config", default=None,
                         help="JSON object or path to a JSON file describing the pipeline.")
    manifest.add_argument("--model", default=None)
    manifest.add_argument("--tokenizer", default=None)
    manifest.add_argument("--model-revision", default=None)
    manifest.add_argument("--tokenizer-revision", default=None)
    manifest.add_argument("--training-config", default=None,
                         help="JSON object or path to a JSON file for training settings.")
    manifest.add_argument("--evaluation-config", default=None,
                         help="JSON object or path to a JSON file for evaluation settings.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "clean":
        report = clean_jsonl(args.input, args.output, text_field=args.text_field,
                             strip_html=not args.keep_html)
        _print_report(report)
    elif args.command == "dedupe":
        report = deduplicate_jsonl(args.input, args.output, text_field=args.text_field,
                                   database_path=args.database)
        _print_report(report)
    elif args.command == "filter":
        report = filter_jsonl(
            args.input, args.output, text_field=args.text_field,
            min_chars=args.min_chars, max_chars=args.max_chars,
            min_unique_word_ratio=args.min_unique_word_ratio, report_path=args.report,
        )
        _print_report(report)
    elif args.command == "split":
        report = split_jsonl(
            args.input, args.train, args.validation, text_field=args.text_field,
            val_fraction=args.val_fraction, seed=args.seed,
        )
        _print_report(report)
    elif args.command == "stats":
        _print_report(corpus_stats(args.input, text_field=args.text_field), args.output)
    elif args.command in {"manifest", "run-manifest", "experiment-manifest"}:
        report = write_run_manifest(
            args.output,
            args.input,
            run_name=args.run_name,
            dataset_identifier=args.dataset_id,
            pipeline_config=args.pipeline_config,
            seed=args.seed,
            model_identifier=args.model,
            tokenizer_identifier=args.tokenizer,
            model_revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            training_config=args.training_config,
            evaluation_config=args.evaluation_config,
        )
        _print_report(report, None)


if __name__ == "__main__":
    main()
