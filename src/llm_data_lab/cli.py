"""Command-line interface for LLM Data Lab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm_data_lab.adapters.cpt import prepare_cpt_dataset
from llm_data_lab.adapters.dpo import check_dpo_split_overlap, prepare_dpo_dataset
from llm_data_lab.adapters.sft import prepare_sft_dataset
from llm_data_lab.leakage import NORMALIZATION_POLICIES, check_split_overlap
from llm_data_lab.pipeline import (
    clean_jsonl,
    corpus_stats,
    deduplicate_jsonl,
    filter_jsonl,
    split_jsonl,
    write_run_manifest,
)
from llm_data_lab.token_stats import _load_tokenizer, token_stats


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
    stats.add_argument(
        "--tokenizer",
        help="Optional Transformers tokenizer name/path for token counts (requires [transformers]).",
    )
    stats.add_argument("--tokenizer-revision", help="Optional tokenizer revision.")

    overlap = commands.add_parser(
        "overlap", help="Check exact text overlap across two or more named JSONL splits."
    )
    overlap.add_argument(
        "--split",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Named split input; repeat at least twice (example: train=data/train.jsonl).",
    )
    overlap.add_argument("--text-field", default="text")
    overlap.add_argument(
        "--normalization",
        choices=NORMALIZATION_POLICIES,
        default="exact",
        help="Text comparison policy; normalized comparison uses NFC, casefolding, and whitespace collapse.",
    )
    overlap.add_argument("--output", default=None)

    cpt = commands.add_parser(
        "cpt-prepare", help="Validate and export raw-text data under CPT token limits."
    )
    cpt.add_argument("--input", required=True)
    cpt.add_argument("--output", required=True)
    cpt.add_argument("--tokenizer", required=True, help="Transformers tokenizer name or path.")
    cpt.add_argument("--tokenizer-revision", default=None)
    cpt.add_argument("--text-field", default="text")
    cpt.add_argument("--max-sequence-length", type=int, required=True)
    cpt.add_argument("--token-budget", type=int, default=None)
    cpt.add_argument(
        "--overflow-policy",
        choices=["error", "drop", "truncate"],
        default="drop",
        help="How to handle documents longer than the sequence limit.",
    )
    cpt.add_argument("--report", default=None, help="Optional JSON preparation report path.")

    sft = commands.add_parser(
        "sft-prepare", help="Validate chat data and export token IDs with assistant masks."
    )
    sft.add_argument("--input", required=True)
    sft.add_argument("--output", required=True)
    sft.add_argument("--tokenizer", required=True, help="Transformers tokenizer name or path.")
    sft.add_argument("--tokenizer-revision", default=None)
    sft.add_argument("--max-sequence-length", type=int, required=True)
    sft.add_argument("--overflow-policy", choices=["error", "drop", "truncate"], default="drop")
    sft.add_argument("--truncation-side", choices=["left", "right"], default="right")
    sft.add_argument("--report", default=None)

    dpo = commands.add_parser(
        "dpo-prepare", help="Validate and export token-counted preference pairs."
    )
    dpo.add_argument("--input", required=True)
    dpo.add_argument("--output", required=True)
    dpo.add_argument("--tokenizer", required=True, help="Transformers tokenizer name or path.")
    dpo.add_argument("--tokenizer-revision", default=None)
    dpo.add_argument("--max-sequence-length", type=int, required=True)
    dpo.add_argument("--overflow-policy", choices=["error", "drop", "truncate"], default="drop")
    dpo.add_argument("--conflict-policy", choices=["report", "error"], default="report")
    dpo.add_argument(
        "--prompt-normalization",
        choices=NORMALIZATION_POLICIES,
        default="exact",
    )
    dpo.add_argument("--report", default=None)

    dpo_overlap = commands.add_parser(
        "dpo-overlap", help="Check prompt overlap across preference-pair splits."
    )
    dpo_overlap.add_argument("--split", action="append", required=True, metavar="NAME=PATH")
    dpo_overlap.add_argument(
        "--normalization", choices=NORMALIZATION_POLICIES, default="exact"
    )
    dpo_overlap.add_argument("--output", default=None)

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
        report = corpus_stats(args.input, text_field=args.text_field)
        if args.tokenizer:
            report["token_stats"] = token_stats(
                args.input,
                tokenizer=args.tokenizer,
                revision=args.tokenizer_revision,
                text_field=args.text_field,
            )
        _print_report(report, args.output)
    elif args.command == "overlap":
        splits: dict[str, str] = {}
        for spec in args.split:
            name, separator, path = spec.partition("=")
            if not separator or not name or not path:
                raise SystemExit(f"Invalid --split {spec!r}; expected NAME=PATH")
            if name in splits:
                raise SystemExit(f"Duplicate split name {name!r}")
            splits[name] = path
        _print_report(
            check_split_overlap(
                splits,
                text_field=args.text_field,
                normalization=args.normalization,
            ),
            args.output,
        )
    elif args.command == "cpt-prepare":
        tokenizer = _load_tokenizer(args.tokenizer, args.tokenizer_revision)
        report = prepare_cpt_dataset(
            args.input,
            args.output,
            tokenizer=tokenizer,
            tokenizer_identifier=args.tokenizer,
            tokenizer_revision=args.tokenizer_revision,
            max_sequence_length=args.max_sequence_length,
            token_budget=args.token_budget,
            overflow_policy=args.overflow_policy,
            text_field=args.text_field,
            report_path=args.report,
        )
        _print_report(report)
    elif args.command == "sft-prepare":
        tokenizer = _load_tokenizer(args.tokenizer, args.tokenizer_revision)
        report = prepare_sft_dataset(
            args.input,
            args.output,
            tokenizer=tokenizer,
            tokenizer_identifier=args.tokenizer,
            tokenizer_revision=args.tokenizer_revision,
            max_sequence_length=args.max_sequence_length,
            overflow_policy=args.overflow_policy,
            truncation_side=args.truncation_side,
            report_path=args.report,
        )
        _print_report(report)
    elif args.command == "dpo-prepare":
        tokenizer = _load_tokenizer(args.tokenizer, args.tokenizer_revision)
        report = prepare_dpo_dataset(
            args.input,
            args.output,
            tokenizer=tokenizer,
            tokenizer_identifier=args.tokenizer,
            tokenizer_revision=args.tokenizer_revision,
            max_sequence_length=args.max_sequence_length,
            overflow_policy=args.overflow_policy,
            conflict_policy=args.conflict_policy,
            prompt_normalization=args.prompt_normalization,
            report_path=args.report,
        )
        _print_report(report)
    elif args.command == "dpo-overlap":
        splits: dict[str, str] = {}
        for spec in args.split:
            name, separator, path = spec.partition("=")
            if not separator or not name or not path:
                raise SystemExit(f"Invalid --split {spec!r}; expected NAME=PATH")
            if name in splits:
                raise SystemExit(f"Duplicate split name {name!r}")
            splits[name] = path
        _print_report(check_dpo_split_overlap(splits, normalization=args.normalization), args.output)
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
