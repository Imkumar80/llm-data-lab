"""Fetch a small public dataset slice and write a reproducible ablation report."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from run_ablation import run_ablation

from llm_data_lab.pipeline import read_jsonl

DATASET = "fancyzhx/ag_news"
CONFIG = "default"
SPLIT = "train"
SAMPLE_ROWS = 300
PAGE_SIZE = 100
VIEWER_URL = "https://datasets-server.huggingface.co/rows"
HUB_API_URL = "https://huggingface.co/api/datasets/"
PINNED_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
PINNED_SAMPLE_SHA256 = "073b6e0b2201d3a35fbbe5c77ce38511f93c379ccf8d685c9c17853a992c8e46"


def _get_json(url: str) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=45) as response:
            return json.load(response)
    except (OSError, URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not fetch {url}: {exc}") from exc


def _dataset_revision() -> str:
    info = _get_json(HUB_API_URL + DATASET)
    revision = info.get("sha")
    if not isinstance(revision, str) or not revision:
        raise RuntimeError(f"Hugging Face did not return a revision for {DATASET}")
    if revision != PINNED_REVISION:
        raise RuntimeError(
            f"{DATASET} moved from pinned revision {PINNED_REVISION} to {revision}; "
            "review the dataset change before updating this experiment."
        )
    return revision


def fetch_sample(destination: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    revision_before = _dataset_revision()
    rows: list[dict[str, Any]] = []
    for offset in range(0, SAMPLE_ROWS, PAGE_SIZE):
        query = urlencode(
            {
                "dataset": DATASET,
                "config": CONFIG,
                "split": SPLIT,
                "offset": offset,
                "length": min(PAGE_SIZE, SAMPLE_ROWS - offset),
            }
        )
        page = _get_json(f"{VIEWER_URL}?{query}")
        if page.get("num_rows_total", 0) < SAMPLE_ROWS:
            raise RuntimeError(f"{DATASET} has fewer than {SAMPLE_ROWS} rows in {SPLIT}")
        for item in page.get("rows", []):
            row_index = item.get("row_idx")
            row = item.get("row", {})
            if row_index != len(rows) or not isinstance(row.get("text"), str):
                raise RuntimeError("Dataset Viewer returned a missing, reordered, or invalid row")
            rows.append(
                {
                    "text": row["text"],
                    "label": row.get("label"),
                    "source": DATASET,
                    "source_row": row_index,
                }
            )
    revision_after = _dataset_revision()
    if revision_before != revision_after:
        raise RuntimeError("Dataset revision changed during download; run again for a consistent slice")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    from llm_data_lab.pipeline import sha256_file

    sample_hash = sha256_file(destination)
    if sample_hash != PINNED_SAMPLE_SHA256:
        raise RuntimeError(
            f"Fetched sample SHA-256 {sample_hash} differs from the pinned sample "
            f"{PINNED_SAMPLE_SHA256}; inspect before updating the experiment."
        )
    metadata = {
        "identifier": DATASET,
        "revision": revision_before,
        "config": CONFIG,
        "split": SPLIT,
        "selection": f"first {SAMPLE_ROWS} rows (row indices 0–{SAMPLE_ROWS - 1})",
        "row_count": len(rows),
        "source_api": VIEWER_URL,
        "source_card": f"https://huggingface.co/datasets/{DATASET}",
        "license_note": (
            "Hugging Face lists the license as unknown. The dataset card describes the "
            "upstream AG News corpus as intended for research and other non-commercial "
            "activities; verify upstream terms before reuse."
        ),
        "labels": {"0": "World", "1": "Sports", "2": "Business", "3": "Sci/Tech"},
    }
    return metadata, rows


def render_report(summary: dict[str, Any]) -> str:
    source = summary["input"]["source"]
    conditions = summary["conditions"]
    label_names = source["labels"]
    for result in conditions.values():
        result["label_counts"] = dict(
            Counter(
                str(row.get("label"))
                for row in read_jsonl(result["corpus_path"])
            )
        )
    raw_label_counts = conditions["raw"]["label_counts"]
    filtered_label_counts = conditions["full_filter"]["label_counts"]
    missing_classes = [
        name for label, name in label_names.items() if not raw_label_counts.get(label, 0)
    ]
    missing_note = (
        f"Absent source classes: {', '.join(missing_classes)}. "
        if missing_classes
        else "All source classes appear in this prefix. "
    )
    class_note = (
        "Source-order class counts are "
        + ", ".join(
            f"{label_names[label]}: {raw_label_counts.get(label, 0)}"
            for label in label_names
        )
        + ". "
        + missing_note
        + "After full filtering, counts are "
        + ", ".join(
            f"{label_names[label]}: {filtered_label_counts.get(label, 0)}"
            for label in label_names
        )
        + ". Consider both total retention and changes in class mix."
    )
    lines = [
        "# AG News data-quality ablation",
        "",
        (
            "This report compares four data-preparation conditions on a fixed public dataset slice. "
            "It reports corpus statistics only; no language model was trained or evaluated."
        ),
        "",
        "## Dataset and reproducibility",
        "",
        f"- Dataset: [{source['identifier']}]({source['source_card']})",
        f"- Hub revision: `{source['revision']}`",
        f"- Config/split: `{source['config']}` / `{source['split']}`",
        f"- Sample: {source['selection']} ({summary['input']['documents']} rows)",
        f"- Input JSONL SHA-256: `{summary['input']['sha256']}`",
        f"- Acquisition API: [{source['source_api']}]({source['source_api']})",
        f"- License note: {source['license_note']}",
        (
            "- The downloaded sample and condition corpora are written under the ignored `data/` directory; "
            "the committed report contains no article text."
        ),
        "",
        "Recreate the sample and report from the repository root with:",
        "",
        "```powershell",
        "python examples/data_quality_ablation/run_public_dataset.py",
        "```",
        "",
        "## Results",
        "",
        "| Condition | Documents | Retained | Characters | Mean chars/doc | Mean words/doc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in conditions.items():
        stats = result["corpus_stats"]
        lines.append(
            f"| `{name}` | {stats['documents']} | {result['retention_rate_vs_input']:.1%} | "
            f"{stats['characters']} | {stats['mean_chars_per_document']:.1f} | "
            f"{stats['mean_words_per_document']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Label counts (the source sample’s original class labels):",
            "",
            "| Condition | " + " | ".join(label_names.values()) + " |",
            "|---|" + "---:|" * len(label_names),
        ]
    )
    for name, result in conditions.items():
        counts = result["label_counts"]
        lines.append(
            f"| `{name}` | "
            + " | ".join(str(counts.get(label, 0)) for label in label_names)
            + " |"
        )
    lines.extend(["", class_note, "", "### Stage effects", ""])
    for name in ("clean", "clean_dedupe", "full_filter"):
        stages = conditions[name]["stages"]
        pieces = [f"cleaned {stages['clean']['output_documents']} documents"]
        if "dedupe" in stages:
            pieces.append(f"removed {stages['dedupe']['exact_duplicates_removed']} exact duplicates")
        if "filter" in stages:
            filter_report = stages["filter"]
            rejected = filter_report.get("rejected_by_reason", {})
            pieces.append(
                f"filtered {filter_report['input_documents'] - filter_report['output_documents']} "
                f"documents ({', '.join(f'{key}: {value}' for key, value in rejected.items()) or 'none'})"
            )
        lines.append(f"- `{name}`: " + "; ".join(pieces) + ".")
    lines.extend(
        [
            "",
            (
                "Filter thresholds: "
                f"minimum {summary['filter_config']['min_chars']} characters, "
                f"maximum {summary['filter_config']['max_chars']} characters, and "
                f"minimum unique-word ratio {summary['filter_config']['min_unique_word_ratio']:.2f}."
            ),
            "",
            "## Interpretation",
            "",
            (
                "These results describe only how this particular 300-row prefix responds to the configured "
                "preprocessing steps. The prefix is not a random sample of the full corpus. Exact deduplication "
                "uses cleaned text, so duplicates with different non-text labels retain only their first row. "
                "The filter is a heuristic; retention does not measure usefulness or model quality. For causal "
                "claims about data quality, train matched models and evaluate them on the same held-out set."
            ),
            "",
            (
                "The machine-readable run, per-condition manifests, and filter report are in "
                "`data/ag_news_ablation/` after running the command above."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/ag_news_ablation"))
    parser.add_argument(
        "--report", type=Path, default=Path("docs/ag_news_ablation_report.md")
    )
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=100_000)
    parser.add_argument("--min-unique-word-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.output_dir / "ag_news_train_first_300.jsonl"
    metadata, _ = fetch_sample(input_path)
    summary = run_ablation(
        input_path,
        args.output_dir / "conditions",
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_unique_word_ratio=args.min_unique_word_ratio,
        seed=args.seed,
        input_metadata=metadata,
    )
    report = render_report(summary)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    (args.output_dir / "conditions" / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
