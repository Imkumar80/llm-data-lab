"""Fetch a small public dataset slice and write a reproducible ablation report."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
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
SAMPLE_ROWS = 400
BLOCK_COUNT = 4
BLOCK_SIZE = SAMPLE_ROWS // BLOCK_COUNT
LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
VIEWER_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HUB_API_URL = "https://huggingface.co/api/datasets/"
PINNED_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
PINNED_SAMPLE_SHA256 = "054d5a8593b706fd750cbe391613467c9195a3dd410dbbeb4aeb3f4cca930764"
DEFAULT_SEED = 42


def _get_json(url: str) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=45) as response:
            payload = json.load(response)
    except (OSError, URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not fetch {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object from {url}")
    return payload


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


def source_index_block_offsets(
    population_size: int,
    *,
    block_size: int = BLOCK_SIZE,
    block_count: int = BLOCK_COUNT,
    seed: int = DEFAULT_SEED,
) -> list[int]:
    """Choose a seeded contiguous block from each equal source-row-index stratum."""
    if block_count <= 0 or block_size <= 0:
        raise ValueError("block_count and block_size must be positive")
    if population_size < block_size * block_count:
        raise ValueError("population_size must be at least block_size * block_count")
    rng = random.Random(seed)
    starts: list[int] = []
    for block in range(block_count):
        lower = (block * population_size) // block_count
        upper = ((block + 1) * population_size) // block_count - block_size
        if upper < lower:
            raise ValueError("Each source-index stratum must fit one complete block")
        starts.append(rng.randint(lower, upper))
    return starts


def _rows_url(offset: int, length: int) -> str:
    query = urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    return f"{VIEWER_ROWS_URL}?{query}"


def _validate_rows_page(
    page: dict[str, Any], *, total: int | None, offset: int, expected_length: int
) -> tuple[int, list[dict[str, Any]]]:
    if not isinstance(page, dict):
        raise TypeError("Dataset Viewer response must be a JSON object")
    if page.get("partial") is True:
        raise RuntimeError("Dataset Viewer returned a partial filter result")
    page_total = page.get("num_rows_total")
    if not isinstance(page_total, int) or isinstance(page_total, bool):
        raise TypeError("Dataset Viewer response is missing a valid num_rows_total")
    if total is not None and page_total != total:
        raise RuntimeError("Training split row count changed during sample acquisition")
    rows = page.get("rows")
    if not isinstance(rows, list):
        raise TypeError("Dataset Viewer response rows must be a list")
    if len(rows) != expected_length:
        raise RuntimeError(
            f"Dataset Viewer returned {len(rows)} rows; expected {expected_length} from offset {offset}"
        )
    for position, item in enumerate(rows):
        if not isinstance(item, dict):
            raise TypeError("Dataset Viewer row entry must be an object")
        if item.get("row_idx") != offset + position:
            raise RuntimeError("Dataset Viewer returned a missing or reordered source row")
    return page_total, rows


def _serialize_rows(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def verify_sample_checksum(sample: bytes, expected_sha256: str) -> str:
    actual = hashlib.sha256(sample).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"Sample SHA-256 mismatch: expected {expected_sha256}, got {actual}. "
            "Review the source change and update the pin explicitly if intended."
        )
    return actual


def fetch_sample(
    destination: Path,
    *,
    seed: int = DEFAULT_SEED,
    expected_sha256: str | None = PINNED_SAMPLE_SHA256,
    get_json: Any | None = None,
    revision_reader: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    get_json = get_json or _get_json
    revision_reader = revision_reader or _dataset_revision
    revision_before = revision_reader()
    population_page = get_json(_rows_url(0, 1))
    population_size, _ = _validate_rows_page(
        population_page, total=None, offset=0, expected_length=1
    )
    block_offsets = source_index_block_offsets(population_size, seed=seed)
    rows: list[dict[str, Any]] = []
    seen_source_rows: set[int] = set()
    for offset in block_offsets:
        page = get_json(_rows_url(offset, BLOCK_SIZE))
        _, selected_rows = _validate_rows_page(
            page, total=population_size, offset=offset, expected_length=BLOCK_SIZE
        )
        for item in selected_rows:
            row_index = item.get("row_idx")
            row = item.get("row")
            if not isinstance(row_index, int) or isinstance(row_index, bool):
                raise TypeError("Dataset Viewer returned a row without an integer row_idx")
            if row_index in seen_source_rows:
                raise RuntimeError(f"Dataset Viewer returned duplicate source row {row_index}")
            if not isinstance(row, dict):
                raise TypeError(f"Dataset Viewer row {row_index} must be an object")
            if not isinstance(row.get("text"), str):
                raise TypeError(f"Dataset Viewer row {row_index} has missing or invalid text")
            source_label = row.get("label")
            if not isinstance(source_label, int) or isinstance(source_label, bool):
                raise TypeError(f"Dataset Viewer row {row_index} has an invalid label")
            if source_label not in LABELS:
                raise ValueError(f"Dataset Viewer row {row_index} has unknown label {source_label}")
            seen_source_rows.add(row_index)
            rows.append(
                {
                    "text": row["text"],
                    "label": source_label,
                    "source": DATASET,
                    "source_row": row_index,
                }
            )

    revision_after = revision_reader()
    if revision_before != revision_after:
        raise RuntimeError(
            f"Dataset revision changed during acquisition: {revision_before} -> {revision_after}"
        )
    rows.sort(key=lambda row: row["source_row"])
    sample_bytes = _serialize_rows(rows)
    sample_hash = hashlib.sha256(sample_bytes).hexdigest()
    if expected_sha256 is not None:
        verify_sample_checksum(sample_bytes, expected_sha256)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(sample_bytes)
    sample_class_counts = Counter(row["label"] for row in rows)
    metadata = {
        "identifier": DATASET,
        "revision": revision_before,
        "config": CONFIG,
        "split": SPLIT,
        "selection": {
            "algorithm": (
                "Divide the training split's source-row index range into four equal strata; "
                "select one seeded contiguous 100-row block uniformly within each stratum."
            ),
            "seed": seed,
            "requested_rows": SAMPLE_ROWS,
            "blocks": BLOCK_COUNT,
            "rows_per_block": BLOCK_SIZE,
            "block_start_offsets": block_offsets,
            "source_index_strata": [
                {
                    "start_inclusive": block * population_size // BLOCK_COUNT,
                    "end_exclusive": (block + 1) * population_size // BLOCK_COUNT,
                    "selected_block_start": block_offsets[block],
                    "selected_block_end_exclusive": block_offsets[block] + BLOCK_SIZE,
                }
                for block in range(BLOCK_COUNT)
            ],
            "selected_source_row_indices": [row["source_row"] for row in rows],
            "sample_class_counts": {
                LABELS[label]: sample_class_counts.get(label, 0) for label in LABELS
            },
        },
        "row_count": len(rows),
        "training_split_rows": population_size,
        "sample_sha256": sample_hash,
        "source_api": VIEWER_ROWS_URL,
        "viewer_revision_limitation": (
            "The Dataset Viewer rows endpoint does not accept an immutable Hub revision. "
            "The script checks the Hub revision before and after fetching and pins the exact "
            "serialized sample checksum to detect changed viewer results."
        ),
        "source_card": f"https://huggingface.co/datasets/{DATASET}",
        "license_note": (
            "Hugging Face lists the license as unknown. The dataset card describes the "
            "upstream AG News corpus as intended for research and other non-commercial "
            "activities; verify upstream terms before reuse."
        ),
        "labels": {str(label): name for label, name in LABELS.items()},
    }
    return metadata, rows


def _length_distribution(lengths: list[int]) -> dict[str, int]:
    if not lengths:
        return {key: 0 for key in ("min", "p25", "median", "p75", "p90", "max")}
    ordered = sorted(lengths)

    def nearest_rank(percentile: float) -> int:
        return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]

    return {
        "min": ordered[0],
        "p25": nearest_rank(0.25),
        "median": nearest_rank(0.50),
        "p75": nearest_rank(0.75),
        "p90": nearest_rank(0.90),
        "max": ordered[-1],
    }


def _corpus_lengths(corpus_path: str | Path) -> list[int]:
    return [len(row["text"]) for row in read_jsonl(corpus_path)]


def render_report(summary: dict[str, Any]) -> str:
    source = summary["input"]["source"]
    conditions = summary["conditions"]
    label_names = source["labels"]
    for result in conditions.values():
        rows = list(read_jsonl(result["corpus_path"]))
        result["label_counts"] = {
            str(label): count
            for label, count in sorted(Counter(str(row.get("label")) for row in rows).items())
        }
        result["length_distribution_chars"] = _length_distribution(
            [len(row["text"]) for row in rows]
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
        "Sample class counts before preparation are "
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
        f"- Sampling method: {source['selection']['algorithm']}",
        (
            f"- Sampling seed: `{source['selection']['seed']}`; requested/actual: "
            f"{source['selection']['requested_rows']}/{source['row_count']} rows"
        ),
        f"- Sample class counts: {source['selection']['sample_class_counts']}",
        f"- Training split rows: {source['training_split_rows']}",
        f"- Selected source row blocks: {source['selection']['source_index_strata']}",
        f"- Input JSONL SHA-256: `{summary['input']['sha256']}`",
        f"- Acquisition API: [{source['source_api']}]({source['source_api']})",
        f"- Revision limitation: {source['viewer_revision_limitation']}",
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
        "## Observed results",
        "",
        "Retention denominator for every condition is the same sampled input corpus.",
        "",
        "| Condition | Input | Output | Retained | Characters | Whitespace words | Mean chars/doc | Length chars min / p25 / median / p75 / p90 / max |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    input_count = summary["input"]["documents"]
    for name, result in conditions.items():
        stats = result["corpus_stats"]
        distribution = result["length_distribution_chars"]
        length_values = " / ".join(
            str(distribution[key]) for key in ("min", "p25", "median", "p75", "p90", "max")
        )
        lines.append(
            f"| `{name}` | {input_count} | {stats['documents']} | "
            f"{result['retention_rate_vs_input']:.1%} | {stats['characters']} | "
            f"{stats['whitespace_words']} | {stats['mean_chars_per_document']:.1f} | "
            f"{length_values} |"
        )
    lines.extend(
        [
            "",
            "Label counts after each condition:",
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
    lines.extend(["", class_note, "", "Source document counts after each condition:", ""])
    source_names = sorted(
        {
            source_name
            for result in conditions.values()
            for source_name in result["corpus_stats"]["source_document_counts"]
        }
    )
    lines.append("| Condition | " + " | ".join(source_names) + " |")
    lines.append("|---|" + "---:|" * len(source_names))
    for name, result in conditions.items():
        counts = result["corpus_stats"]["source_document_counts"]
        lines.append(
            f"| `{name}` | " + " | ".join(str(counts.get(source_name, 0)) for source_name in source_names) + " |"
        )
    lines.extend(["", "Per-condition corpus SHA-256:", ""])
    for name, result in conditions.items():
        lines.append(f"- `{name}`: `{result['corpus_sha256']}`")
    lines.extend(["", "### Stage effects", ""])
    lines.append(
        f"- `raw` validation/copy: {input_count} input → "
        f"{conditions['raw']['corpus_stats']['documents']} output documents."
    )
    for name in ("clean", "clean_dedupe", "full_filter"):
        stages = conditions[name]["stages"]
        clean = stages["clean"]
        pieces = [
            f"cleaned {clean['input_documents']} → {clean['output_documents']} documents",
            f"empty after cleaning: {clean['empty_after_cleaning']}",
        ]
        if "dedupe" in stages:
            dedupe = stages["dedupe"]
            pieces.append(
                f"deduplicated {dedupe['input_documents']} → {dedupe['output_documents']} "
                f"(removed {dedupe['exact_duplicates_removed']} exact duplicates)"
            )
        if "filter" in stages:
            filter_report = stages["filter"]
            rejected = filter_report.get("rejected_by_reason", {})
            pieces.append(
                f"filtered {filter_report['input_documents']} → {filter_report['output_documents']} "
                f"({', '.join(f'{key}: {value}' for key, value in rejected.items()) or 'no rejections'})"
            )
        lines.append(f"- `{name}`: " + "; ".join(pieces) + ".")
    deduped_path = Path(conditions["full_filter"]["corpus_path"]).with_name(
        "deduplicated.jsonl"
    )
    min_chars = summary["filter_config"]["min_chars"]
    short_lengths = [length for length in _corpus_lengths(deduped_path) if length < min_chars]
    short_distribution = _length_distribution(short_lengths)
    lines.extend(
        [
            "",
            (
                "Filter thresholds: "
                f"minimum {summary['filter_config']['min_chars']} characters, "
                f"maximum {summary['filter_config']['max_chars']} characters, and "
                f"minimum unique-word ratio {summary['filter_config']['min_unique_word_ratio']:.2f}."
            ),
            (
                f"The minimum-character rule is the existing general-purpose default ({min_chars}), "
                "kept unchanged for this demonstration; it was not tuned on a test split or model result."
            ),
            (
                f"Pre-filter documents shorter than {min_chars} characters: {len(short_lengths)}; "
                f"length min/median/max: {short_distribution['min']}/"
                f"{short_distribution['median']}/{short_distribution['max']} characters."
            ),
            "",
            "## Interpretation",
            "",
            (
                "This is a source-index-stratified block sample, not a simple random sample or a "
                "representative sample of the full corpus. Contiguous blocks can cluster similar documents, "
                "and class composition is incidental rather than balanced by design; report observed class "
                "counts without generalizing them. Exact deduplication "
                "uses cleaned text, so duplicates with different non-text labels retain only their first row. "
                "The filter is a heuristic; retention does not measure usefulness or model quality. For causal "
                "claims about data quality, train matched models and evaluate them on the same held-out set."
            ),
            "",
            (
                "The machine-readable run, per-condition manifests, and filter report are in "
                "`data/ag_news_ablation/` after running the command above. Raw article text is not included "
                "in this report."
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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.output_dir / "ag_news_train_stratified_400.jsonl"
    metadata, _ = fetch_sample(input_path, seed=args.seed)
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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
