import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from llm_data_lab.pipeline import _software_environment, sha256_file

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "data_quality_ablation"
_SPEC = importlib.util.spec_from_file_location(
    "public_ablation_example", EXAMPLE_DIR / "run_public_dataset.py"
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("Could not load the public ablation example")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.path.insert(0, str(EXAMPLE_DIR))
try:
    _SPEC.loader.exec_module(_MODULE)
finally:
    sys.path.remove(str(EXAMPLE_DIR))


def fake_viewer(*, population=1000, mutate=None):
    def get_json(url):
        query = parse_qs(urlsplit(url).query)
        offset = int(query["offset"][0])
        length = int(query["length"][0])
        rows = [
            {
                "row_idx": index,
                "row": {
                    "text": f"class {index // (population // 4)} item {index} news market policy growth",
                    "label": min(index // (population // 4), 3),
                },
                "truncated_cells": [],
            }
            for index in range(offset, min(offset + length, population))
        ]
        page = {"num_rows_total": population, "partial": False, "rows": rows}
        if mutate is not None and length == _MODULE.BLOCK_SIZE:
            return mutate(page)
        return page

    return get_json


def test_seeded_source_index_blocks_are_deterministic_and_spread():
    first = _MODULE.source_index_block_offsets(1000, seed=17)
    assert first == _MODULE.source_index_block_offsets(1000, seed=17)
    assert first != _MODULE.source_index_block_offsets(1000, seed=18)
    assert len(first) == _MODULE.BLOCK_COUNT
    for stratum, offset in enumerate(first):
        low = stratum * 1000 // _MODULE.BLOCK_COUNT
        high = (stratum + 1) * 1000 // _MODULE.BLOCK_COUNT
        assert low <= offset
        assert offset + _MODULE.BLOCK_SIZE <= high


def test_sample_acquisition_records_class_and_source_row_accounting(tmp_path):
    source = tmp_path / "sample.jsonl"
    revision_checks = iter([_MODULE.PINNED_REVISION, _MODULE.PINNED_REVISION])
    metadata, rows = _MODULE.fetch_sample(
        source,
        seed=17,
        expected_sha256=None,
        get_json=fake_viewer(),
        revision_reader=lambda: next(revision_checks),
    )

    assert len(rows) == 400
    assert Counter(row["label"] for row in rows) == {label: 100 for label in range(4)}
    assert len({row["source_row"] for row in rows}) == 400
    assert [row["source_row"] for row in rows] == sorted(row["source_row"] for row in rows)
    assert metadata["selection"]["seed"] == 17
    assert metadata["selection"]["sample_class_counts"] == {
        "World": 100,
        "Sports": 100,
        "Business": 100,
        "Sci/Tech": 100,
    }
    assert metadata["selection"]["selected_source_row_indices"] == [
        row["source_row"] for row in rows
    ]
    assert metadata["sample_sha256"] == _MODULE.hashlib.sha256(source.read_bytes()).hexdigest()


def test_sample_acquisition_fails_when_revision_changes(tmp_path):
    source = tmp_path / "sample.jsonl"
    revisions = iter([_MODULE.PINNED_REVISION, "changed-revision"])
    with pytest.raises(RuntimeError, match="revision changed"):
        _MODULE.fetch_sample(
            source,
            expected_sha256=None,
            get_json=fake_viewer(),
            revision_reader=lambda: next(revisions),
        )
    assert not source.exists()


def test_dataset_revision_must_match_the_pinned_commit(monkeypatch):
    monkeypatch.setattr(_MODULE, "_get_json", lambda _: {"sha": "unexpected-revision"})
    with pytest.raises(RuntimeError, match="moved from pinned revision"):
        _MODULE._dataset_revision()


def test_sample_checksum_mismatch_fails_clearly():
    with pytest.raises(RuntimeError, match="Sample SHA-256 mismatch"):
        _MODULE.verify_sample_checksum(b"actual", "0" * 64)


def test_acquisition_rejects_missing_and_malformed_rows(tmp_path):
    def missing_row(page):
        page["rows"].pop()
        return page

    with pytest.raises(RuntimeError, match="expected 100"):
        _MODULE.fetch_sample(
            tmp_path / "missing.jsonl",
            expected_sha256=None,
            get_json=fake_viewer(mutate=missing_row),
            revision_reader=lambda: _MODULE.PINNED_REVISION,
        )

    def malformed_text(page):
        page["rows"][0]["row"]["text"] = None
        return page

    with pytest.raises(TypeError, match="invalid text"):
        _MODULE.fetch_sample(
            tmp_path / "malformed.jsonl",
            expected_sha256=None,
            get_json=fake_viewer(mutate=malformed_text),
            revision_reader=lambda: _MODULE.PINNED_REVISION,
        )


def _report_metadata():
    return {
        "identifier": "fancyzhx/ag_news",
        "revision": _MODULE.PINNED_REVISION,
        "config": "default",
        "split": "train",
        "selection": {
            "algorithm": "seeded source-index stratified fixture",
            "seed": 12,
            "requested_rows": 9,
            "sample_class_counts": {"World": 3, "Sports": 2, "Business": 2, "Sci/Tech": 2},
            "selected_source_row_indices": list(range(9)),
            "source_index_strata": [],
        },
        "row_count": 9,
        "sample_sha256": "fixture-hash",
        "source_api": _MODULE.VIEWER_ROWS_URL,
        "viewer_revision_limitation": "fixture revision checked before and after",
        "training_split_rows": 1000,
        "source_card": "https://huggingface.co/datasets/fancyzhx/ag_news",
        "license_note": "Fixture terms",
        "labels": {str(label): name for label, name in _MODULE.LABELS.items()},
    }


def test_ablation_counts_and_manifest_provenance(tmp_path):
    source = tmp_path / "input.jsonl"
    long_texts = {
        label: (
            f"class {label} headline "
            "market company research policy growth trading report demand results outlook"
        )
        for label in range(4)
    }
    rows = []
    for label in range(4):
        rows.extend(
            [
                {"text": long_texts[label], "label": label, "source": "ag_news"},
                {"text": f"Brief {label}", "label": label, "source": "ag_news"},
            ]
        )
    rows.insert(1, {"text": long_texts[0], "label": 0, "source": "ag_news"})
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    summary = _MODULE.run_ablation(
        source,
        tmp_path / "conditions",
        min_chars=50,
        min_unique_word_ratio=0.3,
        input_metadata=_report_metadata(),
    )

    assert summary["input"]["documents"] == 9
    clean_stage = summary["conditions"]["clean"]["stages"]["clean"]
    assert clean_stage == {
        "input_documents": 9,
        "output_documents": 9,
        "retained_documents": 9,
        "rejected_documents": 0,
        "rejected_by_reason": {},
        "empty_after_cleaning": 0,
        "input_sha256": sha256_file(source),
        "output_sha256": sha256_file(tmp_path / "conditions" / "clean" / "corpus.jsonl"),
        "configuration": {"text_field": "text", "strip_html": True},
        "software": _software_environment(),
    }
    assert summary["conditions"]["clean_dedupe"]["stages"]["dedupe"][
        "exact_duplicates_removed"
    ] == 1
    filtered = summary["conditions"]["full_filter"]
    assert filtered["documents_retained"] == 4
    assert filtered["stages"]["filter"]["rejected_by_reason"] == {"too_short": 4}
    manifest = json.loads(
        Path(filtered["manifest_path"]).read_text(encoding="utf-8")
    )
    assert manifest["pipeline_configuration"]["source_metadata"]["revision"] == (
        _MODULE.PINNED_REVISION
    )
    assert manifest["pipeline_configuration"]["source_metadata"]["selection"]["seed"] == 12

    report_one = _MODULE.render_report(summary)
    report_two = _MODULE.render_report(summary)
    assert report_one == report_two
    assert "Input | Output | Retained" in report_one
    assert "too_short: 4" in report_one
    assert "World | Sports | Business | Sci/Tech" in report_one
    assert "representative sample" in report_one
    assert "corpus-level" in report_one.lower() or "model quality" in report_one.lower()


def test_empty_input_has_zero_retention_and_a_valid_report(tmp_path):
    source = tmp_path / "empty.jsonl"
    source.write_text("", encoding="utf-8")
    summary = _MODULE.run_ablation(source, tmp_path / "empty-runs")
    assert summary["input"]["documents"] == 0
    for condition in summary["conditions"].values():
        assert condition["documents_retained"] == 0
        assert condition["retention_rate_vs_input"] == 0.0
