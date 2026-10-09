import json

import pytest

from llm_data_lab.pipeline import (
    clean_jsonl, clean_text, corpus_stats, deduplicate_jsonl, filter_jsonl, split_jsonl,
)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_clean_text_normalizes_unicode_and_whitespace():
    assert clean_text("  Cafe\u0301\u200b   test....  ") == "Café test.."


def test_clean_jsonl_preserves_metadata_and_drops_empty(tmp_path):
    source, target = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_rows(source, [{"text": "  hello   world ", "source": "demo"}, {"text": "  "}])
    report = clean_jsonl(source, target)
    assert report == {"input_documents": 2, "output_documents": 1, "empty_after_cleaning": 1}
    assert read_rows(target) == [{"text": "hello world", "source": "demo"}]


def test_clean_fails_for_missing_text_field(tmp_path):
    source, target = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_rows(source, [{"content": "hello"}])
    with pytest.raises(ValueError, match="text"):
        clean_jsonl(source, target)


def test_dedupe_removes_exact_text_but_preserves_first_metadata(tmp_path):
    source, target = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_rows(source, [{"text": "same", "source": "first"},
                        {"text": "same", "source": "second"}, {"text": "different"}])
    report = deduplicate_jsonl(source, target)
    assert report["exact_duplicates_removed"] == 1
    assert read_rows(target) == [{"text": "same", "source": "first"}, {"text": "different"}]


def test_filter_reports_rejection_reasons(tmp_path):
    source, target = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_rows(source, [{"text": "x"}, {"text": "hello hello hello"}, {"text": "one two three"}])
    report = filter_jsonl(source, target, min_chars=3, min_unique_word_ratio=0.5)
    assert report["input_documents"] == 3
    assert report["output_documents"] == 1
    assert report["rejected_by_reason"] == {"too_short": 1, "low_unique_word_ratio": 1}


def test_split_is_deterministic_and_disjoint(tmp_path):
    source, train, val = tmp_path / "in.jsonl", tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    write_rows(source, [{"text": f"document {i}"} for i in range(100)])
    report1 = split_jsonl(source, train, val, val_fraction=0.2, seed=9)
    first_train, first_val = train.read_text(), val.read_text()
    report2 = split_jsonl(source, train, val, val_fraction=0.2, seed=9)
    assert train.read_text() == first_train
    assert val.read_text() == first_val
    assert report1 == report2
    assert set(row["text"] for row in read_rows(train)).isdisjoint(
        row["text"] for row in read_rows(val)
    )


def test_stats_counts_sources(tmp_path):
    source = tmp_path / "in.jsonl"
    write_rows(source, [{"text": "one two", "source": "a"}, {"text": "three"}])
    report = corpus_stats(source)
    assert report["documents"] == 2
    assert report["characters"] == 12
    assert report["source_document_counts"] == {"a": 1, "unknown": 1}


def test_invalid_split_fraction_rejected(tmp_path):
    source, train, val = tmp_path / "in.jsonl", tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    write_rows(source, [{"text": "hello"}])
    with pytest.raises(ValueError):
        split_jsonl(source, train, val, val_fraction=1.0)
