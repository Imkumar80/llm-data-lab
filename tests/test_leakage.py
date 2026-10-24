import json

import pytest

from llm_data_lab.leakage import check_split_overlap
from llm_data_lab.pipeline import sha256_file


def write_rows(path, texts):
    path.write_text(
        "".join(json.dumps({"text": text}) + "\n" for text in texts), encoding="utf-8"
    )


def test_overlap_reports_duplicate_and_cross_split_counts_without_text(tmp_path):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    write_rows(train, ["shared secret text", "shared secret text", "train only"])
    write_rows(validation, ["shared secret text", "validation only"])

    report = check_split_overlap({"train": train, "validation": validation})

    assert report["has_overlap"] is True
    assert report["split_sha256"] == {
        "train": sha256_file(train),
        "validation": sha256_file(validation),
    }
    assert report["affected_splits"] == ["train", "validation"]
    assert report["overlapping_documents_by_split"] == {"train": 2, "validation": 1}
    assert report["pairs"] == [
        {
            "splits": ["train", "validation"],
            "unique_overlapping_texts": 1,
            "overlapping_documents": {"train": 2, "validation": 1},
        }
    ]
    assert "shared secret text" not in json.dumps(report)


def test_normalization_policy_is_explicit_and_configurable():
    splits = {
        "train": [{"text": "  Café\nNews "}],
        "validation": [{"text": "cafe\u0301   news"}],
    }
    assert not check_split_overlap(splits, normalization="exact")["has_overlap"]
    normalized = check_split_overlap(splits, normalization="nfc_casefold_whitespace")
    assert normalized["has_overlap"]
    assert normalized["normalization"] == "nfc_casefold_whitespace"


def test_overlap_handles_no_overlap_empty_split_and_invalid_records():
    report = check_split_overlap({"train": [], "validation": [{"text": "valid"}]})
    assert report["has_overlap"] is False
    assert report["affected_splits"] == []
    with pytest.raises(TypeError, match="record 1.*string"):
        check_split_overlap({"train": [{"text": None}], "validation": []})
    with pytest.raises(ValueError, match="At least two"):
        check_split_overlap({"train": []})
    with pytest.raises(ValueError, match="Unknown normalization"):
        check_split_overlap({"a": [], "b": []}, normalization="semantic")
