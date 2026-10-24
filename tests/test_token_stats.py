import json

import pytest

from llm_data_lab.pipeline import _software_environment, sha256_file
from llm_data_lab.token_stats import token_stats


class WordTokenizer:
    name_or_path = "fixture/word-tokenizer"
    init_kwargs = {"_commit_hash": "abc123"}

    def encode(self, text, *, add_special_tokens=True):
        return text.split()


def test_token_stats_counts_tokens_and_empty_text_deterministically(tmp_path):
    corpus = tmp_path / "tiny.jsonl"
    corpus.write_text(
        ''.join(json.dumps({"text": value}) + "\n" for value in ["one two", " ", "three"]),
        encoding="utf-8",
    )

    report = token_stats(corpus, tokenizer=WordTokenizer())

    assert report["input_sha256"] == sha256_file(corpus)
    assert report["tokenizer"] == {"identifier": "fixture/word-tokenizer", "revision": "abc123"}
    assert report["documents"] == 3
    assert report["empty_documents"] == 1
    assert report["total_tokens"] == 3
    assert report["token_length"]["histogram"] == {"0": 1, "1": 1, "2": 1}
    assert report == token_stats(corpus, tokenizer=WordTokenizer())


def test_token_stats_handles_empty_corpus():
    assert token_stats([], tokenizer=WordTokenizer()) == {
        "schema_version": 1,
        "input_sha256": None,
        "configuration": {"text_field": "text", "add_special_tokens": False},
        "software": _software_environment(),
        "tokenizer": {"identifier": "fixture/word-tokenizer", "revision": "abc123"},
        "documents": 0,
        "empty_documents": 0,
        "total_tokens": 0,
        "token_length": {
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "histogram": {},
        },
    }


def test_token_stats_reports_malformed_records_and_tokenizer_failures():
    with pytest.raises(TypeError, match="Record 1.*string"):
        token_stats([{"text": 3}], tokenizer=WordTokenizer())

    class BrokenTokenizer:
        def encode(self, text, *, add_special_tokens=True):
            raise ValueError("fixture failure")

    with pytest.raises(RuntimeError, match="record 1.*ValueError"):
        token_stats([{"text": "hello"}], tokenizer=BrokenTokenizer())


def test_token_stats_requires_tokenizer_and_optional_dependency(monkeypatch):
    with pytest.raises(ValueError, match="Provide a tokenizer"):
        token_stats([], tokenizer=None)

    def unavailable(_name):
        raise ImportError("not installed")

    monkeypatch.setattr("llm_data_lab.token_stats.importlib.import_module", unavailable)
    with pytest.raises(ImportError, match="pip install.*transformers"):
        token_stats([], tokenizer="fixture/tokenizer")
