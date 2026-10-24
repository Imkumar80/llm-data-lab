import json

import pytest

from llm_data_lab.adapters.cpt import prepare_cpt_dataset
from llm_data_lab.pipeline import sha256_file


class SpaceTokenizer:
    name_or_path = "fixture/space-tokenizer"
    init_kwargs = {"_commit_hash": "fixture-revision"}

    def encode(self, text, *, add_special_tokens=True):
        return text.split()

    def decode(self, tokens, *, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return " ".join(tokens)


def write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_cpt_prepare_truncates_enforces_budget_and_writes_audit_report(tmp_path):
    source, output, report_path = (
        tmp_path / "raw.jsonl",
        tmp_path / "prepared.jsonl",
        tmp_path / "report.json",
    )
    write_rows(
        source,
        [
            {"text": "one two three four", "source": "fixture"},
            {"text": "   "},
            {"text": "five six"},
            {"text": "seven"},
        ],
    )

    result = prepare_cpt_dataset(
        source,
        output,
        tokenizer=SpaceTokenizer(),
        max_sequence_length=3,
        token_budget=5,
        overflow_policy="truncate",
        report_path=report_path,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"text": "one two three", "source": "fixture"},
        {"text": "five six"},
    ]
    assert result["input"]["documents"] == 4
    assert result["input"]["token_lengths"]["total_tokens"] == 7
    assert result["output"]["documents"] == 2
    assert result["output"]["retained_documents"] == 2
    assert result["output"]["rejected_documents"] == 2
    assert result["output"]["rejected_by_reason"] == {
        "empty_text": 1,
        "token_budget_exceeded": 1,
    }
    assert result["output"]["token_lengths"]["total_tokens"] == 5
    assert result["output"]["sha256"] == sha256_file(output)
    assert result["tokenizer"] == {
        "identifier": "fixture/space-tokenizer",
        "revision": "fixture-revision",
    }
    assert json.loads(report_path.read_text(encoding="utf-8")) == result


def test_cpt_prepare_is_deterministic_and_drops_overlong_documents(tmp_path):
    source = tmp_path / "raw.jsonl"
    write_rows(source, [{"text": "one two three four"}, {"text": "a b"}])
    first = prepare_cpt_dataset(
        source,
        tmp_path / "first.jsonl",
        tokenizer=SpaceTokenizer(),
        max_sequence_length=3,
        overflow_policy="drop",
    )
    second = prepare_cpt_dataset(
        source,
        tmp_path / "second.jsonl",
        tokenizer=SpaceTokenizer(),
        max_sequence_length=3,
        overflow_policy="drop",
    )
    assert (tmp_path / "first.jsonl").read_bytes() == (tmp_path / "second.jsonl").read_bytes()
    assert first["output"]["sha256"] == second["output"]["sha256"]
    assert first["output"]["rejected_by_reason"] == {"sequence_too_long": 1}


def test_cpt_prepare_rejects_invalid_config_and_sequence_overflow(tmp_path):
    source = tmp_path / "raw.jsonl"
    write_rows(source, [{"text": "one two three"}])
    with pytest.raises(ValueError, match="max_sequence_length"):
        prepare_cpt_dataset(source, tmp_path / "out.jsonl", tokenizer=SpaceTokenizer(), max_sequence_length=0)
    with pytest.raises(ValueError, match="token_budget"):
        prepare_cpt_dataset(
            source,
            tmp_path / "out.jsonl",
            tokenizer=SpaceTokenizer(),
            max_sequence_length=4,
            token_budget=0,
        )
    with pytest.raises(ValueError, match="Record 1 has 3 tokens"):
        prepare_cpt_dataset(
            source,
            tmp_path / "out.jsonl",
            tokenizer=SpaceTokenizer(),
            max_sequence_length=2,
            overflow_policy="error",
        )


def test_cpt_prepare_reports_malformed_rows_and_tokenizer_failures(tmp_path):
    source, output = tmp_path / "raw.jsonl", tmp_path / "out.jsonl"
    write_rows(source, [{"text": 4}])
    with pytest.raises(TypeError, match="Record 1.*must be a string"):
        prepare_cpt_dataset(source, output, tokenizer=SpaceTokenizer(), max_sequence_length=4)
    assert not output.exists()

    write_rows(source, [{"text": "one two"}])

    class BrokenTokenizer:
        def encode(self, text, *, add_special_tokens=True):
            raise ValueError("private text should not appear")

    with pytest.raises(RuntimeError, match="record 1.*ValueError") as error:
        prepare_cpt_dataset(source, output, tokenizer=BrokenTokenizer(), max_sequence_length=4)
    assert "private text" not in str(error.value)
    assert not output.exists()


def test_cpt_truncate_requires_a_decoder(tmp_path):
    source = tmp_path / "raw.jsonl"
    write_rows(source, [{"text": "one two three"}])

    class EncoderOnly:
        def encode(self, text, *, add_special_tokens=True):
            return text.split()

    with pytest.raises(RuntimeError, match="Could not truncate.*TypeError"):
        prepare_cpt_dataset(
            source,
            tmp_path / "out.jsonl",
            tokenizer=EncoderOnly(),
            max_sequence_length=2,
            overflow_policy="truncate",
        )
