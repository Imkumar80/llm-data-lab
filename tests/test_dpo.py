import json

import pytest

from llm_data_lab.adapters.dpo import check_dpo_split_overlap, prepare_dpo_dataset


class SpaceTokenizer:
    name_or_path = "fixture/dpo-tokenizer"
    init_kwargs = {"_commit_hash": "dpo-revision"}

    def encode(self, text, *, add_special_tokens=True):
        return text.split()

    def decode(self, tokens, *, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return " ".join(tokens)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_dpo_reports_duplicate_prompts_conflicts_and_identical_answers(tmp_path):
    source, output = tmp_path / "raw.jsonl", tmp_path / "prepared.jsonl"
    write_rows(
        source,
        [
            {"prompt": "choose a color", "chosen": "red is good", "rejected": "blue is bad"},
            {"prompt": "choose a color", "chosen": "blue is bad", "rejected": "red is good"},
            {"prompt": "same answer", "chosen": "same", "rejected": "same"},
            {"prompt": "empty answer", "chosen": "", "rejected": "not empty"},
        ],
    )
    report = prepare_dpo_dataset(
        source,
        output,
        tokenizer=SpaceTokenizer(),
        max_sequence_length=10,
        conflict_policy="report",
    )
    assert report["output"]["documents"] == 2
    assert report["output"]["rejected_by_reason"] == {
        "identical_answers": 1,
        "empty_field": 1,
    }
    assert report["integrity"] == {
        "duplicate_prompt_groups": 1,
        "duplicate_prompt_records": 1,
        "conflicting_prompt_count": 1,
        "conflict_policy": "report",
    }
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["token_counts"] == {
        "prompt": 3,
        "chosen": 3,
        "rejected": 3,
        "prompt_plus_chosen": 5,
        "prompt_plus_rejected": 5,
    }


def test_dpo_conflict_error_policy_stops_before_output(tmp_path):
    source, output = tmp_path / "raw.jsonl", tmp_path / "prepared.jsonl"
    write_rows(
        source,
        [
            {"prompt": "p", "chosen": "a", "rejected": "b"},
            {"prompt": "p", "chosen": "b", "rejected": "a"},
        ],
    )
    with pytest.raises(ValueError, match="contradictory preferences"):
        prepare_dpo_dataset(
            source,
            output,
            tokenizer=SpaceTokenizer(),
            max_sequence_length=8,
            conflict_policy="error",
        )
    assert not output.exists()


def test_dpo_truncates_answers_and_rejects_overlong_prompt(tmp_path):
    source, output = tmp_path / "raw.jsonl", tmp_path / "prepared.jsonl"
    write_rows(
        source,
        [
            {"prompt": "question ", "chosen": "one two three", "rejected": "four five six"},
            {"prompt": "one two three four", "chosen": "answer", "rejected": "other"},
        ],
    )
    report = prepare_dpo_dataset(
        source,
        output,
        tokenizer=SpaceTokenizer(),
        max_sequence_length=3,
        overflow_policy="truncate",
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["token_counts"]["prompt_plus_chosen"] <= 3
    assert report["output"]["rejected_by_reason"] == {"prompt_too_long": 1}


def test_dpo_prompt_level_split_overlap_and_normalization():
    report = check_dpo_split_overlap(
        {
            "train": [{"prompt": "  Help me! "}],
            "validation": [{"prompt": "help   me!"}],
        },
        normalization="nfc_casefold_whitespace",
    )
    assert report["has_overlap"]
    assert report["integrity_scope"] == "DPO prompt-level pair leakage"


def test_dpo_rejects_pairs_that_become_identical_after_truncation(tmp_path):
    source, output = tmp_path / "raw.jsonl", tmp_path / "prepared.jsonl"
    write_rows(
        source,
        [{"prompt": "p ", "chosen": "x a", "rejected": "x b"}],
    )
    report = prepare_dpo_dataset(
        source,
        output,
        tokenizer=SpaceTokenizer(),
        max_sequence_length=2,
        overflow_policy="truncate",
    )
    assert report["output"]["documents"] == 0
    assert report["output"]["rejected_by_reason"] == {"identical_after_truncation": 1}
