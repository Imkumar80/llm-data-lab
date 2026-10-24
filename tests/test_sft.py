import json

import pytest

from llm_data_lab.adapters.sft import prepare_sft_dataset


class ChatTokenizer:
    name_or_path = "fixture/chat-tokenizer"
    init_kwargs = {"_commit_hash": "chat-revision"}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_dict,
        return_assistant_tokens_mask,
    ):
        ids, mask = [], []
        for message in messages:
            for token in message["content"].split():
                ids.append(len(ids) + 1)
                mask.append(int(message["role"] == "assistant"))
        return {"input_ids": ids, "assistant_masks": mask}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_sft_prepare_validates_multiple_turns_and_exports_assistant_mask(tmp_path):
    source, output = tmp_path / "raw.jsonl", tmp_path / "prepared.jsonl"
    write_rows(
        source,
        [
            {
                "messages": [
                    {"role": "system", "content": "be helpful"},
                    {"role": "user", "content": "first question"},
                    {"role": "assistant", "content": "first answer"},
                    {"role": "user", "content": "follow up"},
                    {"role": "assistant", "content": "second answer"},
                ],
                "id": "fixture-1",
            }
        ],
    )
    report = prepare_sft_dataset(
        source, output, tokenizer=ChatTokenizer(), max_sequence_length=20
    )
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["id"] == "fixture-1"
    assert len(row["input_ids"]) == 10
    assert row["assistant_mask"] == [0, 0, 0, 0, 1, 1, 0, 0, 1, 1]
    assert report["output"]["documents"] == 1
    assert report["output"]["token_lengths"]["total_tokens"] == 10


def test_sft_reports_missing_empty_and_invalid_assistant_messages(tmp_path):
    source, output = tmp_path / "raw.jsonl", tmp_path / "prepared.jsonl"
    write_rows(
        source,
        [
            {"messages": [{"role": "user", "content": "hello"}]},
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "  "},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "moderator", "content": "not allowed"},
                    {"role": "assistant", "content": "response"},
                ]
            },
        ],
    )
    report = prepare_sft_dataset(source, output, tokenizer=ChatTokenizer(), max_sequence_length=10)
    assert report["output"]["documents"] == 0
    assert report["output"]["rejected_by_reason"] == {
        "missing_assistant_response": 1,
        "empty_assistant_response": 1,
        "invalid_role": 1,
    }


def test_sft_truncation_keeps_mask_aligned_and_requires_supervision(tmp_path):
    source, output = tmp_path / "raw.jsonl", tmp_path / "prepared.jsonl"
    write_rows(
        source,
        [
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "one two three"},
                ]
            }
        ],
    )
    report = prepare_sft_dataset(
        source,
        output,
        tokenizer=ChatTokenizer(),
        max_sequence_length=2,
        overflow_policy="truncate",
        truncation_side="right",
    )
    row = json.loads(output.read_text(encoding="utf-8"))
    assert len(row["input_ids"]) == len(row["assistant_mask"]) == 2
    assert sum(row["assistant_mask"]) > 0
    assert report["output"]["documents"] == 1

    class NoAssistantTargets(ChatTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": [1, 2], "assistant_masks": [0, 0]}

    report = prepare_sft_dataset(
        source,
        tmp_path / "no-targets.jsonl",
        tokenizer=NoAssistantTargets(),
        max_sequence_length=4,
    )
    assert report["output"]["rejected_by_reason"] == {"no_supervised_tokens": 1}


def test_sft_template_errors_and_mask_mismatch_are_actionable(tmp_path):
    source = tmp_path / "raw.jsonl"
    write_rows(
        source,
        [{"messages": [{"role": "assistant", "content": "answer"}]}],
    )

    class BrokenTemplate:
        def apply_chat_template(self, *args, **kwargs):
            raise ValueError("private conversation text")

    with pytest.raises(RuntimeError, match="Chat template failed.*ValueError") as error:
        prepare_sft_dataset(
            source, tmp_path / "failed.jsonl", tokenizer=BrokenTemplate(), max_sequence_length=4
        )
    assert "private conversation" not in str(error.value)

    class MismatchedMask(ChatTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": [1, 2], "assistant_masks": [1]}

    with pytest.raises(ValueError, match="lengths differ"):
        prepare_sft_dataset(
            source,
            tmp_path / "mismatch.jsonl",
            tokenizer=MismatchedMask(),
            max_sequence_length=4,
        )

    class InvalidMask(ChatTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": [1], "assistant_masks": [2]}

    with pytest.raises(ValueError, match="only 0 or 1"):
        prepare_sft_dataset(
            source,
            tmp_path / "invalid-mask.jsonl",
            tokenizer=InvalidMask(),
            max_sequence_length=4,
        )
