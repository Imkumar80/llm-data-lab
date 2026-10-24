import json

from llm_data_lab import cli


class WordTokenizer:
    name_or_path = "fixture/cli-tokenizer"
    init_kwargs = {"_commit_hash": "fixture-commit"}

    def encode(self, text, *, add_special_tokens=True):
        return text.split()

    def decode(self, tokens, *, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return " ".join(tokens)


class ChatTokenizer(WordTokenizer):
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_dict,
        return_assistant_tokens_mask,
    ):
        ids, masks = [], []
        for message in messages:
            for _token in message["content"].split():
                ids.append(len(ids) + 1)
                masks.append(int(message["role"] == "assistant"))
        return {"input_ids": ids, "assistant_masks": masks}


def write_jsonl(path, record):
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_cpt_prepare_cli_routes_flags_and_writes_report(tmp_path, monkeypatch, capsys):
    source, output = tmp_path / "raw.jsonl", tmp_path / "prepared.jsonl"
    write_jsonl(source, {"text": "one two three four"})
    monkeypatch.setattr(cli, "_load_tokenizer", lambda *_: WordTokenizer())
    monkeypatch.setattr(
        "sys.argv",
        [
            "llm-data-lab",
            "cpt-prepare",
            "--input",
            str(source),
            "--output",
            str(output),
            "--tokenizer",
            "fixture",
            "--max-sequence-length",
            "3",
            "--overflow-policy",
            "truncate",
        ],
    )

    cli.main()

    result = json.loads(capsys.readouterr().out)
    assert json.loads(output.read_text(encoding="utf-8"))["text"] == "one two three"
    assert result["output"]["documents"] == 1


def test_sft_prepare_cli_routes_chat_tokens_and_mask(tmp_path, monkeypatch, capsys):
    source, output = tmp_path / "raw.jsonl", tmp_path / "prepared.jsonl"
    write_jsonl(
        source,
        {"messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]},
    )
    monkeypatch.setattr(cli, "_load_tokenizer", lambda *_: ChatTokenizer())
    monkeypatch.setattr(
        "sys.argv",
        [
            "llm-data-lab",
            "sft-prepare",
            "--input",
            str(source),
            "--output",
            str(output),
            "--tokenizer",
            "fixture",
            "--max-sequence-length",
            "8",
        ],
    )

    cli.main()

    result = json.loads(capsys.readouterr().out)
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["assistant_mask"] == [0, 1]
    assert result["output"]["documents"] == 1


def test_dpo_prepare_cli_routes_pair_limits_and_report(tmp_path, monkeypatch, capsys):
    source, output = tmp_path / "raw.jsonl", tmp_path / "prepared.jsonl"
    write_jsonl(source, {"prompt": "Question ", "chosen": "good answer", "rejected": "bad answer"})
    monkeypatch.setattr(cli, "_load_tokenizer", lambda *_: WordTokenizer())
    monkeypatch.setattr(
        "sys.argv",
        [
            "llm-data-lab",
            "dpo-prepare",
            "--input",
            str(source),
            "--output",
            str(output),
            "--tokenizer",
            "fixture",
            "--max-sequence-length",
            "8",
        ],
    )

    cli.main()

    result = json.loads(capsys.readouterr().out)
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["token_counts"]["prompt_plus_chosen"] <= 8
    assert result["output"]["documents"] == 1
