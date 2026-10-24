"""Optional GPU smoke run using pinned TinyStories and SmolLM2 revisions.

This is an example training script, not part of the core package. It is designed
for a CUDA Colab runtime and writes downloaded/prepared data and checkpoints only
under ignored ``data/`` and ``checkpoints/`` directories.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import sys
import time
from collections.abc import Sequence
from itertools import islice
from pathlib import Path
from typing import Any

from llm_data_lab.adapters.cpt import prepare_cpt_dataset
from llm_data_lab.leakage import check_split_overlap
from llm_data_lab.pipeline import _git_commit, sha256_file

DATASET_ID = "roneneldan/TinyStories"
DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
TRAIN_STORIES = 5_000
VALIDATION_STORIES = 512
SEQUENCE_LENGTH = 256
TRAIN_TOKEN_BUDGET = 200_000
VALIDATION_TOKEN_BUDGET = 25_600
SEED = 42


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _tokenize_and_pack(dataset: Any, tokenizer: Any, sequence_length: int) -> Any:
    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["text"], add_special_tokens=False, truncation=False)

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing prepared stories",
    )
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("The pinned tokenizer must define eos_token_id for story boundaries")

    def pack(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        stream: list[int] = []
        for ids in batch["input_ids"]:
            stream.extend(ids)
            stream.append(eos_id)
        block_count = len(stream) // sequence_length
        packed = [
            stream[index * sequence_length : (index + 1) * sequence_length]
            for index in range(block_count)
        ]
        return {"input_ids": packed}

    packed = tokenized.map(
        pack,
        batched=True,
        remove_columns=tokenized.column_names,
        desc=f"Packing {sequence_length}-token blocks",
    )
    if len(packed) == 0:
        raise ValueError("Preparation produced no complete training blocks")
    return packed


def run(output_dir: Path, *, train_stories: int = TRAIN_STORIES) -> dict[str, Any]:
    try:
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError('Install the optional stack with `pip install -e ".[training]"`.') from exc

    if not torch.cuda.is_available():
        raise RuntimeError("This pilot expects a CUDA Colab runtime, such as an A100.")
    if train_stories < 5_000 or train_stories > 10_000:
        raise ValueError("train_stories must be between 5,000 and 10,000")

    data_dir = Path("data/cpt_smoke")
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_train_path = data_dir / "tinystories_train_raw.jsonl"
    raw_validation_path = data_dir / "tinystories_validation_raw.jsonl"
    prepared_train_path = data_dir / "tinystories_train_prepared.jsonl"
    prepared_validation_path = data_dir / "tinystories_validation_prepared.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    train_source = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        split="train",
        streaming=True,
    )
    validation_source = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        split="validation",
        streaming=True,
    )
    train_rows = list(islice(train_source, train_stories))
    validation_rows = list(islice(validation_source, VALIDATION_STORIES))
    if len(train_rows) != train_stories or len(validation_rows) != VALIDATION_STORIES:
        raise ValueError("Pinned TinyStories revision returned fewer rows than requested")
    _write_jsonl(
        raw_train_path,
        [
            {"text": row["text"], "source": DATASET_ID, "source_row": index}
            for index, row in enumerate(train_rows)
        ],
    )
    _write_jsonl(
        raw_validation_path,
        [
            {"text": row["text"], "source": DATASET_ID, "source_row": index}
            for index, row in enumerate(validation_source)
        ],
    )

    train_report = prepare_cpt_dataset(
        raw_train_path,
        prepared_train_path,
        tokenizer=tokenizer,
        max_sequence_length=SEQUENCE_LENGTH - 1,
        token_budget=TRAIN_TOKEN_BUDGET,
        overflow_policy="truncate",
        report_path=data_dir / "train_preparation_report.json",
    )
    validation_report = prepare_cpt_dataset(
        raw_validation_path,
        prepared_validation_path,
        tokenizer=tokenizer,
        max_sequence_length=SEQUENCE_LENGTH - 1,
        token_budget=VALIDATION_TOKEN_BUDGET,
        overflow_policy="truncate",
        report_path=data_dir / "validation_preparation_report.json",
    )
    overlap = check_split_overlap(
        {"train": prepared_train_path, "validation": prepared_validation_path}
    )
    if overlap["has_overlap"]:
        raise ValueError("Prepared train and validation splits contain overlapping stories")

    datasets = load_dataset(
        "json",
        data_files={"train": str(prepared_train_path), "validation": str(prepared_validation_path)},
    )
    train_dataset = _tokenize_and_pack(datasets["train"], tokenizer, SEQUENCE_LENGTH)
    validation_dataset = _tokenize_and_pack(datasets["validation"], tokenizer, SEQUENCE_LENGTH)
    actual_train_tokens = len(train_dataset) * SEQUENCE_LENGTH
    actual_validation_tokens = len(validation_dataset) * SEQUENCE_LENGTH
    if not 100_000 <= actual_train_tokens <= 250_000:
        raise ValueError(
            f"Packed training set has {actual_train_tokens} tokens; expected 100k–250k"
        )

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    bf16 = bool(torch.cuda.is_bf16_supported())
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=10,
        seed=SEED,
        data_seed=SEED,
        bf16=bf16,
        fp16=not bf16,
        report_to="none",
        dataloader_num_workers=2,
    )
    def collate_fixed_blocks(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        input_ids = torch.tensor([feature["input_ids"] for feature in features], dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids.clone()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collate_fixed_blocks,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train_result = trainer.train()
    training_seconds = time.perf_counter() - started
    evaluation = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    peak_memory_bytes = torch.cuda.max_memory_allocated()

    del trainer, model
    torch.cuda.empty_cache()
    reloaded_model = AutoModelForCausalLM.from_pretrained(output_dir).to("cuda")
    reload_trainer = Trainer(
        model=reloaded_model,
        args=training_args,
        eval_dataset=validation_dataset,
        data_collator=collate_fixed_blocks,
    )
    reload_evaluation = reload_trainer.evaluate()
    reload_ok = math.isclose(
        float(evaluation["eval_loss"]), float(reload_evaluation["eval_loss"]), rel_tol=1e-4
    )
    if not reload_ok:
        raise RuntimeError("Reloaded checkpoint evaluation did not match the saved model")

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "dataset": {
            "identifier": DATASET_ID,
            "revision": DATASET_REVISION,
            "train_rows_requested": train_stories,
            "validation_rows_requested": VALIDATION_STORIES,
            "selection": {
                "train": f"first {train_stories} rows via streaming",
                "validation": f"first {VALIDATION_STORIES} rows via streaming",
            },
            "train_input_sha256": sha256_file(raw_train_path),
            "validation_input_sha256": sha256_file(raw_validation_path),
            "license": "cdla-sharing-1.0",
        },
        "model": {
            "identifier": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": "apache-2.0",
        },
        "preparation": {
            "sequence_length": SEQUENCE_LENGTH,
            "train_token_budget": TRAIN_TOKEN_BUDGET,
            "validation_token_budget": VALIDATION_TOKEN_BUDGET,
            "train": train_report,
            "validation": validation_report,
            "split_overlap": overlap,
        },
        "training": {
            "actual_train_tokens": actual_train_tokens,
            "actual_validation_tokens": actual_validation_tokens,
            "optimizer_steps": train_result.global_step,
            "train_loss": train_result.training_loss,
            "validation_loss": evaluation["eval_loss"],
            "validation_perplexity": math.exp(float(evaluation["eval_loss"])),
            "training_seconds": training_seconds,
            "peak_gpu_memory_bytes": peak_memory_bytes,
            "bf16": bf16,
            "fp16": not bf16,
            "checkpoint_reload_ok": reload_ok,
            "reloaded_validation_loss": reload_evaluation["eval_loss"],
        },
        "configuration": {
            "code_commit": _git_commit(),
            "seed": SEED,
            "epochs": 1,
            "per_device_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.03,
            "scheduler": "cosine",
            "optimizer": "adamw_torch",
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "max_sequence_length": SEQUENCE_LENGTH,
            "hardware": torch.cuda.get_device_name(0),
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
        },
        "software_versions": {
            "llm_data_lab": importlib.metadata.version("llm-data-lab"),
            "torch": torch.__version__,
            "datasets": importlib.metadata.version("datasets"),
            "transformers": importlib.metadata.version("transformers"),
        },
        "scope_note": "Smoke integration run only; this result does not establish a model-quality improvement.",
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/cpt-smoke"))
    parser.add_argument("--train-stories", type=int, default=TRAIN_STORIES)
    args = parser.parse_args()
    run(args.output_dir, train_stories=args.train_stories)


if __name__ == "__main__":
    main()
