"""Minimal optional TRL DPO integration for an already prepared JSONL file."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise SystemExit('Install the optional stack with `pip install -e ".[trl]"`.') from exc

    dataset = load_dataset("json", data_files=str(args.prepared), split="train")
    required = {"prompt", "chosen", "rejected"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"Prepared DPO JSONL is missing columns: {sorted(missing)}")
    dataset = dataset.remove_columns(
        [column for column in dataset.column_names if column not in required]
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision)
    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(output_dir=str(args.output_dir), report_to="none"),
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
