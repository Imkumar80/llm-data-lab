# Optional TRL integrations

These examples show how the SFT and DPO adapter outputs can be passed to existing TRL trainers. They do not make TRL a core dependency.

TRL's [SFT trainer](https://huggingface.co/docs/trl/en/sft_trainer) accepts conversational `messages`, while its [preference dataset format](https://huggingface.co/docs/trl/en/dataset_formats) uses `prompt`, `chosen`, and `rejected`. The examples preserve those contracts and leave trainer behavior with TRL.

Install the optional trainer stack with `pip install -e ".[trl]"`. Prepare the input with `llm-data-lab sft-prepare` or `dpo-prepare`, then run:

```bash
python examples/trl_integrations/sft.py \
  --prepared data/sft_prepared.jsonl --model /path/to/model \
  --revision MODEL_COMMIT --output-dir checkpoints/sft

python examples/trl_integrations/dpo.py \
  --prepared data/dpo_prepared.jsonl --model /path/to/model \
  --revision MODEL_COMMIT --output-dir checkpoints/dpo
```

Use the same model/tokenizer revision used during preparation. TRL still owns training behavior and may apply its own tokenization and loss-mask logic. The scripts are integration examples, not tested training runs.
