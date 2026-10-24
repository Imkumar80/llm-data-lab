# CPT smoke workflow

This example is an opt-in GPU integration run. It uses the immutable [TinyStories dataset revision](https://huggingface.co/datasets/roneneldan/TinyStories/commit/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64) `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` and [SmolLM2-135M Base revision](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2) `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. The dataset card declares CDLA-Sharing-1.0; the model card declares Apache-2.0. Confirm that the terms fit your use before downloading or training.

Install in a CUDA Colab runtime:

```bash
pip install -e ".[training]"
python examples/cpt_smoke/run_training.py --train-stories 5000
```

The script selects 5,000 training stories and 512 stories from TinyStories' separate validation split, prepares them under 256-token document limits and a 200k training-token budget, checks for text overlap, packs fixed-length training blocks and runs one epoch. It enables BF16 when supported, otherwise FP16, then evaluates held-out loss/perplexity, saves and reloads the checkpoint, and records optimizer steps, token counts, runtime, peak GPU memory, revisions, checksums and configuration.

Data is written under ignored `data/cpt_smoke/`; checkpoints and the final manifest go under ignored `checkpoints/cpt-smoke/`. The script requires CUDA and was not run by unit tests. A successful smoke run validates integration and reproducibility only; it does not show that a data-preparation choice improves model quality.
