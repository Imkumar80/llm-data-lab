# Evaluation design

This document defines a narrow direction for the evaluation roadmap. Its declarative benchmark, scorer, and resumable-run ideas are informed by [Gouti629/eval-anything](https://github.com/Gouti629/eval-anything); LLM Data Lab keeps them scoped to reproducible text-data experiments. It is a design target, not a claim that evaluation runners are implemented.

## What Eval Anything evaluates

[Gouti629/eval-anything](https://github.com/Gouti629/eval-anything) is primarily a **model-generation benchmark runner**. A YAML entry selects a dataset/split, input and reference fields, prompt settings, and a scorer. The runner loads a model through Hugging Face Transformers or vLLM, generates answers in batches, scores each example, and saves aggregate and per-example outcomes. Its built-in tasks cover math and knowledge questions, multiple choice, and code generation; it also tracks timing and supports resumable runs. See its [benchmark configs](https://github.com/Gouti629/eval-anything/blob/main/eval_anything/default_evals.yaml), [scorers](https://github.com/Gouti629/eval-anything/blob/main/eval_anything/rewards/scorers.py), and [runner](https://github.com/Gouti629/eval-anything/blob/main/eval_anything/runner.py).

LLM Data Lab has a different center of gravity: it prepares and describes corpora so researchers can compare data conditions reproducibly. Its first evaluation stage should answer questions about the **corpus**—retention, token counts, source/language mix, and train/validation overlap. Its later model-evaluation stage can borrow the benchmark runner pattern to ask whether those corpus choices changed a model's held-out performance. Keeping these stages distinct helps connect a model result to the data pipeline that produced its training set.

The useful transfer is the benchmark configuration, explicit scorer contract, per-example result record, and resumable run fingerprint. The project should not import Eval Anything's whole backend/runtime, benchmark catalog, or code-execution scorer as a dependency. Those are beyond the initial scope and make evaluation more expensive and environment-dependent.

## Scope

- Start with text benchmarks over local JSONL or versioned Hugging Face datasets.
- Keep benchmark definitions declarative and independent of the inference backend.
- Make metrics explicit, named, and versioned; save both aggregate scores and per-example outcomes.
- Keep the first release deterministic and auditable. Do not require an LLM judge, model training framework, multimodal runtime, or hosted evaluation service.
- Treat scripts, language rules, and task-specific scoring as configurable extensions, not universal assumptions.

## Benchmark definition

A future benchmark config should specify at least:

```yaml
schema_version: 1
name: example_qa
dataset:
  identifier: organization/dataset
  revision: immutable-revision
  config: default
  split: test
fields:
  prompt: question
  reference: answer
prompt:
  template: "Answer briefly: {prompt}"
generation:
  max_new_tokens: 128
  temperature: 0
metric:
  name: exact_match
  version: 1
limit: null
```

Pin the dataset revision where the source supports it. Any sampling or few-shot selection must record its seed and source split. A benchmark config must not silently download or execute arbitrary code.

## Result artifact and resuming

Write one JSON result artifact per run. Include the benchmark config and its fingerprint, model and tokenizer identifiers/revisions, backend and generation settings, software/code revision, start time, and completion status. For each example, record a stable example identifier or content hash, metric score, and only the prediction/reference details needed for audit. Make retention of raw prompts, references, or model outputs configurable because datasets may contain licensed or sensitive text.

Compute the run fingerprint from the benchmark and dataset revisions, model/tokenizer revisions, metric name/version, generation settings, seed, and relevant code version. Resume only matching incomplete runs, matching examples by stable identity rather than row number. Never merge partial results with a different fingerprint. Aggregate metrics should include the evaluated example count and clearly state the denominator when examples are skipped or invalid.

## Extension points

- **Inference adapters:** accept a text prompt and return generated text plus available timing/token usage metadata. Adapters should not own benchmark definitions or metric semantics.
- **Metrics:** accept prediction and reference and return a numeric score plus structured details. Include simple deterministic metrics first; document normalization behavior.
- **Corpus diagnostics:** allow evaluation slices by supplied source, language, or script metadata. Do not auto-detect or filter languages as a default.

Tokenizer-aware corpus counts and leakage checks remain an earlier roadmap milestone. Evaluation should consume the resulting corpus and manifest metadata rather than duplicate preprocessing logic. Training runners remain optional and outside the initial evaluation interface.
