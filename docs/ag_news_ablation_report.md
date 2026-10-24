# AG News data-quality ablation

This report compares four data-preparation conditions on a fixed public dataset slice. It reports corpus statistics only; no language model was trained or evaluated.

## Dataset and reproducibility

- Dataset: [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news)
- Hub revision: `eb185aade064a813bc0b7f42de02595523103ca4`
- Config/split: `default` / `train`
- Sampling method: Divide the training split's source-row index range into four equal strata; select one seeded contiguous 100-row block uniformly within each stratum.
- Sampling seed: `42`; requested/actual: 400/400 rows
- Sample class counts: {'World': 80, 'Sports': 79, 'Business': 109, 'Sci/Tech': 132}
- Training split rows: 120000
- Selected source row blocks: [{'start_inclusive': 0, 'end_exclusive': 30000, 'selected_block_start': 20952, 'selected_block_end_exclusive': 21052}, {'start_inclusive': 30000, 'end_exclusive': 60000, 'selected_block_start': 33648, 'selected_block_end_exclusive': 33748}, {'start_inclusive': 60000, 'end_exclusive': 90000, 'selected_block_start': 60819, 'selected_block_end_exclusive': 60919}, {'start_inclusive': 90000, 'end_exclusive': 120000, 'selected_block_start': 114299, 'selected_block_end_exclusive': 114399}]
- Input JSONL SHA-256: `054d5a8593b706fd750cbe391613467c9195a3dd410dbbeb4aeb3f4cca930764`
- Acquisition API: [https://datasets-server.huggingface.co/rows](https://datasets-server.huggingface.co/rows)
- Revision limitation: The Dataset Viewer rows endpoint does not accept an immutable Hub revision. The script checks the Hub revision before and after fetching and pins the exact serialized sample checksum to detect changed viewer results.
- License note: Hugging Face lists the license as unknown. The dataset card describes the upstream AG News corpus as intended for research and other non-commercial activities; verify upstream terms before reuse.
- The downloaded sample and condition corpora are written under the ignored `data/` directory; the committed report contains no article text.

Recreate the sample and report from the repository root with:

```powershell
python examples/data_quality_ablation/run_public_dataset.py
```

## Observed results

Retention denominator for every condition is the same sampled input corpus.

| Condition | Input | Output | Retained | Characters | Whitespace words | Mean chars/doc | Length chars min / p25 / median / p75 / p90 / max |
|---|---:|---:|---:|---:|---:|---:|---|
| `raw` | 400 | 400 | 100.0% | 92712 | 14867 | 231.8 | 101 / 193 / 227 / 259 / 288 / 906 |
| `clean` | 400 | 400 | 100.0% | 92150 | 14923 | 230.4 | 101 / 193 / 226 / 258 / 287 / 851 |
| `clean_dedupe` | 400 | 400 | 100.0% | 92150 | 14923 | 230.4 | 101 / 193 / 226 / 258 / 287 / 851 |
| `full_filter` | 400 | 277 | 69.2% | 71268 | 11470 | 257.3 | 200 / 225 / 245 / 273 / 304 / 851 |

Label counts after each condition:

| Condition | World | Sports | Business | Sci/Tech |
|---|---:|---:|---:|---:|
| `raw` | 80 | 79 | 109 | 132 |
| `clean` | 80 | 79 | 109 | 132 |
| `clean_dedupe` | 80 | 79 | 109 | 132 |
| `full_filter` | 57 | 55 | 74 | 91 |

Sample class counts before preparation are World: 80, Sports: 79, Business: 109, Sci/Tech: 132. All source classes appear in this prefix. After full filtering, counts are World: 57, Sports: 55, Business: 74, Sci/Tech: 91. Consider both total retention and changes in class mix.

Source document counts after each condition:

| Condition | fancyzhx/ag_news |
|---|---:|
| `raw` | 400 |
| `clean` | 400 |
| `clean_dedupe` | 400 |
| `full_filter` | 277 |

Per-condition corpus SHA-256:

- `raw`: `054d5a8593b706fd750cbe391613467c9195a3dd410dbbeb4aeb3f4cca930764`
- `clean`: `99535cb689f98c9223d4e66435c6349e3675bd7ce848d84a5c1460dc5ed2cd2a`
- `clean_dedupe`: `5da71d248460409d05fdca300de7d5f41263562f6a094270731ede2e6adb37b9`
- `full_filter`: `b3a658a862a85153ac14878ef71a10205dfe63b421fe39eb07e005cf5218b6f5`

### Stage effects

- `raw` validation/copy: 400 input → 400 output documents.
- `clean`: cleaned 400 → 400 documents; empty after cleaning: 0.
- `clean_dedupe`: cleaned 400 → 400 documents; empty after cleaning: 0; deduplicated 400 → 400 (removed 0 exact duplicates).
- `full_filter`: cleaned 400 → 400 documents; empty after cleaning: 0; deduplicated 400 → 400 (removed 0 exact duplicates); filtered 400 → 277 (too_short: 123).

Filter thresholds: minimum 200 characters, maximum 100000 characters, and minimum unique-word ratio 0.30.
The minimum-character rule is the existing general-purpose default (200), kept unchanged for this demonstration; it was not tuned on a test split or model result.
Pre-filter documents shorter than 200 characters: 123; length min/median/max: 101/176/199 characters.

## Interpretation

This is a source-index-stratified block sample, not a simple random sample or a representative sample of the full corpus. Contiguous blocks can cluster similar documents, and class composition is incidental rather than balanced by design; report observed class counts without generalizing them. Exact deduplication uses cleaned text, so duplicates with different non-text labels retain only their first row. The filter is a heuristic; retention does not measure usefulness or model quality. For causal claims about data quality, train matched models and evaluate them on the same held-out set.

The machine-readable run, per-condition manifests, and filter report are in `data/ag_news_ablation/` after running the command above. Raw article text is not included in this report.
