# AG News data-quality ablation

This report compares four data-preparation conditions on a fixed public dataset slice. It reports corpus statistics only; no language model was trained or evaluated.

## Dataset and reproducibility

- Dataset: [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news)
- Hub revision: `eb185aade064a813bc0b7f42de02595523103ca4`
- Config/split: `default` / `train`
- Sample: first 300 rows (row indices 0–299) (300 rows)
- Input JSONL SHA-256: `073b6e0b2201d3a35fbbe5c77ce38511f93c379ccf8d685c9c17853a992c8e46`
- Acquisition API: [https://datasets-server.huggingface.co/rows](https://datasets-server.huggingface.co/rows)
- License note: Hugging Face lists the license as unknown. The dataset card describes the upstream AG News corpus as intended for research and other non-commercial activities; verify upstream terms before reuse.
- The downloaded sample and condition corpora are written under the ignored `data/` directory; the committed report contains no article text.

Recreate the sample and report from the repository root with:

```powershell
python examples/data_quality_ablation/run_public_dataset.py
```

## Results

| Condition | Documents | Retained | Characters | Mean chars/doc | Mean words/doc |
|---|---:|---:|---:|---:|---:|
| `raw` | 300 | 100.0% | 82734 | 275.8 | 44.0 |
| `clean` | 300 | 100.0% | 82064 | 273.5 | 44.2 |
| `clean_dedupe` | 300 | 100.0% | 82064 | 273.5 | 44.2 |
| `full_filter` | 205 | 68.3% | 68130 | 332.3 | 53.3 |

Label counts (the source sample’s original class labels):

| Condition | World | Sports | Business | Sci/Tech |
|---|---:|---:|---:|---:|
| `raw` | 0 | 0 | 78 | 222 |
| `clean` | 0 | 0 | 78 | 222 |
| `clean_dedupe` | 0 | 0 | 78 | 222 |
| `full_filter` | 0 | 0 | 43 | 162 |

Source-order class counts are World: 0, Sports: 0, Business: 78, Sci/Tech: 222. Absent source classes: World, Sports. After full filtering, counts are World: 0, Sports: 0, Business: 43, Sci/Tech: 162. Consider both total retention and changes in class mix.

### Stage effects

- `clean`: cleaned 300 documents.
- `clean_dedupe`: cleaned 300 documents; removed 0 exact duplicates.
- `full_filter`: cleaned 300 documents; removed 0 exact duplicates; filtered 95 documents (too_short: 95).

Filter thresholds: minimum 200 characters, maximum 100000 characters, and minimum unique-word ratio 0.30.

## Interpretation

These results describe only how this particular 300-row prefix responds to the configured preprocessing steps. The prefix is not a random sample of the full corpus. Exact deduplication uses cleaned text, so duplicates with different non-text labels retain only their first row. The filter is a heuristic; retention does not measure usefulness or model quality. For causal claims about data quality, train matched models and evaluate them on the same held-out set.

The machine-readable run, per-condition manifests, and filter report are in `data/ag_news_ablation/` after running the command above.
