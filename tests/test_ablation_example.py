import importlib.util
import json
from pathlib import Path

import pytest

# Load the example by path: examples/ is intentionally not an installed package.
_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "data_quality_ablation"
    / "run_ablation.py"
)
_SPEC = importlib.util.spec_from_file_location("run_ablation_example", _EXAMPLE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load ablation example from {_EXAMPLE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
run_ablation = _MODULE.run_ablation


def write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_ablation_writes_four_conditions_and_manifests(tmp_path):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "runs"
    write_rows(source, [
        {"text": "This is a useful document with enough unique words for filtering.", "source": "a"},
        {"text": "This is a useful document with enough unique words for filtering.", "source": "b"},
        {"text": "ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha ha", "source": "a"},
        {"text": "tiny", "source": "a"},
    ])

    result = run_ablation(
        source,
        output,
        min_chars=20,
        min_unique_word_ratio=0.3,
        seed=123,
    )

    assert set(result["conditions"]) == {"raw", "clean", "clean_dedupe", "full_filter"}
    assert result["conditions"]["raw"]["documents_retained"] == 4
    assert result["conditions"]["clean"]["documents_retained"] == 4
    assert result["conditions"]["clean_dedupe"]["documents_retained"] == 3
    assert result["conditions"]["clean_dedupe"]["stages"]["dedupe"][
        "exact_duplicates_removed"
    ] == 1
    assert result["conditions"]["full_filter"]["documents_retained"] == 1
    assert result["conditions"]["full_filter"]["stages"]["filter"][
        "rejected_by_reason"
    ] == {"low_unique_word_ratio": 1, "too_short": 1}
    for name in result["conditions"]:
        condition_dir = output / name
        assert (condition_dir / "corpus.jsonl").exists()
        assert (condition_dir / "manifest.json").exists()
        manifest = json.loads((condition_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["evaluation"]["status"] == "not_run"
    assert (output / "ablation_summary.json").exists()


def test_ablation_fails_on_non_string_text_field(tmp_path):
    source = tmp_path / "input.jsonl"
    write_rows(source, [{"text": 123}])
    with pytest.raises(TypeError, match="must be a string"):
        run_ablation(source, tmp_path / "runs")
