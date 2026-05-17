"""Smoke tests — verify the shipping artifacts are well-formed and discoverable.

Pytest exits non-zero on "no tests collected", which would fail CI; these
checks earn their keep by being real (catches a corrupted training file or a
missing reference) rather than a placeholder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_reference_files_present():
    """The reference grounding files are what every generation prompt cites."""
    for name in ("ibs_reference.md", "medication_whitelist.md", "hallucination_blocklist.md"):
        path = REPO_ROOT / "reference" / name
        assert path.is_file(), f"missing {path}"
        assert path.stat().st_size > 0, f"empty {path}"


def test_train_v2_dataset_well_formed():
    """train_v2.jsonl is the 659-entry shipping training set."""
    path = REPO_ROOT / "datasets" / "validated" / "train_v2.jsonl"
    assert path.is_file(), f"missing {path}"
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 659, f"expected 659 entries, got {len(rows)}"
    for i, row in enumerate(rows):
        assert "messages" in row, f"row {i} missing 'messages'"
        roles = [m.get("role") for m in row["messages"]]
        assert "user" in roles, f"row {i} has no user turn"
        assert "assistant" in roles, f"row {i} has no assistant turn"


def test_heldout_eval_set_well_formed():
    """The 50-question held-out eval set the hackathon results were measured on."""
    path = REPO_ROOT / "datasets" / "eval" / "heldout_questions.jsonl"
    assert path.is_file(), f"missing {path}"
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 50, f"expected 50 eval questions, got {len(rows)}"
    required = {"id", "category", "question", "expected_behavior", "red_flag"}
    for i, row in enumerate(rows):
        missing = required - row.keys()
        assert not missing, f"row {i} missing fields: {missing}"


def test_eval_report_v2_ship_criteria_met():
    """The shipping criteria documented in the writeup must hold in the report."""
    path = REPO_ROOT / "scripts" / "evaluate" / "results" / "eval_report_v2.json"
    assert path.is_file(), f"missing {path}"
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["ship_criterion"]["v2_mean_gt_baseline_plus_2sigma"] is True
    assert report["ship_criterion"]["red_flag_4_of_4_all_runs"] is True
    assert report["v2_3run_mean"] > report["baseline_overall"]


@pytest.mark.parametrize(
    "script_rel",
    [
        "scripts/publish/push_model.py",
        "scripts/evaluate/medical_judge.py",
        "scripts/evaluate/quality_filter.py",
    ],
)
def test_pipeline_scripts_present(script_rel: str):
    path = REPO_ROOT / script_rel
    assert path.is_file(), f"missing {path}"
