#!/usr/bin/env python3
"""阶段七 Bad Case 挖掘测试。"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "fine_tuning" / "scripts"))

from mine_stage7_bad_cases import (  # noqa: E402
    build_golden_candidates,
    classify_trace,
    mine_bad_cases,
    sample_traces,
)


def test_classify_multiple_bad_types() -> None:
    trace = {
        "provider": "base",
        "latency_ms": 9000,
        "answer_chars": 12,
        "prompt_chars": 9000,
        "context_count": 0,
        "used_citations": [],
        "is_refusal": False,
        "has_answer": True,
        "error": "",
    }
    bad_types = classify_trace(
        trace,
        latency_threshold_ms=8000,
        prompt_threshold_chars=8000,
        short_answer_chars=20,
    )
    assert bad_types == ["no_context_answered", "high_latency", "long_prompt", "short_answer"]


def test_mine_sample_traces() -> None:
    bad_cases = mine_bad_cases(sample_traces())
    type_set = {bad_type for row in bad_cases for bad_type in row["bad_types"]}
    assert "no_context_answered" in type_set
    assert "missing_citation" in type_set
    assert "sft_over_refusal_candidate" in type_set
    assert "high_latency" in type_set
    assert bad_cases[0]["severity"] >= bad_cases[-1]["severity"]


def test_golden_candidates_are_deduped() -> None:
    bad_cases = [
        {
            "trace_id": "t1",
            "ts": "2026-06-07T10:00:00+08:00",
            "query_hash": "sha256:q1",
            "provider": "base",
            "model_name": "m",
            "bad_types": ["missing_citation"],
            "severity": 2,
        },
        {
            "trace_id": "t2",
            "ts": "2026-06-07T11:00:00+08:00",
            "query_hash": "sha256:q1",
            "provider": "sft",
            "model_name": "kb-sft",
            "bad_types": ["sft_over_refusal_candidate"],
            "severity": 2,
        },
    ]
    candidates = build_golden_candidates(bad_cases)
    assert len(candidates) == 1
    assert candidates[0]["source_trace_id"] == "t2"
    assert candidates[0]["needs_human_label"] is True


def main() -> None:
    test_classify_multiple_bad_types()
    test_mine_sample_traces()
    test_golden_candidates_are_deduped()
    print("[ok] stage7 bad case mining")


if __name__ == "__main__":
    main()
