#!/usr/bin/env python3
"""阶段六回答 trace 测试。"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from knowledge.utils.client.answer_model_config import AnswerModelSettings  # noqa: E402
from knowledge.utils.observability.answer_trace import (  # noqa: E402
    AnswerTraceSettings,
    build_answer_trace,
    record_answer_trace,
)


@contextmanager
def patched_env(values: Dict[str, str]) -> Iterator[None]:
    keys = {
        "ANSWER_TRACE_ENABLED",
        "ANSWER_TRACE_PATH",
        "ANSWER_TRACE_INCLUDE_TEXT",
        "ANSWER_TRACE_INCLUDE_CONTEXT",
        "ANSWER_TRACE_MAX_TEXT_CHARS",
        "ANSWER_MODEL_PROVIDER",
        "ANSWER_OPENAI_API_BASE",
        "ANSWER_BASE_MODEL",
        "ANSWER_SFT_MODEL",
    }
    old = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        os.environ.update(values)
        yield
    finally:
        for key in keys:
            os.environ.pop(key, None)
        for key, value in old.items():
            if value is not None:
                os.environ[key] = value


def fake_state() -> Dict[str, object]:
    return {
        "task_id": "t1",
        "is_stream": False,
        "rewritten_query": "RS-12 如何测量电阻？",
        "answer": "根据资料，应选择电阻档位并连接表笔 [1]。",
        "prompt": "prompt text",
        "reranked_docs": [{"chunk_id": "c1", "title": "电阻测量", "score": 0.9}],
    }


def test_trace_is_redacted_by_default() -> None:
    settings = AnswerTraceSettings(
        enabled=True,
        path=Path("/tmp/not-used.jsonl"),
        include_text=False,
        include_context=False,
        max_text_chars=100,
    )
    model_settings = AnswerModelSettings(
        provider="sft",
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        base_model="Qwen/Qwen2.5-3B-Instruct",
        sft_model="kb-sft",
        temperature=0,
        max_tokens=1024,
        timeout_sec=60,
    )
    trace = build_answer_trace(fake_state(), latency_ms=42, settings=settings, model_settings=model_settings)
    assert trace["provider"] == "sft"
    assert trace["model_name"] == "kb-sft"
    assert trace["latency_ms"] == 42
    assert trace["intent_type"] == "operation"
    assert trace["used_citations"] == ["1"]
    assert "query" not in trace
    assert "answer_preview" not in trace


def test_trace_write_when_enabled() -> None:
    with TemporaryDirectory() as tmp_dir:
        trace_path = str(Path(tmp_dir) / "answer_traces.jsonl")
        with patched_env(
            {
                "ANSWER_TRACE_ENABLED": "true",
                "ANSWER_TRACE_PATH": trace_path,
                "ANSWER_OPENAI_API_BASE": "http://127.0.0.1:8000/v1",
                "ANSWER_BASE_MODEL": "Qwen/Qwen2.5-3B-Instruct",
            }
        ):
            trace = record_answer_trace(fake_state(), latency_ms=11)
            assert trace is not None
            rows = Path(trace_path).read_text(encoding="utf-8").strip().splitlines()
            assert len(rows) == 1
            payload = json.loads(rows[0])
            assert payload["model_name"] == "Qwen/Qwen2.5-3B-Instruct"
            assert payload["context_count"] == 1


def test_trace_disabled_does_not_write() -> None:
    with TemporaryDirectory() as tmp_dir:
        trace_path = Path(tmp_dir) / "answer_traces.jsonl"
        with patched_env(
            {
                "ANSWER_TRACE_ENABLED": "false",
                "ANSWER_TRACE_PATH": str(trace_path),
                "ANSWER_OPENAI_API_BASE": "http://127.0.0.1:8000/v1",
                "ANSWER_BASE_MODEL": "Qwen/Qwen2.5-3B-Instruct",
            }
        ):
            trace = record_answer_trace(fake_state(), latency_ms=11)
            assert trace is None
            assert not trace_path.exists()


def main() -> None:
    test_trace_is_redacted_by_default()
    test_trace_write_when_enabled()
    test_trace_disabled_does_not_write()
    print("[ok] stage6 answer trace")


if __name__ == "__main__":
    main()
