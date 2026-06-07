"""回答链路 trace 记录。

阶段六默认关闭 trace。开启后写 JSONL，默认只写脱敏字段，避免把业务问题和答案原文直接落盘。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from knowledge.utils.client.answer_model_config import AnswerModelSettings


REPO_ROOT = Path(__file__).resolve().parents[3]
_WRITE_LOCK = threading.Lock()


def _bool_env(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _repo_path(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _hash_text(text: str) -> str:
    digest = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _truncate(text: str, max_chars: int) -> str:
    value = text or ""
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "...[truncated]"


@dataclass(frozen=True)
class AnswerTraceSettings:
    """回答 trace 配置。"""

    enabled: bool
    path: Path
    include_text: bool
    include_context: bool
    max_text_chars: int

    @classmethod
    def from_env(cls) -> "AnswerTraceSettings":
        return cls(
            enabled=_bool_env("ANSWER_TRACE_ENABLED", False),
            path=_repo_path(os.getenv("ANSWER_TRACE_PATH", "fine_tuning/data/online/answer_traces.jsonl")),
            include_text=_bool_env("ANSWER_TRACE_INCLUDE_TEXT", False),
            include_context=_bool_env("ANSWER_TRACE_INCLUDE_CONTEXT", False),
            max_text_chars=max(20, _int_env("ANSWER_TRACE_MAX_TEXT_CHARS", 500)),
        )

    def safe_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "path": str(self.path),
            "include_text": self.include_text,
            "include_context": self.include_context,
            "max_text_chars": self.max_text_chars,
        }


def _citations_in(answer: str) -> List[str]:
    import re

    return sorted(set(re.findall(r"\[(?:C)?(\d+)\]", answer or "")))


def _is_refusal(answer: str) -> bool:
    cues = [
        "无法回答",
        "无法据此",
        "无法确定",
        "资料不足",
        "没有相关",
        "未提及",
        "缺少",
        "存在冲突",
        "不一致",
    ]
    return any(cue in (answer or "") for cue in cues)


def _context_sources(contexts: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for idx, context in enumerate(contexts, 1):
        sources.append(
            {
                "rank": idx,
                "chunk_id": context.get("chunk_id"),
                "title": context.get("title"),
                "file_title": context.get("file_title"),
                "score": context.get("score"),
            }
        )
    return sources


def build_answer_trace(
    state: Dict[str, Any],
    *,
    latency_ms: int,
    settings: Optional[AnswerTraceSettings] = None,
    model_settings: Optional[AnswerModelSettings] = None,
    error: str = "",
) -> Dict[str, Any]:
    """构造回答 trace，不负责写入。"""
    settings = settings or AnswerTraceSettings.from_env()
    model_settings = model_settings or AnswerModelSettings.from_env()
    query = state.get("rewritten_query") or state.get("query") or ""
    answer = state.get("answer") or ""
    prompt = state.get("prompt") or ""
    contexts = state.get("reranked_docs") or []

    trace: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).astimezone().isoformat(),
        "task_id": state.get("task_id", ""),
        "provider": model_settings.provider,
        "model_name": model_settings.model_name,
        "is_stream": bool(state.get("is_stream")),
        "latency_ms": latency_ms,
        "query_hash": _hash_text(query),
        "answer_hash": _hash_text(answer),
        "query_chars": len(query),
        "answer_chars": len(answer),
        "prompt_chars": len(prompt),
        "context_count": len(contexts),
        "used_citations": _citations_in(answer),
        "is_refusal": _is_refusal(answer),
        "has_answer": bool(answer.strip()),
        "error": error,
    }

    if settings.include_text:
        trace["query"] = _truncate(query, settings.max_text_chars)
        trace["answer_preview"] = _truncate(answer, settings.max_text_chars)
    if settings.include_context:
        trace["context_sources"] = _context_sources(contexts)

    return trace


def write_answer_trace(trace: Dict[str, Any], settings: Optional[AnswerTraceSettings] = None) -> None:
    """写入 JSONL。调用方保证 settings.enabled 为 true。"""
    settings = settings or AnswerTraceSettings.from_env()
    settings.path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(trace, ensure_ascii=False, separators=(",", ":"))
    with _WRITE_LOCK:
        with settings.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def record_answer_trace(
    state: Dict[str, Any],
    *,
    latency_ms: int,
    logger: Any = None,
    error: str = "",
) -> Optional[Dict[str, Any]]:
    """按环境配置记录回答 trace。

    任何异常都会被吞掉并写日志，避免观测逻辑影响回答主链路。
    """
    try:
        settings = AnswerTraceSettings.from_env()
        if not settings.enabled:
            return None
        trace = build_answer_trace(state, latency_ms=latency_ms, settings=settings, error=error)
        write_answer_trace(trace, settings)
        return trace
    except Exception as exc:
        if logger:
            logger.warning(f"记录回答 trace 失败，原因：{exc}")
        return None
