#!/usr/bin/env python3
"""阶段六线上观测配置检查脚本。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def load_project_env() -> None:
    """加载 knowledge/.env；没有 python-dotenv 时用轻量解析兜底。"""
    env_path = REPO_ROOT / "knowledge" / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return
    load_dotenv(env_path)


load_project_env()

from knowledge.utils.observability.answer_trace import (  # noqa: E402
    AnswerTraceSettings,
    build_answer_trace,
    write_answer_trace,
)


def sample_state() -> dict:
    """构造一条本地 sample，不依赖真实查询链路。"""
    return {
        "task_id": "stage6-sample",
        "is_stream": False,
        "rewritten_query": "RS-12 如何测量电阻？",
        "answer": "根据资料，RS-12 测量电阻时需要选择电阻档位，并连接表笔 [1]。",
        "prompt": "【参考内容】...\n【用户问题】RS-12 如何测量电阻？",
        "reranked_docs": [
            {
                "chunk_id": "chunk-1",
                "title": "电阻测量",
                "file_title": "万用表RS-12的使用.pdf",
                "score": 0.91,
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true", help="只输出 trace 配置。")
    parser.add_argument("--write-sample", action="store_true", help="写入一条 sample trace。")
    args = parser.parse_args()

    settings = AnswerTraceSettings.from_env()
    print("[stage6] answer trace settings:")
    print(json.dumps(settings.safe_dict(), ensure_ascii=False, indent=2))

    trace = build_answer_trace(sample_state(), latency_ms=123, settings=settings)
    print("[stage6] sample trace:")
    print(json.dumps(trace, ensure_ascii=False, indent=2))

    if args.write_sample:
        if not settings.enabled:
            raise SystemExit("[stage6] ANSWER_TRACE_ENABLED is false; set it to true before writing sample.")
        write_answer_trace(trace, settings)
        print(f"[stage6] sample trace written -> {settings.path}")
        return

    if args.check_only:
        print("[stage6] check passed")
        return

    print("[stage6] add --write-sample to test JSONL writing.")


if __name__ == "__main__":
    main()
