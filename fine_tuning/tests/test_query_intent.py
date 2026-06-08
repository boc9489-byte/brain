#!/usr/bin/env python3
"""查询意图识别规则测试。"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from knowledge.processor.query_processor.intent import (  # noqa: E402
    INTENT_GENERAL,
    INTENT_IMAGE,
    INTENT_INSTALL,
    INTENT_OPERATION,
    INTENT_PARAMETER,
    INTENT_TROUBLESHOOTING,
    classify_query_intent,
    normalize_intent,
)


def test_classify_query_intent() -> None:
    assert classify_query_intent("RS-12 如何测量电阻？") == INTENT_OPERATION
    assert classify_query_intent("H3C 网关怎么接线配置？") == INTENT_INSTALL
    assert classify_query_intent("设备连不上网络，提示认证失败") == INTENT_TROUBLESHOOTING
    assert classify_query_intent("L420x 支持的电压和接口参数") == INTENT_PARAMETER
    assert classify_query_intent("有没有接线图或结构示意图？") == INTENT_IMAGE
    assert classify_query_intent("") == INTENT_GENERAL


def test_normalize_intent() -> None:
    assert normalize_intent("operation") == INTENT_OPERATION
    assert normalize_intent("bad-value") == INTENT_GENERAL
    assert normalize_intent(None) == INTENT_GENERAL


def main() -> None:
    test_classify_query_intent()
    test_normalize_intent()
    print("[ok] query intent")


if __name__ == "__main__":
    main()
