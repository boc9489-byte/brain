#!/usr/bin/env python3
"""阶段五回答模型配置规则测试。"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from knowledge.utils.client.answer_model_config import AnswerModelSettings  # noqa: E402


@contextmanager
def patched_env(values: Dict[str, str]) -> Iterator[None]:
    """临时覆盖阶段五相关环境变量。"""
    keys = {
        "ANSWER_MODEL_PROVIDER",
        "ANSWER_OPENAI_API_BASE",
        "ANSWER_OPENAI_API_KEY",
        "ANSWER_BASE_MODEL",
        "ANSWER_SFT_MODEL",
        "ANSWER_TEMPERATURE",
        "ANSWER_MAX_TOKENS",
        "ANSWER_TIMEOUT_SEC",
        "OPENAI_API_BASE",
        "OPENAI_API_KEY",
        "LLM_DEFAULT_MODEL",
        "MODEL",
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


def test_base_provider_fallbacks() -> None:
    with patched_env(
        {
            "OPENAI_API_BASE": "http://127.0.0.1:8000/v1",
            "LLM_DEFAULT_MODEL": "Qwen/Qwen2.5-3B-Instruct",
        }
    ):
        settings = AnswerModelSettings.from_env()
        assert settings.provider == "base"
        assert settings.model_name == "Qwen/Qwen2.5-3B-Instruct"
        assert settings.base_url == "http://127.0.0.1:8000/v1"
        assert settings.issues() == []


def test_sft_provider_uses_lora_alias() -> None:
    with patched_env(
        {
            "ANSWER_MODEL_PROVIDER": "sft",
            "ANSWER_OPENAI_API_BASE": "http://127.0.0.1:8000/v1",
            "ANSWER_BASE_MODEL": "Qwen/Qwen2.5-3B-Instruct",
            "ANSWER_SFT_MODEL": "kb-sft",
            "ANSWER_MAX_TOKENS": "512",
        }
    ):
        settings = AnswerModelSettings.from_env()
        assert settings.provider == "sft"
        assert settings.model_name == "kb-sft"
        assert settings.max_tokens == 512
        assert settings.issues() == []


def test_invalid_provider_reports_issue() -> None:
    with patched_env(
        {
            "ANSWER_MODEL_PROVIDER": "shadow",
            "ANSWER_OPENAI_API_BASE": "http://127.0.0.1:8000/v1",
            "ANSWER_BASE_MODEL": "Qwen/Qwen2.5-3B-Instruct",
        }
    ):
        settings = AnswerModelSettings.from_env()
        assert settings.issues() == ["ANSWER_MODEL_PROVIDER must be base or sft"]


def main() -> None:
    test_base_provider_fallbacks()
    test_sft_provider_uses_lora_alias()
    test_invalid_provider_reports_issue()
    print("[ok] stage5 answer model config")


if __name__ == "__main__":
    main()
