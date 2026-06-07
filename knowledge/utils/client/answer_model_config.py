"""回答模型接入配置。

阶段五只在回答生成层做 Base / SFT 切换，配置全部来自环境变量。
默认 provider 是 base，只有显式设置 `ANSWER_MODEL_PROVIDER=sft` 才会走 LoRA alias。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


def _first_env(*keys: str, default: str = "") -> str:
    """按优先级读取第一个非空环境变量。"""
    for key in keys:
        value = os.getenv(key)
        if value is not None and value.strip():
            return value.strip()
    return default


def _float_env(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be a float, got: {value}") from exc


def _int_env(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got: {value}") from exc


@dataclass(frozen=True)
class AnswerModelSettings:
    """回答模型配置快照。

    `model_name` 是最终传给 OpenAI-compatible 服务的 model。
    - provider=base 时使用 base_model；
    - provider=sft 时使用 sft_model，也就是 vLLM `--lora-modules` 的 alias。
    """

    provider: str
    base_url: str
    api_key: str
    base_model: str
    sft_model: str
    temperature: float
    max_tokens: int
    timeout_sec: int

    @classmethod
    def from_env(cls) -> "AnswerModelSettings":
        provider = _first_env("ANSWER_MODEL_PROVIDER", default="base").lower()
        return cls(
            provider=provider,
            base_url=_first_env("ANSWER_OPENAI_API_BASE", "OPENAI_API_BASE"),
            api_key=_first_env("ANSWER_OPENAI_API_KEY", "OPENAI_API_KEY", default="EMPTY"),
            base_model=_first_env("ANSWER_BASE_MODEL", "LLM_DEFAULT_MODEL", "MODEL"),
            sft_model=_first_env("ANSWER_SFT_MODEL", default="kb-sft"),
            temperature=_float_env("ANSWER_TEMPERATURE", 0.0),
            max_tokens=_int_env("ANSWER_MAX_TOKENS", 1024),
            timeout_sec=_int_env("ANSWER_TIMEOUT_SEC", 60),
        )

    @property
    def model_name(self) -> str:
        """当前 provider 对应的实际模型名。"""
        if self.provider == "sft":
            return self.sft_model
        return self.base_model

    def issues(self) -> List[str]:
        """返回配置问题列表，供检查脚本和客户端初始化复用。"""
        problems: List[str] = []
        if self.provider not in {"base", "sft"}:
            problems.append("ANSWER_MODEL_PROVIDER must be base or sft")
        if not self.base_url:
            problems.append("missing ANSWER_OPENAI_API_BASE or OPENAI_API_BASE")
        if not self.model_name:
            problems.append("missing active answer model name")
        if self.provider == "sft" and not self.sft_model:
            problems.append("missing ANSWER_SFT_MODEL")
        if self.timeout_sec <= 0:
            problems.append("ANSWER_TIMEOUT_SEC must be greater than 0")
        if self.max_tokens <= 0:
            problems.append("ANSWER_MAX_TOKENS must be greater than 0")
        return problems

    def cache_key(self, response_format: bool) -> Tuple[str, str, str, bool]:
        """客户端缓存键，避免 base/sft 切换时复用错客户端。"""
        return (self.provider, self.base_url, self.model_name, response_format)

    def safe_dict(self) -> Dict[str, object]:
        """脱敏后的配置，允许写入日志或检查脚本输出。"""
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model_name": self.model_name,
            "base_model": self.base_model,
            "sft_model": self.sft_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_sec": self.timeout_sec,
            "api_key_set": bool(self.api_key),
        }
