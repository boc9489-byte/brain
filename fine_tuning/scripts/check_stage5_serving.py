#!/usr/bin/env python3
"""阶段五回答模型接入检查脚本。

默认只做本地配置检查，不访问模型服务。
加 `--health` 后才会请求 OpenAI-compatible `/models`，用于确认 vLLM LoRA alias 是否可见。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def load_project_env() -> None:
    """加载 knowledge/.env；没有 python-dotenv 时使用轻量解析兜底。"""
    env_path = REPO_ROOT / "knowledge" / ".env"
    if not env_path.exists():
        return
    if load_dotenv:
        load_dotenv(env_path)
        return

    # GPU 服务器可能只装了最小依赖，这里保留 dotenv 的轻量兜底解析。
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_project_env()

from knowledge.utils.client.answer_model_config import AnswerModelSettings  # noqa: E402


def _models_url(base_url: str) -> str:
    """根据 OpenAI-compatible base_url 拼出 /models 地址。"""
    return base_url.rstrip("/") + "/models"


def _auth_headers(api_key: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def check_models(settings: AnswerModelSettings) -> List[str]:
    """请求 /models，并确认当前模型名是否存在。"""
    url = _models_url(settings.base_url)
    req = urllib.request.Request(url, headers=_auth_headers(settings.api_key), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=settings.timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SystemExit(f"[stage5-health] request failed: {url}; reason={exc}") from exc

    model_ids = [item.get("id") for item in payload.get("data", []) if isinstance(item, dict)]
    print(f"[stage5-health] models={model_ids}")
    if settings.model_name not in model_ids:
        return [f"active model {settings.model_name!r} not found in /models"]
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true", help="只检查本地配置，不访问模型服务。")
    parser.add_argument("--health", action="store_true", help="请求 /models 检查 vLLM 服务和 LoRA alias。")
    args = parser.parse_args()

    settings = AnswerModelSettings.from_env()
    print("[stage5] answer model settings:")
    print(json.dumps(settings.safe_dict(), ensure_ascii=False, indent=2))

    issues = settings.issues()
    if issues:
        for issue in issues:
            print(f"[stage5][error] {issue}")
        raise SystemExit(1)

    # 默认不访问网络，避免本地只想检查配置时被未启动的 vLLM 阻断。
    if args.health:
        issues.extend(check_models(settings))
        if issues:
            for issue in issues:
                print(f"[stage5-health][error] {issue}")
            raise SystemExit(1)
        print("[stage5-health] passed")
        return

    print("[stage5] check passed")
    if not args.check_only:
        print("[stage5] add --health to verify the running vLLM OpenAI-compatible service.")


if __name__ == "__main__":
    main()
