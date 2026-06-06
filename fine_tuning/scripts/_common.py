"""fine_tuning 阶段一公共工具。

这个文件只放多个脚本都会用到的轻量能力：
- 读取配置，并从 `knowledge/.env` 兜底加载 Milvus 等环境变量；
- 读写 JSONL / JSON；
- 做引用编号、数字、拒答话术的规则识别；
- 封装可选的 OpenAI-compatible LLM 调用。

注意：这里不放业务流程，避免后续训练、评估脚本把公共层变成“大杂烩”。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# 脚本约定从仓库根目录运行，但这里仍然用文件路径反推根目录，
# 这样在 PyCharm / conda run / CI 里执行时不依赖当前 shell 的 cwd。
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = "fine_tuning/configs/config.yaml"

# 训练样本中要求答案引用上下文片段，例如 [C1]、[C2]。
CITE_RE = re.compile(r"\[(C\d+)\]")

# 用于拒答样本的启发式检查：拒答答案不应凭空编造上下文里没有的数字。
NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# 拒答判断是规则兜底，不代替人工抽检。正式训练前仍需人工看一批样本。
REFUSAL_CUES = [
    "无法回答",
    "无法据此",
    "无法直接",
    "无法确定",
    "无法给出",
    "资料不足",
    "没有关于",
    "没有相关",
    "未提及",
    "未涵盖",
    "建议补充",
    "存在冲突",
    "互相矛盾",
    "不一致",
    "缺少",
]


def load_project_env() -> None:
    """尽量加载主项目的 `knowledge/.env`。

    fine_tuning 阶段一要复用主项目 Milvus 地址和 collection 名称，所以这里做
    “配置文件优先，环境变量兜底”的设计。没有安装 python-dotenv 时直接跳过，
    不影响显式填写 config.yaml 的场景。
    """
    env_path = REPO_ROOT / "knowledge" / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path)


def citations_in(text: str) -> set[str]:
    """提取答案中使用过的引用编号。"""
    return set(CITE_RE.findall(text or ""))


def numbers_in(text: str) -> set[str]:
    """提取文本中的数字，用于粗略发现拒答样本里的参数幻觉。"""
    return set(NUM_RE.findall(text or ""))


def is_refusal(text: str) -> bool:
    """判断一段答案是否具备拒答/冲突提示倾向。"""
    return any(cue in (text or "") for cue in REFUSAL_CUES)


def fingerprint(question: str, contexts: Iterable[Dict[str, Any]]) -> str:
    """根据问题和上下文生成指纹，用于去重和 train/holdout 泄漏检查。"""
    context_text = "||".join(sorted((c.get("text", "") or "") for c in contexts))
    raw = f"{question or ''}||{context_text}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """加载 YAML 配置，并补齐阶段一运行所需默认值。"""
    load_project_env()
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Missing dependency: pyyaml. Install it with `pip install pyyaml`.") from exc

    config_path = REPO_ROOT / path
    if not config_path.exists():
        example_path = REPO_ROOT / "fine_tuning" / "configs" / "config.example.yaml"
        if example_path.exists():
            # 本地首次运行时通常还没有 config.yaml。这里允许用 example + env fallback
            # 跑通只读/离线流程，避免刚开始就卡在配置复制上。
            print(f"[config] {path} not found, using config.example.yaml with environment fallbacks.")
            config_path = example_path
        else:
            raise SystemExit(f"Config not found: {path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    return apply_env_defaults(cfg)


def apply_env_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """给配置补默认值。

    这里的默认值服务于“阶段一可跑通”，不是生产配置中心。真实接口、密钥、
    私有 endpoint 仍应写在被 Git 忽略的 config.yaml 或 knowledge/.env 里。
    """
    milvus = cfg.setdefault("milvus", {})
    milvus["uri"] = milvus.get("uri") or os.getenv("MILVUS_URL", "")
    milvus["collection"] = milvus.get("collection") or os.getenv("CHUNKS_COLLECTION", "kb_chunks")
    milvus.setdefault("db_name", "default")
    milvus.setdefault("token", "")
    milvus.setdefault("expr", "")
    milvus.setdefault("batch_size", 1000)
    milvus.setdefault("timeout_sec", 20)

    dataset = cfg.setdefault("dataset", {})
    dataset.setdefault("out_dir", "fine_tuning/data")
    dataset.setdefault("total", 120)
    dataset.setdefault("holdout_ratio", 0.15)
    dataset.setdefault("seed", 42)
    dataset.setdefault("max_chars", 3000)
    dataset.setdefault(
        "ratios",
        {
            "answerable": 0.35,
            "multi_chunk": 0.25,
            "unanswerable": 0.25,
            "conflicting": 0.15,
        },
    )

    cfg.setdefault("llm", {})
    cfg.setdefault("messages", {})
    return cfg


def repo_path(path: str) -> Path:
    """把仓库相对路径转换为绝对路径。"""
    return REPO_ROOT / path


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """读取 JSONL。文件不存在时返回空列表，便于脚本给出业务化错误。"""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    """写 JSONL，并自动创建父目录。"""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """写普通 JSON 文件，用于保存统计信息。"""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def get_llm(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """创建可选的 OpenAI-compatible LLM 客户端。

    没有配置 base_url / api_key 时返回 None，调用方进入 stub 模式。stub 模式
    只用于验证流水线，不用于正式训练数据。
    """
    llm_cfg = cfg.get("llm", {}) or {}
    if not llm_cfg.get("base_url") or not llm_cfg.get("api_key"):
        return None
    try:
        from openai import OpenAI
    except ImportError:
        print("[warn] openai is not installed, falling back to stub mode.")
        return None
    return {
        "client": OpenAI(base_url=llm_cfg["base_url"], api_key=llm_cfg["api_key"]),
        "model": llm_cfg.get("model", "qwen-plus"),
        "temperature": llm_cfg.get("temperature", 0.7),
        "max_tokens": llm_cfg.get("max_tokens", 512),
    }


def llm_json(llm: Dict[str, Any], system: str, user: str) -> Optional[Dict[str, Any]]:
    """调用强模型生成 JSON。

    数据构造模型可能返回 Markdown 代码块或夹杂解释文字，所以这里做一层宽松
    JSON 提取。失败返回 None，让上层跳过该样本，避免坏样本进入训练集。
    """
    try:
        resp = llm["client"].chat.completions.create(
            model=llm["model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=llm["temperature"],
            max_tokens=llm["max_tokens"],
        )
        text = (resp.choices[0].message.content or "").strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.S)
            return json.loads(match.group(0)) if match else None
    except Exception as exc:
        print(f"[warn] LLM generation failed: {exc}")
        return None
