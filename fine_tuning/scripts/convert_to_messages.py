#!/usr/bin/env python3
"""把阶段一样本转换为 SFT 训练常用的 messages JSONL 格式。

build_sft_dataset.py 产出的是便于校验的中间结构：
    question + contexts + answer

训练时需要转换成：
    system / user / assistant

这样 LLaMA-Factory / TRL 可以只对 assistant 段计算 loss，system 和 user
里的检索资料只作为条件输入，不让模型学习复述输入。
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import _common as C


DEFAULT_SYSTEM_PROMPT = (
    "你是掌柜智库的知识库问答助手。你只能依据【检索资料】回答问题；"
    "当资料不足、资料无关或资料互相冲突时，必须明确拒答并说明缺少什么信息；"
    "回答中的事实应使用 [C1]、[C2] 这样的引用编号标注来源。"
)

# 当前四类冷启动样本与作战方案五类能力的映射。
# 注意：format 能力需要后续新增独立样本生成器。
BATTLE_CAPABILITY_MAP = {
    "answerable": ["faithful", "cite"],
    "multi_chunk": ["multi_hop"],
    "unanswerable": ["refuse"],
    "conflicting": ["refuse"],
}


def render_contexts(contexts: List[Dict[str, Any]]) -> str:
    """把 contexts 渲染进 user 消息，保留 C 编号和来源。"""
    rendered = []
    for context in contexts:
        cid = context.get("cid", "")
        source = context.get("source", "未知来源")
        text = context.get("text", "")
        rendered.append(f"[{cid}] 来源：{source}\n{text}")
    return "\n\n".join(rendered).strip() or "无可用检索资料"


def to_messages(sample: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
    """单条样本转换。

    meta 会保留原始样本 id/type，并补充 battle_capabilities，方便后续评估时按
    新作战方案的五类能力分桶统计。
    """
    user_content = f"【检索资料】\n{render_contexts(sample.get('contexts', []))}\n\n【问题】\n{sample.get('question', '')}"
    sample_type = sample.get("type")
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": sample.get("answer", "")},
        ],
        "meta": {
            **(sample.get("meta") or {}),
            "id": sample.get("id"),
            "type": sample_type,
            "battle_capabilities": BATTLE_CAPABILITY_MAP.get(sample_type, []),
            "source_schema": "stage1_question_context_answer",
        },
    }


def convert_split(split: str, cfg: Dict[str, Any], system_prompt: str) -> int:
    """转换 train 或 holdout split。"""
    processed_dir = C.repo_path(cfg["dataset"]["out_dir"]) / "processed"
    source_path = processed_dir / f"{split}.jsonl"
    target_path = processed_dir / f"messages_{split}.jsonl"
    rows = C.read_jsonl(source_path)
    converted = [to_messages(row, system_prompt) for row in rows]
    C.write_jsonl(target_path, converted)
    print(f"[convert] {source_path} -> {target_path} rows={len(converted)}")
    return len(converted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=C.DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    cfg = C.load_config(args.config)
    system_prompt = cfg.get("messages", {}).get("system_prompt") or DEFAULT_SYSTEM_PROMPT
    train_count = convert_split("train", cfg, system_prompt)
    holdout_count = convert_split("holdout", cfg, system_prompt)
    C.write_json(
        C.repo_path(cfg["dataset"]["out_dir"]) / "processed" / "_messages_stats.json",
        {
            "messages_train": train_count,
            "messages_holdout": holdout_count,
            "system_prompt_chars": len(system_prompt),
            "battle_capability_map": BATTLE_CAPABILITY_MAP,
        },
    )


if __name__ == "__main__":
    main()
