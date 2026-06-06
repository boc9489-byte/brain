#!/usr/bin/env python3
"""阶段二 messages 数据集校验脚本。

阶段一 validate_dataset.py 校验 question + contexts + answer 中间格式。
阶段二 expand_dataset.py 直接输出 messages 格式，因此单独提供校验脚本，避免两种
schema 混在一起导致规则变得难维护。

默认校验：
    fine_tuning/data/processed/sft_train.jsonl
    fine_tuning/data/processed/sft_holdout.jsonl
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

import _common as C


TYPES = {"faithful", "multi_hop", "cite", "refuse", "format"}
GROUNDED_TYPES = {"faithful", "multi_hop", "cite", "format"}
REFUSE_SUBTYPES = {"no_recall", "weak_recall", "conflict"}
FORMAT_CUES = ["1.", "1、", "步骤", "要点", "排查", "定义", "如下"]


def message_roles(sample: Dict[str, Any]) -> List[str]:
    """取 messages 的 role 序列。"""
    return [message.get("role", "") for message in sample.get("messages", [])]


def sample_id(sample: Dict[str, Any]) -> str:
    """取样本 id，没有则返回问号。"""
    return str((sample.get("meta") or {}).get("id") or "?")


def get_user_and_answer(sample: Dict[str, Any]) -> Tuple[str, str]:
    """取 user 和 assistant 内容。"""
    messages = sample.get("messages") or []
    user = messages[1].get("content", "") if len(messages) > 1 else ""
    answer = messages[2].get("content", "") if len(messages) > 2 else ""
    return user, answer


def available_citations(user: str) -> set[str]:
    """从 user 的【检索资料】中提取可用资料编号。"""
    return set(re.findall(r"^\[(\d+)\]", user or "", flags=re.M))


def rendered_len(sample: Dict[str, Any]) -> int:
    """粗略计算样本字符长度。"""
    return sum(len(message.get("content", "") or "") for message in sample.get("messages", []))


def fingerprint(sample: Dict[str, Any]) -> str:
    """按 user 内容去重，防止 train / holdout 泄漏。"""
    user, _answer = get_user_and_answer(sample)
    return C.fingerprint(user, [{"text": user}])


def validate_sample(
    sample: Dict[str, Any],
    split: str,
    cfg: Dict[str, Any],
    errors: List[str],
    warns: List[str],
    seen: Dict[str, str],
) -> None:
    """校验单条 messages 样本。"""
    sid = sample_id(sample)
    roles = message_roles(sample)
    if roles != ["system", "user", "assistant"]:
        errors.append(f"[{split}] {sid} invalid roles: {roles}")
        return

    meta = sample.get("meta") or {}
    sample_type = meta.get("type")
    subtype = meta.get("subtype")
    user, answer = get_user_and_answer(sample)
    available = available_citations(user)
    used = C.num_citations_in(answer)

    if sample_type not in TYPES:
        errors.append(f"[{split}] {sid} invalid type={sample_type}")
    if "【检索资料】" not in user or "【问题】" not in user:
        errors.append(f"[{split}] {sid} user content missing required sections")
    if not answer.strip():
        errors.append(f"[{split}] {sid} empty assistant answer")
    if not meta.get("battle_capabilities"):
        errors.append(f"[{split}] {sid} missing battle_capabilities")

    if sample_type in GROUNDED_TYPES:
        if C.is_refusal(answer):
            errors.append(f"[{split}] {sid} {sample_type} should not refuse")
        if not used:
            errors.append(f"[{split}] {sid} {sample_type} answer has no citation")
        if not used.issubset(available):
            errors.append(f"[{split}] {sid} citation out of range: {sorted(used - available)}")

    if sample_type == "refuse":
        if not C.is_refusal(answer):
            errors.append(f"[{split}] {sid} refuse sample did not refuse")
        if subtype not in REFUSE_SUBTYPES:
            errors.append(f"[{split}] {sid} invalid refuse subtype={subtype}")

    if sample_type == "format" and not any(cue in answer for cue in FORMAT_CUES):
        warns.append(f"[{split}] {sid} format answer may lack structure cues")

    fp = fingerprint(sample)
    if fp in seen:
        errors.append(f"[{split}] {sid} duplicates {seen[fp]}")
    else:
        seen[fp] = f"{split}:{sid}"

    max_chars = int(cfg["dataset"].get("max_chars", 3000))
    length = rendered_len(sample)
    if length > max_chars:
        warns.append(f"[{split}] {sid} rendered length {length} > {max_chars}")


def validate_split(
    samples: Iterable[Dict[str, Any]],
    split: str,
    cfg: Dict[str, Any],
    errors: List[str],
    warns: List[str],
    seen: Dict[str, str],
) -> Counter:
    """校验一个 split 并返回 type 分布。"""
    counts: Counter = Counter()
    for sample in samples:
        counts[(sample.get("meta") or {}).get("type")] += 1
        validate_sample(sample, split, cfg, errors, warns, seen)
    return counts


def render_report(
    train: List[Dict[str, Any]],
    holdout: List[Dict[str, Any]],
    counts: Counter,
    subtype_counts: Counter,
    errors: List[str],
    warns: List[str],
) -> str:
    """渲染 Markdown 校验报告。"""
    total = len(train) + len(holdout)
    lines = [
        "# 阶段二 messages 数据集校验报告",
        "",
        f"- train={len(train)} holdout={len(holdout)} total={total}",
        "- 类型分布: "
        + ", ".join(f"{sample_type}={counts.get(sample_type, 0)}" for sample_type in sorted(TYPES)),
        "- refuse 子类: "
        + ", ".join(f"{subtype}={subtype_counts.get(subtype, 0)}" for subtype in sorted(REFUSE_SUBTYPES)),
        f"- 硬错误: {len(errors)}",
        f"- 软告警: {len(warns)}",
        "",
    ]
    if errors:
        lines.append("## 硬错误")
        lines.extend(f"- {error}" for error in errors[:300])
        lines.append("")
    if warns:
        lines.append("## 软告警")
        lines.extend(f"- {warn}" for warn in warns[:300])
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=C.DEFAULT_CONFIG_PATH)
    parser.add_argument("--train", default=None)
    parser.add_argument("--holdout", default=None)
    args = parser.parse_args()

    cfg = C.load_config(args.config)
    processed_dir = C.repo_path(cfg["dataset"]["out_dir"]) / "processed"
    train_path = C.repo_path(args.train) if args.train else processed_dir / "sft_train.jsonl"
    holdout_path = C.repo_path(args.holdout) if args.holdout else processed_dir / "sft_holdout.jsonl"
    train = C.read_jsonl(train_path)
    holdout = C.read_jsonl(holdout_path)
    if not train and not holdout:
        raise SystemExit("No messages dataset found. Run expand_dataset.py first.")

    errors: List[str] = []
    warns: List[str] = []
    seen: Dict[str, str] = {}
    train_counts = validate_split(train, "train", cfg, errors, warns, seen)
    holdout_counts = validate_split(holdout, "holdout", cfg, errors, warns, seen)
    counts = train_counts + holdout_counts
    subtype_counts = Counter(
        (sample.get("meta") or {}).get("subtype")
        for sample in train + holdout
        if (sample.get("meta") or {}).get("type") == "refuse"
    )

    for sample_type in TYPES:
        if counts.get(sample_type, 0) == 0:
            warns.append(f"[coverage] missing type={sample_type}")
    for subtype in REFUSE_SUBTYPES:
        if subtype_counts.get(subtype, 0) == 0:
            warns.append(f"[coverage] missing refuse subtype={subtype}")

    report = render_report(train, holdout, counts, subtype_counts, errors, warns)
    report_path = processed_dir / "_messages_validation_report.md"
    report_path.write_text(report, encoding="utf-8")

    print("\n".join(report.splitlines()[:7]))
    print(f"[validate_messages] report -> {report_path}")
    if errors:
        raise SystemExit(f"[validate_messages] failed with {len(errors)} hard errors.")
    print("[validate_messages] passed hard checks.")


if __name__ == "__main__":
    main()

