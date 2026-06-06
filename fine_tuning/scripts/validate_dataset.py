#!/usr/bin/env python3
"""训练前数据校验脚本。

校验分两类：
- 硬错误：必须修复，否则退出码非 0；
- 软告警：不阻塞，但需要在数据报告里说明。

阶段一重点防止三类坏数据进入训练：
1. 引用编号越界；
2. 应拒答样本没有拒答；
3. train / holdout 重复导致评估泄漏。
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import _common as C


REQUIRED = ["id", "type", "question", "answer", "contexts"]
TYPES = {"answerable", "multi_chunk", "unanswerable", "conflicting"}
CONFLICT_CUES = ["冲突", "不一致", "矛盾"]


def rendered_len(sample: Dict[str, Any]) -> int:
    """粗略估算样本渲染后的长度，提前发现超过 max_seq_len 的风险。"""
    contexts = "".join((c.get("text", "") or "") for c in sample.get("contexts", []))
    return len((sample.get("question", "") or "") + (sample.get("answer", "") or "") + contexts)


def validate_split(
    samples: List[Dict[str, Any]],
    split: str,
    cfg: Dict[str, Any],
    errors: List[str],
    warns: List[str],
    seen: Dict[str, str],
) -> Dict[str, int]:
    """校验单个 split，并把错误/告警追加到调用方列表。"""
    max_chars = cfg["dataset"].get("max_chars", 3000)
    counts = {sample_type: 0 for sample_type in TYPES}

    for sample in samples:
        sample_id = sample.get("id", "?")
        for key in REQUIRED:
            if key not in sample:
                errors.append(f"[{split}] {sample_id} missing field: {key}")

        sample_type = sample.get("type")
        counts[sample_type] = counts.get(sample_type, 0) + 1
        if sample_type not in TYPES:
            errors.append(f"[{split}] {sample_id} invalid type={sample_type}")

        context_ids = {c.get("cid") for c in sample.get("contexts", [])}
        used_citations = C.citations_in(sample.get("answer", ""))
        # 答案引用了不存在的 C 编号，会直接破坏引用可信度，属于硬错误。
        if not used_citations.issubset(context_ids):
            errors.append(f"[{split}] {sample_id} citation out of range: {sorted(used_citations - context_ids)}")

        answer = sample.get("answer", "")
        if sample_type in ("answerable", "multi_chunk"):
            # 可回答样本必须有上下文、有引用、不能变成拒答。
            if not sample.get("contexts"):
                errors.append(f"[{split}] {sample_id} {sample_type} has no contexts")
            if not used_citations:
                errors.append(f"[{split}] {sample_id} {sample_type} answer has no citation")
            if C.is_refusal(answer):
                errors.append(f"[{split}] {sample_id} {sample_type} should not refuse")

        if sample_type in ("unanswerable", "conflicting") and not C.is_refusal(answer):
            # 拒答类样本如果没有拒答，会把模型训练成“资料不足也硬答”。
            errors.append(f"[{split}] {sample_id} {sample_type} should refuse")

        if sample_type == "conflicting":
            if not any(cue in answer for cue in CONFLICT_CUES):
                errors.append(f"[{split}] {sample_id} conflicting answer does not mention conflict")
            if len(used_citations) < 2:
                warns.append(f"[{split}] {sample_id} conflicting answer has fewer than 2 citations")

        if sample_type == "unanswerable":
            # 这里只做启发式数字检查，不等价于完整事实校验。
            context_numbers = set()
            for context in sample.get("contexts", []):
                context_numbers |= C.numbers_in(context.get("text", ""))
                context_numbers |= C.numbers_in(context.get("source", ""))
            extra_numbers = C.numbers_in(answer) - context_numbers - C.numbers_in(sample.get("question", ""))
            if extra_numbers:
                warns.append(f"[{split}] {sample_id} refusal has suspicious extra numbers: {sorted(extra_numbers)}")

        fp = C.fingerprint(sample.get("question", ""), sample.get("contexts", []))
        # 同一问题+上下文不能同时出现在 train 和 holdout，否则评估会虚高。
        if fp in seen:
            errors.append(f"[{split}] {sample_id} duplicates {seen[fp]}")
        else:
            seen[fp] = f"{split}:{sample_id}"

        length = rendered_len(sample)
        if length > max_chars:
            warns.append(f"[{split}] {sample_id} rendered length {length} > {max_chars}")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=C.DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    cfg = C.load_config(args.config)
    processed_dir = C.repo_path(cfg["dataset"]["out_dir"]) / "processed"
    train = C.read_jsonl(processed_dir / "train.jsonl")
    holdout = C.read_jsonl(processed_dir / "holdout.jsonl")
    if not train and not holdout:
        raise SystemExit("No dataset found. Run build_sft_dataset.py first.")

    errors: List[str] = []
    warns: List[str] = []
    seen: Dict[str, str] = {}
    train_counts = validate_split(train, "train", cfg, errors, warns, seen)
    holdout_counts = validate_split(holdout, "holdout", cfg, errors, warns, seen)

    total = len(train) + len(holdout)
    merged = {sample_type: train_counts.get(sample_type, 0) + holdout_counts.get(sample_type, 0) for sample_type in TYPES}
    for sample_type in TYPES:
        target = cfg["dataset"]["ratios"].get(sample_type, 0)
        actual = merged[sample_type] / max(1, total)
        # 配比偏移只做软告警，因为小样本 dry-run 时很容易自然偏移。
        if abs(actual - target) > 0.10:
            warns.append(f"[ratio] {sample_type} actual {actual:.0%} differs from target {target:.0%} by > 10pt")

    lines = [
        "# 数据集校验报告",
        "",
        f"- train={len(train)} holdout={len(holdout)} total={total}",
        "- 类型分布: " + ", ".join(f"{t}={merged[t]}({merged[t] / max(1, total):.0%})" for t in sorted(TYPES)),
        f"- 硬错误: {len(errors)}",
        f"- 软告警: {len(warns)}",
        "",
    ]
    if errors:
        lines.append("## 硬错误")
        lines.extend(f"- {err}" for err in errors[:300])
        lines.append("")
    if warns:
        lines.append("## 软告警")
        lines.extend(f"- {warn}" for warn in warns[:300])
        lines.append("")

    report_path = processed_dir / "_validation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[:6]))
    print(f"[validate] report -> {report_path}")
    if errors:
        # 退出码非 0，方便后续挂到 CI 或本地 preflight。
        raise SystemExit(f"[validate] failed with {len(errors)} hard errors.")
    print("[validate] passed hard checks.")


if __name__ == "__main__":
    main()
