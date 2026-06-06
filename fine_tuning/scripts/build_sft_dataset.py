#!/usr/bin/env python3
"""基于导出的知识库 chunk 构造阶段一 SFT 样本。

阶段一先用四类样本跑通数据闭环：
- answerable：单片段忠实回答；
- multi_chunk：多片段综合回答；
- unanswerable：资料不足拒答；
- conflicting：合成冲突拒答。

新作战方案中的五类能力映射为：
- faithful / cite 主要来自 answerable；
- multi_hop 来自 multi_chunk；
- refuse 来自 unanswerable + conflicting；
- format 会在阶段一增强版中单独补生成策略。
"""

from __future__ import annotations

import argparse
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import _common as C


SYS_GEN = "你是 SFT 数据构造助手。严格只输出 JSON，不要解释。"


def source_of(chunk: Dict[str, Any]) -> str:
    """拼接样本引用来源，便于后续回答中标注依据。"""
    file_title = (chunk.get("file_title") or "").strip()
    title = (chunk.get("title") or chunk.get("parent_title") or "").strip()
    return (f"{file_title} / {title}".strip(" /")) or "未知来源"


def first_sentences(text: str, n: int = 2) -> str:
    """stub 模式下取前几句作为占位答案，正式训练前必须用强模型重建。"""
    parts = re.split(r"(?<=[。!?\n])", (text or "").strip())
    return "".join(parts[:n]).strip()


def perturb_number(text: str) -> Tuple[Optional[str], Optional[Tuple[str, str]]]:
    """修改文本里的一个数字，用来合成“资料冲突”样本。

    真实知识库通常自洽，冲突样本很少自然出现；为了训练模型学会“不硬选答案”，
    阶段一用合成冲突作为冷启动。
    """
    matches = [m for m in C.NUM_RE.finditer(text or "") if len(m.group(0)) <= 6]
    if not matches:
        return None, None
    candidates = [m for m in matches if len(m.group(0)) >= 2]
    match = random.choice(candidates or matches)
    original = match.group(0)
    value = float(original)
    changed = value * 2 if value < 10 else value + max(1, round(value * 0.5))
    replacement = str(int(changed)) if "." not in original else f"{changed:.1f}"
    return text[: match.start()] + replacement + text[match.end() :], (original, replacement)


def sample_record(
    sample_type: str,
    question: str,
    answer: str,
    contexts: List[Dict[str, Any]],
    source_chunk_ids: List[Any],
    item_name: str,
    synthetic: bool,
) -> Dict[str, Any]:
    """统一样本结构，避免四类构造函数各写一套字段。"""
    return {
        "type": sample_type,
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "meta": {
            "item_name": item_name or "",
            "source_chunk_ids": source_chunk_ids,
            "synthetic": synthetic,
        },
    }


def dedupe_samples(samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """按校验脚本同款指纹去重，避免 train / holdout 泄漏。

    阶段一的 stub 模式会从真实 chunk 中抽取标题和正文生成占位样本。如果同一份
    文档被重复导入，或者不同 chunk 的正文完全一致，就可能出现问题和上下文相同
    的样本。这里在切分数据集前去重，保留第一条，丢弃后续重复项。
    """
    seen = set()
    unique: List[Dict[str, Any]] = []
    dropped = 0
    for sample in samples:
        fp = C.fingerprint(sample.get("question", ""), sample.get("contexts", []))
        if fp in seen:
            dropped += 1
            continue
        seen.add(fp)
        unique.append(sample)
    return unique, dropped


def prompt_answerable(context: str) -> str:
    """强模型构造单片段可回答样本的提示词。"""
    return (
        f"参考资料:\n{context}\n\n"
        "基于且仅基于上述资料，生成一个用户会问的问题，以及一个忠实、在相关句子末尾用 [C1] 标注依据的中文答案。"
        '只输出 JSON: {"question":"...","answer":"..."}'
    )


def prompt_multi(context: str, cids: List[str]) -> str:
    """强模型构造多片段综合样本的提示词。"""
    return (
        f"参考资料:\n{context}\n\n"
        f"生成一个必须综合 {','.join(cids)} 才能回答的问题，以及整合后的中文答案。"
        "答案中的事实分别标注对应 [C编号]。"
        '只输出 JSON: {"question":"...","answer":"..."}'
    )


def prompt_unanswerable(context: str, topic: str) -> str:
    """强模型构造资料不足拒答样本的提示词。"""
    return (
        f"参考资料:\n{context}\n\n"
        f"围绕「{topic}」生成一个合理问题。由于参考资料无法回答它，答案必须明确说明资料不足、无法回答，并指出缺少什么。"
        '只输出 JSON: {"question":"...","answer":"..."}'
    )


def build_answerable(chunks: List[Dict[str, Any]], llm: Optional[Dict[str, Any]], target: int) -> List[Dict[str, Any]]:
    """构造 answerable 样本。

    有 LLM 时生成更自然的问题和答案；无 LLM 时进入 stub 模式，只验证流水线。
    """
    out = []
    pool = [c for c in chunks if len(c.get("content") or "") > 30]
    random.shuffle(pool)
    for chunk in pool:
        if len(out) >= target:
            break
        ctxs = [{"cid": "C1", "source": source_of(chunk), "text": chunk["content"]}]
        if llm:
            qa = C.llm_json(llm, SYS_GEN, prompt_answerable(f"[C1] {ctxs[0]['source']}: {ctxs[0]['text']}"))
            if not qa:
                continue
            question, answer = qa.get("question"), qa.get("answer")
            # 训练忠实回答时，答案必须真的带上 C1 引用，否则直接丢弃。
            if not question or not answer or "C1" not in C.citations_in(answer):
                continue
        else:
            question = f"关于{chunk.get('item_name') or chunk.get('file_title')}，{chunk.get('title') or '这部分内容'}是怎样的？"
            answer = f"根据资料，{first_sentences(chunk['content'])} [C1]。"
        out.append(
            sample_record(
                "answerable",
                question,
                answer,
                ctxs,
                [chunk.get("chunk_id")],
                chunk.get("item_name") or "",
                not bool(llm),
            )
        )
    return out


def build_multi(by_item: Dict[str, List[Dict[str, Any]]], llm: Optional[Dict[str, Any]], target: int) -> List[Dict[str, Any]]:
    """构造 multi_chunk 样本，训练模型整合 2-3 个片段的能力。"""
    out = []
    # 同一商品至少需要两个有效 chunk，才有构造多片段综合问题的意义。
    items = [item for item, chunks in by_item.items() if len([c for c in chunks if len(c.get("content") or "") > 30]) >= 2]
    random.shuffle(items)
    for item in items:
        if len(out) >= target:
            break
        pool = [c for c in by_item[item] if len(c.get("content") or "") > 30]
        random.shuffle(pool)
        picked = pool[: random.choice([2, 3])]
        ctxs = [{"cid": f"C{i + 1}", "source": source_of(c), "text": c["content"]} for i, c in enumerate(picked)]
        if llm:
            context = "\n".join(f"[{x['cid']}] {x['source']}: {x['text']}" for x in ctxs)
            qa = C.llm_json(llm, SYS_GEN, prompt_multi(context, [x["cid"] for x in ctxs]))
            if not qa:
                continue
            question, answer = qa.get("question"), qa.get("answer")
            if not question or not answer or not C.citations_in(answer):
                continue
        else:
            question = f"请综合说明{item}的相关信息。"
            answer = " ".join(f"{first_sentences(c['content'], 1)} [{ctxs[i]['cid']}]。" for i, c in enumerate(picked))
        out.append(
            sample_record(
                "multi_chunk",
                question,
                answer,
                ctxs,
                [c.get("chunk_id") for c in picked],
                item,
                not bool(llm),
            )
        )
    return out


def build_unanswerable(by_item: Dict[str, List[Dict[str, Any]]], llm: Optional[Dict[str, Any]], target: int) -> List[Dict[str, Any]]:
    """构造资料不足拒答样本。

    做法是“问 A 商品的问题，给 B 商品的上下文”，模拟无关召回。
    如果知识库里只有一个 item，这类样本无法构造。
    """
    out = []
    items = list(by_item.keys())
    if len(items) < 2:
        return out
    random.shuffle(items)
    for item in items:
        if len(out) >= target:
            break
        other_item = random.choice([x for x in items if x != item])
        candidates = [c for c in by_item[other_item] if len(c.get("content") or "") > 30] or by_item[other_item]
        chunk = random.choice(candidates)
        ctxs = [{"cid": "C1", "source": source_of(chunk), "text": chunk["content"]}]
        if llm:
            qa = C.llm_json(llm, SYS_GEN, prompt_unanswerable(f"[C1] {ctxs[0]['source']}: {ctxs[0]['text']}", item))
            if not qa:
                continue
            question, answer = qa.get("question"), qa.get("answer")
            if not question or not answer or not C.is_refusal(answer):
                continue
        else:
            question = f"{item}的保修期限是多久？"
            answer = f"现有资料只包含与「{other_item}」相关的内容，没有关于「{item}」保修期限的信息，我无法据此回答。建议补充对应资料后再确认。"
        out.append(
            sample_record(
                "unanswerable",
                question,
                answer,
                ctxs,
                [chunk.get("chunk_id")],
                item,
                not bool(llm),
            )
        )
    return out


def build_conflicting(chunks: List[Dict[str, Any]], target: int) -> List[Dict[str, Any]]:
    """构造冲突拒答样本。

    通过篡改数字制造 C1/C2 冲突，让模型学习“资料冲突时拒答并提示确认”。
    """
    out = []
    pool = [c for c in chunks if C.NUM_RE.search(c.get("content") or "")]
    random.shuffle(pool)
    for chunk in pool:
        if len(out) >= target:
            break
        perturbed, pair = perturb_number(chunk["content"])
        if not perturbed or not pair or pair[0] == pair[1]:
            continue
        ctxs = [
            {"cid": "C1", "source": source_of(chunk), "text": chunk["content"]},
            {"cid": "C2", "source": source_of(chunk) + "（合成冲突版）", "text": perturbed},
        ]
        question = f"关于{chunk.get('item_name') or chunk.get('file_title')}，{chunk.get('title') or '相关参数'}的数值是多少？"
        answer = f"现有资料存在冲突：一处为 {pair[0]} [C1]，另一处为 {pair[1]} [C2]，数值不一致，无法直接给出确定答案，建议以最新生效资料为准。"
        out.append(
            sample_record(
                "conflicting",
                question,
                answer,
                ctxs,
                [chunk.get("chunk_id")],
                chunk.get("item_name") or "",
                True,
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=C.DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Force offline stub mode.")
    args = parser.parse_args()

    cfg = C.load_config(args.config)
    dataset_cfg = cfg["dataset"]
    random.seed(dataset_cfg.get("seed", 42))

    raw_path = C.repo_path(dataset_cfg["out_dir"]) / "raw" / "kb_chunks.jsonl"
    chunks = C.read_jsonl(raw_path)
    if not chunks:
        raise SystemExit("No chunks found. Run export_kb_chunks.py first.")

    by_item: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in chunks:
        # 优先按 item_name 分组；没有商品名时退回 file_title，保证导入早期也能跑。
        key = chunk.get("item_name") or chunk.get("file_title") or "UNKNOWN"
        by_item.setdefault(key, []).append(chunk)

    llm = None if args.dry_run else C.get_llm(cfg)
    print(f"[build] mode={'llm' if llm else 'stub'} chunks={len(chunks)} items={len(by_item)}")

    total = int(dataset_cfg.get("total", 120))
    ratios = dataset_cfg["ratios"]
    # 第一阶段仍按四类冷启动样本分配数量；五类作战能力在文档中做映射，
    # 后续再把 format 独立拆成第五类生成器。
    targets = {
        "answerable": round(total * ratios["answerable"]),
        "multi_chunk": round(total * ratios["multi_chunk"]),
        "unanswerable": round(total * ratios["unanswerable"]),
        "conflicting": round(total * ratios["conflicting"]),
    }

    samples: List[Dict[str, Any]] = []
    samples.extend(build_answerable(chunks, llm, targets["answerable"]))
    samples.extend(build_multi(by_item, llm, targets["multi_chunk"]))
    samples.extend(build_unanswerable(by_item, llm, targets["unanswerable"]))
    samples.extend(build_conflicting(chunks, targets["conflicting"]))
    random.shuffle(samples)
    samples, dropped_duplicates = dedupe_samples(samples)

    for idx, sample in enumerate(samples, 1):
        sample["id"] = f"kb-{idx:06d}"

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        by_type.setdefault(sample["type"], []).append(sample)

    holdout_ratio = float(dataset_cfg.get("holdout_ratio", 0.15))
    train: List[Dict[str, Any]] = []
    holdout: List[Dict[str, Any]] = []
    for typed_samples in by_type.values():
        # 按 type 分层切分，避免 holdout 只落在某一类样本上。
        split_at = max(1, round(len(typed_samples) * holdout_ratio)) if typed_samples else 0
        holdout.extend(typed_samples[:split_at])
        train.extend(typed_samples[split_at:])

    processed_dir = C.repo_path(dataset_cfg["out_dir"]) / "processed"
    C.write_jsonl(processed_dir / "train.jsonl", train)
    C.write_jsonl(processed_dir / "holdout.jsonl", holdout)

    stats = {
        "requested_total": total,
        "actual_total": len(samples),
        "train": len(train),
        "holdout": len(holdout),
        "by_type": {sample_type: len(rows) for sample_type, rows in by_type.items()},
        "dropped_duplicates": dropped_duplicates,
        "synthetic": sum(1 for sample in samples if sample["meta"]["synthetic"]),
        "mode": "llm" if llm else "stub",
    }
    C.write_json(processed_dir / "_build_stats.json", stats)
    print(f"[build] {stats}")
    if not llm:
        # stub 数据只验证工程通路，不能作为正式 SFT 训练集。
        print("[build] stub data is only for pipeline verification. Use a strong model before training.")


if __name__ == "__main__":
    main()
