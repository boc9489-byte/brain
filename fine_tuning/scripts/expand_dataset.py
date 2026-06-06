#!/usr/bin/env python3
"""阶段二强模型造数脚本。

阶段一的 build_sft_dataset.py 产出中间格式，用于验证流水线。阶段二从这里开始
产出训练前更接近真实需求的 messages 格式：

    data/raw/kb_chunks.jsonl
      -> sft_train.jsonl
      -> sft_holdout.jsonl

运行方式：
    python fine_tuning/scripts/expand_dataset.py --retriever local --dry-run
    python fine_tuning/scripts/expand_dataset.py --retriever local
    python fine_tuning/scripts/expand_dataset.py --retriever milvus

说明：
    --dry-run 仍然只验证工程通路，不是正式训练数据；
    不加 --dry-run 且配置 llm.base_url / llm.api_key 后，才会调用强模型造数。
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import _common as C


DEFAULT_SYSTEM_PROMPT = (
    "你是掌柜智库的产品知识库问答助手。你只能依据【检索资料】回答问题；"
    "资料不足、资料无关或资料互相冲突时必须明确拒答，并说明缺少什么信息；"
    "回答中的事实应使用 [1]、[2] 这样的编号标注来源。"
)
SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT

GROUNDED_TYPES = {"faithful", "multi_hop", "cite", "format"}
ALL_TYPES = GROUNDED_TYPES | {"refuse"}
REFUSE_SUBTYPES = {"no_recall", "weak_recall", "conflict"}
BATTLE_CAPABILITY_MAP = {
    "faithful": ["faithful", "cite"],
    "multi_hop": ["multi_hop", "cite"],
    "cite": ["cite", "faithful"],
    "format": ["format", "faithful", "cite"],
    "refuse": ["refuse"],
}


def source_of(chunk: Dict[str, Any]) -> str:
    """生成可读来源名，优先保留文件和标题。"""
    file_title = (chunk.get("file_title") or chunk.get("source") or "").strip()
    title = (chunk.get("title") or chunk.get("parent_title") or "").strip()
    return (f"{file_title} / {title}".strip(" /")) or (chunk.get("item_name") or "未知来源")


def item_key(chunk: Dict[str, Any]) -> str:
    """用于多片段和拒答构造的分组键。"""
    return (chunk.get("item_name") or chunk.get("file_title") or "UNKNOWN").strip()


def clean_content(text: str) -> str:
    """清理 Markdown 图片、裸 URL 和多余空行，降低训练噪声。"""
    value = text or ""
    value = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", value)
    value = re.sub(r"https?://\S+", "", value)
    value = re.sub(r"[ \t]*\n{3,}[ \t]*", "\n\n", value)
    return value.strip()


def first_sentences(text: str, n: int = 1) -> str:
    """stub 模式使用的短答案片段。"""
    parts = re.split(r"(?<=[。!?\n])", (text or "").strip())
    return "".join(parts[:n]).strip()


def context_from_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """把 raw chunk 转成 messages 渲染上下文。"""
    return {
        "text": clean_content(chunk.get("content") or ""),
        "source": source_of(chunk),
        "chunk_id": chunk.get("chunk_id"),
        "item_name": chunk.get("item_name") or "",
    }


def render_contexts(contexts: Sequence[Dict[str, Any]]) -> str:
    """渲染 user 消息中的检索资料。"""
    lines = []
    for idx, context in enumerate(contexts, 1):
        source = context.get("source") or "未知来源"
        text = context.get("text") or ""
        lines.append(f"[{idx}] 来源：{source}\n{text}")
    return "\n\n".join(lines)


def user_message(contexts: Sequence[Dict[str, Any]], question: str) -> str:
    """组装 user 消息。"""
    return f"【检索资料】\n{render_contexts(contexts)}\n\n【问题】\n{question}"


def sample_fingerprint(sample: Dict[str, Any]) -> str:
    """messages 样本去重指纹。"""
    user = sample["messages"][1]["content"]
    return C.fingerprint(user, [{"text": user}])


def make_sample(
    sample_type: str,
    contexts: Sequence[Dict[str, Any]],
    question: str,
    answer: str,
    synthetic: bool,
    subtype: Optional[str] = None,
) -> Dict[str, Any]:
    """统一阶段二 messages 样本结构。"""
    source_chunk_ids = [c.get("chunk_id") for c in contexts if c.get("chunk_id") is not None]
    sources = [c.get("source") or "未知来源" for c in contexts]
    meta = {
        "type": sample_type,
        "synthetic": synthetic,
        "sources": sources,
        "source_chunk_ids": source_chunk_ids,
        "battle_capabilities": BATTLE_CAPABILITY_MAP.get(sample_type, []),
        "source_schema": "stage2_messages",
    }
    if subtype:
        meta["subtype"] = subtype
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(contexts, question)},
            {"role": "assistant", "content": answer},
        ],
        "meta": meta,
    }


def perturb_number(text: str) -> Tuple[Optional[str], Optional[Tuple[str, str]]]:
    """合成冲突样本：修改文本中的一个数字。"""
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


class LocalRetriever:
    """本地关键词召回，用于阶段二 dry-run 和无 Milvus 检索环境的造数自测。"""

    def __init__(self, chunks: Sequence[Dict[str, Any]]):
        self.chunks = list(chunks)

    def search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        query_chars = set(query or "")
        scored = sorted(
            self.chunks,
            key=lambda chunk: -len(query_chars & set(chunk.get("content") or "")),
        )
        return scored[:k]


class MilvusRetriever:
    """生产候选召回器：BGE-M3 query 编码 + Milvus dense_vector 检索。"""

    def __init__(self, cfg: Dict[str, Any]):
        milvus_cfg = cfg["milvus"]
        try:
            from pymilvus import MilvusClient
            from pymilvus.model.hybrid import BGEM3EmbeddingFunction
        except ImportError as exc:
            raise SystemExit("MilvusRetriever requires pymilvus with BGEM3EmbeddingFunction.") from exc

        client_kwargs = {"uri": milvus_cfg["uri"]}
        if milvus_cfg.get("token"):
            client_kwargs["token"] = milvus_cfg["token"]
        if milvus_cfg.get("db_name"):
            client_kwargs["db_name"] = milvus_cfg["db_name"]
        self.client = MilvusClient(**client_kwargs)
        self.collection = milvus_cfg["collection"]
        self.fields = milvus_cfg["fields"]
        self.vector_field = milvus_cfg.get("vector_field", "dense_vector")
        self.metric_type = milvus_cfg.get("metric_type", "COSINE")
        self.embedder = BGEM3EmbeddingFunction(
            model_name=milvus_cfg.get("embed_model", "BAAI/bge-m3"),
            device=milvus_cfg.get("embed_device", "cpu"),
            use_fp16=bool(milvus_cfg.get("embed_use_fp16", False)),
        )

    def search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        dense = self.embedder.encode_queries([query])["dense"][0].tolist()
        rows = self.client.search(
            collection_name=self.collection,
            data=[dense],
            anns_field=self.vector_field,
            search_params={"metric_type": self.metric_type},
            limit=k,
            output_fields=list(self.fields.values()),
        )
        out = []
        for hit in rows[0]:
            entity = hit.get("entity", hit)
            out.append({logical: entity.get(actual) for logical, actual in self.fields.items()})
        return out


def fewshot(seeds: Sequence[Dict[str, Any]], sample_type: str, n: int = 2) -> str:
    """从种子样本中取 few-shot 示例。"""
    examples = [s for s in seeds if (s.get("meta") or {}).get("type") == sample_type][:n]
    if not examples:
        return ""
    blocks = []
    for sample in examples:
        messages = sample.get("messages") or []
        if len(messages) < 3:
            continue
        blocks.append(f"<示例>\n{messages[1]['content']}\n答：{messages[2]['content']}\n</示例>")
    return "\n\n".join(blocks)


def gen_question(llm: Dict[str, Any], contexts: Sequence[Dict[str, Any]], sample_type: str, seeds: Sequence[Dict[str, Any]]) -> Optional[str]:
    """调用强模型生成问题。"""
    hints = {
        "faithful": "一个可由资料直接回答的问题",
        "multi_hop": "一个需要综合多个片段才能回答的问题",
        "cite": "一个答案必须标注来源的问题",
        "format": "一个答案适合用步骤、定义或故障排查格式组织的问题",
    }
    context_text = "\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(contexts))
    prompt = (
        f"{fewshot(seeds, sample_type)}\n\n"
        f"参考资料：\n{context_text}\n\n"
        f"请基于上述资料提出{hints[sample_type]}。只输出问题文本。"
    )
    text = C.llm_text(llm, "你是数据构造助手，只输出要求的内容。", prompt)
    return (text or "").strip().removeprefix("问题：").removeprefix("问题:").strip() or None


def gen_answer(
    llm: Dict[str, Any],
    contexts: Sequence[Dict[str, Any]],
    question: str,
    sample_type: str,
    seeds: Sequence[Dict[str, Any]],
) -> Optional[str]:
    """调用强模型生成答案。"""
    prompt = (
        f"{fewshot(seeds, sample_type)}\n\n"
        f"{user_message(contexts, question)}\n\n"
        "请严格依据 system 规则回答。只输出 assistant 答案，不要解释造数过程。"
    )
    return C.llm_text(llm, SYSTEM_PROMPT, prompt)


def format_subtype(text: str) -> str:
    """识别 format 子类。"""
    value = text or ""
    if re.search(r"常见原因|故障|排查|失败|异常|不能|无法", value):
        return "troubleshooting"
    if re.search(r"步骤|流程|如何|怎么|依次|请按", value) or re.search(r"(^|\n)\s*\d+[\.\、\)]", value):
        return "steps"
    return "definition"


def validate_inline(sample: Dict[str, Any]) -> Tuple[bool, str]:
    """生成即校验，避免明显坏样本进入 processed 文件。"""
    messages = sample.get("messages") or []
    if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
        return False, "messages roles invalid"
    sample_type = (sample.get("meta") or {}).get("type")
    if sample_type not in ALL_TYPES:
        return False, f"invalid type: {sample_type}"
    user = messages[1].get("content") or ""
    answer = messages[2].get("content") or ""
    if not answer.strip():
        return False, "empty answer"
    available = set(re.findall(r"^\[(\d+)\]", user, flags=re.M))
    used = C.num_citations_in(answer)
    refusal = C.is_refusal(answer)
    if sample_type in GROUNDED_TYPES:
        if refusal:
            return False, "grounded sample refused"
        if not used:
            return False, "grounded answer has no citation"
        if not used.issubset(available):
            return False, f"citation out of range: {sorted(used - available)}"
    if sample_type == "refuse" and not refusal:
        return False, "refuse sample did not refuse"
    return True, "ok"


def pick_related(by_item: Dict[str, List[Dict[str, Any]]], min_count: int = 2) -> Optional[List[Dict[str, Any]]]:
    """选出同 item 的多个 chunk。"""
    candidates = [chunks for chunks in by_item.values() if len(chunks) >= min_count]
    return random.choice(candidates) if candidates else None


def build_grounded(
    sample_type: str,
    chunks: Sequence[Dict[str, Any]],
    by_item: Dict[str, List[Dict[str, Any]]],
    retriever: Any,
    llm: Optional[Dict[str, Any]],
    seeds: Sequence[Dict[str, Any]],
    top_k: int,
) -> Optional[Dict[str, Any]]:
    """构造 faithful / multi_hop / cite / format 样本。"""
    pool = [c for c in chunks if len(clean_content(c.get("content") or "")) > 20]
    if not pool:
        return None
    if sample_type == "multi_hop":
        related = pick_related(by_item, 2)
        if not related:
            return None
        picked = random.sample(related, min(len(related), random.choice([2, 3])))
    else:
        picked = [random.choice(pool)]

    seed_contexts = [context_from_chunk(chunk) for chunk in picked]
    subtype = format_subtype(seed_contexts[0]["text"]) if sample_type == "format" else None

    if llm:
        question = gen_question(llm, seed_contexts, sample_type, seeds)
        if not question:
            return None
        retrieved = [context_from_chunk(chunk) for chunk in retriever.search(question, top_k)]
        seed_texts = {ctx["text"] for ctx in seed_contexts}
        if not any(ctx["text"] in seed_texts for ctx in retrieved):
            return build_refuse("weak_recall", picked, chunks, by_item, llm, seeds, question)
        answer = gen_answer(llm, retrieved, question, sample_type, seeds)
        if not answer:
            return None
        return make_sample(sample_type, retrieved, question, answer, synthetic=False, subtype=subtype)

    if sample_type == "multi_hop":
        question = "请综合说明这些资料中的关键要求。"
        answer = " ".join(
            f"{first_sentences(ctx['text'])} [{idx + 1}]。"
            for idx, ctx in enumerate(seed_contexts)
            if ctx["text"]
        )
    elif sample_type == "format":
        question = "这部分内容应该如何操作或理解？"
        answer = f"可按以下要点处理：\n1. {first_sentences(seed_contexts[0]['text'])} [1]。"
    elif sample_type == "cite":
        question = "资料中的关键依据是什么？"
        answer = f"资料中的关键依据是：{first_sentences(seed_contexts[0]['text'])} [1]。"
    else:
        question = "这条资料具体说明了什么？"
        answer = f"根据资料，{first_sentences(seed_contexts[0]['text'])} [1]。"
    return make_sample(sample_type, seed_contexts, question, answer, synthetic=True, subtype=subtype)


def build_refuse(
    subtype: str,
    seed_chunks: Optional[Sequence[Dict[str, Any]]],
    chunks: Sequence[Dict[str, Any]],
    by_item: Dict[str, List[Dict[str, Any]]],
    llm: Optional[Dict[str, Any]],
    seeds: Sequence[Dict[str, Any]],
    q_hint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """构造 no_recall / weak_recall / conflict 拒答样本。"""
    pool = [c for c in chunks if len(clean_content(c.get("content") or "")) > 20]
    if not pool:
        return None

    if subtype == "conflict":
        base = next((c for c in (seed_chunks or pool) if C.NUM_RE.search(clean_content(c.get("content") or ""))), None)
        base = base or random.choice(pool)
        base_ctx = context_from_chunk(base)
        perturbed, pair = perturb_number(base_ctx["text"])
        if not perturbed or not pair or pair[0] == pair[1]:
            return None
        contexts = [
            base_ctx,
            {**base_ctx, "text": perturbed, "source": f"{base_ctx['source']}（合成冲突版）"},
        ]
        question = q_hint or "这些资料中的数值到底以哪个为准？"
        answer = (
            f"检索到的资料存在冲突：片段 [1] 为 {pair[0]}，片段 [2] 为 {pair[1]}，"
            "两者不一致，无法直接给出确定答案。建议核对最新有效资料后再确认。"
        )
        if llm:
            generated = gen_answer(llm, contexts, question, "refuse", seeds)
            if generated and C.is_refusal(generated):
                answer = generated
        return make_sample("refuse", contexts, question, answer, synthetic=True, subtype="conflict")

    anchor = random.choice(pool)
    if subtype == "no_recall":
        other_items = [c for c in pool if item_key(c) != item_key(anchor)]
        context_chunk = random.choice(other_items or pool)
        question_topic = item_key(anchor)
    else:
        context_chunk = anchor
        question_topic = item_key(anchor)
    contexts = [context_from_chunk(context_chunk)]

    if llm:
        question = q_hint or C.llm_text(
            llm,
            "你是数据构造助手，只输出问题。",
            (
                f"围绕「{question_topic}」提出一个合理问题，但该问题不能被以下资料回答：\n"
                f"{contexts[0]['text']}\n\n只输出问题。"
            ),
        )
        question = question or "这个问题如何处理？"
        answer = gen_answer(llm, contexts, question, "refuse", seeds)
        if not answer or not C.is_refusal(answer):
            answer = "现有检索资料无法回答这个问题，缺少对应的产品说明或参数依据，建议补充相关资料后再确认。"
    else:
        question = "这款产品的保修期限和售后政策是什么？"
        answer = "现有检索资料没有提供该问题所需的保修期限或售后政策信息，无法据此回答。建议补充对应资料后再确认。"
    return make_sample("refuse", contexts, question, answer, synthetic=not bool(llm), subtype=subtype)


def split_by_type(samples: Sequence[Dict[str, Any]], holdout_ratio: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """按 type 分层切分 train / holdout。"""
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_type[sample["meta"]["type"]].append(sample)
    train: List[Dict[str, Any]] = []
    holdout: List[Dict[str, Any]] = []
    for typed_samples in by_type.values():
        random.shuffle(typed_samples)
        split_at = max(1, round(len(typed_samples) * holdout_ratio)) if typed_samples else 0
        holdout.extend(typed_samples[:split_at])
        train.extend(typed_samples[split_at:])
    return train, holdout


def attach_ids(samples: Iterable[Dict[str, Any]], prefix: str) -> None:
    """给样本补稳定 id。"""
    for idx, sample in enumerate(samples, 1):
        sample["meta"]["id"] = f"{prefix}-{idx:06d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=C.DEFAULT_CONFIG_PATH)
    parser.add_argument("--retriever", choices=["local", "milvus"], default=None)
    parser.add_argument("--dry-run", action="store_true", help="强制使用 stub，不调用 LLM。")
    parser.add_argument("--total", type=int, default=None, help="覆盖 expand.total，便于小样本验证。")
    args = parser.parse_args()

    cfg = C.load_config(args.config)
    expand_cfg = cfg.get("expand", {})
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = expand_cfg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
    random.seed(cfg.get("dataset", {}).get("seed", 42))

    raw_path = C.repo_path(cfg["dataset"]["out_dir"]) / "raw" / "kb_chunks.jsonl"
    chunks = C.read_jsonl(raw_path)
    if not chunks:
        raise SystemExit("No chunks found. Run export_kb_chunks.py first.")

    by_item: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        by_item[item_key(chunk)].append(chunk)

    seed_path = C.repo_path(cfg["dataset"]["out_dir"]) / "seed" / "seeds.jsonl"
    seeds = C.read_jsonl(seed_path)

    llm = None if args.dry_run else C.get_llm(cfg)
    retriever_name = args.retriever or expand_cfg.get("retriever", "local")
    retriever = MilvusRetriever(cfg) if retriever_name == "milvus" else LocalRetriever(chunks)
    total = int(args.total if args.total is not None else expand_cfg.get("total", 300))
    ratios = expand_cfg.get("ratios", {})
    top_k = int(expand_cfg.get("top_k", 4))
    holdout_ratio = float(expand_cfg.get("holdout_ratio", 0.12))
    refuse_subtypes = list(expand_cfg.get("refuse_subtypes", ["no_recall", "weak_recall", "conflict"]))

    print(
        f"[expand] retriever={retriever_name} mode={'llm' if llm else 'stub'} "
        f"chunks={len(chunks)} items={len(by_item)} seeds={len(seeds)} total={total}"
    )

    targets = {sample_type: round(total * float(ratio)) for sample_type, ratio in ratios.items()}
    samples: List[Dict[str, Any]] = []
    dropped = Counter()
    seen = set()

    def add(sample: Optional[Dict[str, Any]]) -> bool:
        if not sample:
            dropped["empty"] += 1
            return False
        ok, reason = validate_inline(sample)
        if not ok:
            dropped[reason] += 1
            return False
        fp = sample_fingerprint(sample)
        if fp in seen:
            dropped["duplicate"] += 1
            return False
        seen.add(fp)
        samples.append(sample)
        return True

    for sample_type in ("faithful", "multi_hop", "cite", "format"):
        need = int(targets.get(sample_type, 0))
        tries = 0
        cap = max(50, need * 8)
        while sum(1 for s in samples if s["meta"]["type"] == sample_type) < need and tries < cap:
            add(build_grounded(sample_type, chunks, by_item, retriever, llm, seeds, top_k))
            tries += 1

    refuse_need = int(targets.get("refuse", 0))
    subtype_targets: Dict[str, int] = {}
    if refuse_need:
        base = refuse_need // max(1, len(refuse_subtypes))
        remainder = refuse_need % max(1, len(refuse_subtypes))
        for idx, subtype in enumerate(refuse_subtypes):
            subtype_targets[subtype] = base + (1 if idx < remainder else 0)
    for subtype in refuse_subtypes:
        per_subtype = subtype_targets.get(subtype, 0)
        tries = 0
        cap = max(50, per_subtype * 10)
        while sum(1 for s in samples if s["meta"].get("subtype") == subtype) < per_subtype and tries < cap:
            add(build_refuse(subtype, None, chunks, by_item, llm, seeds))
            tries += 1

    random.shuffle(samples)
    train, holdout = split_by_type(samples, holdout_ratio)
    attach_ids(train, "sft-train")
    attach_ids(holdout, "sft-holdout")

    processed_dir = C.repo_path(cfg["dataset"]["out_dir"]) / "processed"
    C.write_jsonl(processed_dir / "sft_train.jsonl", train)
    C.write_jsonl(processed_dir / "sft_holdout.jsonl", holdout)

    by_type = Counter(sample["meta"]["type"] for sample in samples)
    by_subtype = Counter(sample["meta"].get("subtype") for sample in samples if sample["meta"]["type"] == "refuse")
    stats = {
        "requested_total": total,
        "actual_total": len(samples),
        "train": len(train),
        "holdout": len(holdout),
        "by_type": dict(by_type),
        "refuse_subtypes": dict(by_subtype),
        "dropped": dict(dropped),
        "mode": "llm" if llm else "stub",
        "retriever": retriever_name,
    }
    C.write_json(processed_dir / "_expand_stats.json", stats)
    print(f"[expand] {stats}")
    if not llm:
        print("[expand] stub data is only for pipeline verification. Configure a strong model before training.")


if __name__ == "__main__":
    main()
