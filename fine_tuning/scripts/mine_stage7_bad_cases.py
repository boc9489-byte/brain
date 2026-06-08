#!/usr/bin/env python3
"""阶段七：从 answer trace 挖掘 bad case 和 Golden Set 候选。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACE_PATH = REPO_ROOT / "fine_tuning" / "data" / "online" / "answer_traces.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "fine_tuning" / "data" / "online"

# 每类 bad case 都绑定一个处理建议，报告可以直接变成排障清单。
SUGGESTIONS = {
    "empty_answer": "检查模型服务异常、超时和 AnswerOutPutNode 兜底逻辑。",
    "model_error": "优先查看模型服务日志和请求超时配置。",
    "no_context_answered": "检查召回为空时的拒答约束，补充 no-recall 回归样本。",
    "missing_citation": "检查回答 prompt 的引用约束，必要时加入引用缺失回归样本。",
    "sft_over_refusal_candidate": "人工复核是否误拒；如确认误拒，降低拒答样本比例或补充可回答样本。",
    "high_latency": "检查 prompt 长度、模型服务负载和 max_tokens 配置。",
    "long_prompt": "检查 context packing、top-k 和 max_context_chars。",
    "short_answer": "人工复核回答是否不完整，必要时补 format / completeness 样本。",
}

# 严重度 3 优先影响正确性或可用性；严重度 1 多为优化项或人工复核线索。
SEVERITY = {
    "empty_answer": 3,
    "model_error": 3,
    "no_context_answered": 3,
    "missing_citation": 2,
    "sft_over_refusal_candidate": 2,
    "high_latency": 2,
    "long_prompt": 1,
    "short_answer": 1,
}


def sample_traces() -> List[Dict[str, Any]]:
    """内置样例，便于无线上 trace 时验证阶段七通路。"""
    return [
        {
            "ts": "2026-06-07T11:00:00+08:00",
            "task_id": "sample-no-context",
            "provider": "base",
            "model_name": "qwen-flash",
            "latency_ms": 850,
            "query_hash": "sha256:q1",
            "answer_hash": "sha256:a1",
            "answer_chars": 80,
            "prompt_chars": 400,
            "context_count": 0,
            "used_citations": [],
            "is_refusal": False,
            "has_answer": True,
            "error": "",
        },
        {
            "ts": "2026-06-07T11:01:00+08:00",
            "task_id": "sample-missing-citation",
            "provider": "sft",
            "model_name": "kb-sft",
            "latency_ms": 1300,
            "query_hash": "sha256:q2",
            "answer_hash": "sha256:a2",
            "answer_chars": 90,
            "prompt_chars": 1800,
            "context_count": 3,
            "used_citations": [],
            "is_refusal": False,
            "has_answer": True,
            "error": "",
        },
        {
            "ts": "2026-06-07T11:02:00+08:00",
            "task_id": "sample-over-refusal",
            "provider": "sft",
            "model_name": "kb-sft",
            "latency_ms": 900,
            "query_hash": "sha256:q3",
            "answer_hash": "sha256:a3",
            "answer_chars": 35,
            "prompt_chars": 1500,
            "context_count": 2,
            "used_citations": [],
            "is_refusal": True,
            "has_answer": True,
            "error": "",
        },
        {
            "ts": "2026-06-07T11:03:00+08:00",
            "task_id": "sample-high-latency",
            "provider": "base",
            "model_name": "qwen-flash",
            "latency_ms": 12000,
            "query_hash": "sha256:q4",
            "answer_hash": "sha256:a4",
            "answer_chars": 18,
            "prompt_chars": 9000,
            "context_count": 4,
            "used_citations": ["1"],
            "is_refusal": False,
            "has_answer": True,
            "error": "",
        },
    ]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL。"""
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """写 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def trace_id(trace: Dict[str, Any]) -> str:
    """生成稳定 trace id。"""
    raw = "|".join(
        [
            str(trace.get("task_id", "")),
            str(trace.get("query_hash", "")),
            str(trace.get("answer_hash", "")),
            str(trace.get("ts", "")),
        ]
    )
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def classify_trace(
    trace: Dict[str, Any],
    *,
    latency_threshold_ms: int,
    prompt_threshold_chars: int,
    short_answer_chars: int,
) -> List[str]:
    """返回 trace 命中的 bad case 类型。"""
    bad_types: List[str] = []
    context_count = int(trace.get("context_count") or 0)
    answer_chars = int(trace.get("answer_chars") or 0)
    prompt_chars = int(trace.get("prompt_chars") or 0)
    latency_ms = int(trace.get("latency_ms") or 0)
    used_citations = trace.get("used_citations") or []
    is_refusal = bool(trace.get("is_refusal"))
    has_answer = bool(trace.get("has_answer", answer_chars > 0))
    provider = str(trace.get("provider") or "").lower()
    error = str(trace.get("error") or "").strip()

    # 规则顺序从硬故障到质量问题排列，方便报告里先看到最高风险原因。
    if not has_answer or answer_chars <= 0:
        bad_types.append("empty_answer")
    if error:
        bad_types.append("model_error")
    if context_count <= 0 and has_answer and not is_refusal:
        bad_types.append("no_context_answered")
    if context_count > 0 and has_answer and not is_refusal and not used_citations:
        bad_types.append("missing_citation")
    if provider == "sft" and is_refusal and context_count > 0:
        bad_types.append("sft_over_refusal_candidate")
    if latency_ms >= latency_threshold_ms:
        bad_types.append("high_latency")
    if prompt_chars >= prompt_threshold_chars:
        bad_types.append("long_prompt")
    if has_answer and not is_refusal and 0 < answer_chars <= short_answer_chars:
        bad_types.append("short_answer")

    return bad_types


def suggestion_for(bad_types: Sequence[str]) -> str:
    """合并修复建议。"""
    return "；".join(SUGGESTIONS[t] for t in bad_types if t in SUGGESTIONS)


def build_bad_case(trace: Dict[str, Any], bad_types: Sequence[str]) -> Dict[str, Any]:
    """构造 bad case 输出记录。"""
    severity = max(SEVERITY[t] for t in bad_types)
    out = {
        "trace_id": trace_id(trace),
        "task_id": trace.get("task_id", ""),
        "ts": trace.get("ts", ""),
        "provider": trace.get("provider", ""),
        "model_name": trace.get("model_name", ""),
        "bad_types": list(bad_types),
        "severity": severity,
        "latency_ms": trace.get("latency_ms", 0),
        "query_hash": trace.get("query_hash", ""),
        "answer_hash": trace.get("answer_hash", ""),
        "answer_chars": trace.get("answer_chars", 0),
        "prompt_chars": trace.get("prompt_chars", 0),
        "context_count": trace.get("context_count", 0),
        "used_citations": trace.get("used_citations", []),
        "is_refusal": trace.get("is_refusal", False),
        "suggestion": suggestion_for(bad_types),
    }
    if trace.get("query"):
        out["query"] = trace["query"]
    if trace.get("answer_preview"):
        out["answer_preview"] = trace["answer_preview"]
    return out


def mine_bad_cases(
    traces: Sequence[Dict[str, Any]],
    *,
    latency_threshold_ms: int = 8000,
    prompt_threshold_chars: int = 8000,
    short_answer_chars: int = 20,
) -> List[Dict[str, Any]]:
    """挖掘 bad case，并按严重度和延迟排序。"""
    bad_cases: List[Dict[str, Any]] = []
    for trace in traces:
        bad_types = classify_trace(
            trace,
            latency_threshold_ms=latency_threshold_ms,
            prompt_threshold_chars=prompt_threshold_chars,
            short_answer_chars=short_answer_chars,
        )
        if bad_types:
            bad_cases.append(build_bad_case(trace, bad_types))
    return sorted(bad_cases, key=lambda row: (-row["severity"], -int(row.get("latency_ms") or 0), row["trace_id"]))


def build_golden_candidates(bad_cases: Sequence[Dict[str, Any]], min_severity: int = 2) -> List[Dict[str, Any]]:
    """从 bad case 中抽取 Golden Set 候选，并按 query_hash 去重。"""
    dedup: Dict[str, Dict[str, Any]] = {}
    for bad_case in bad_cases:
        query_hash = bad_case.get("query_hash")
        if not query_hash or int(bad_case.get("severity") or 0) < min_severity:
            continue
        old = dedup.get(query_hash)
        # 同一问题只保留最新一次异常，避免线上重复提问把 Golden 候选刷爆。
        if old is None or str(bad_case.get("ts", "")) >= str(old.get("ts", "")):
            dedup[query_hash] = bad_case

    candidates: List[Dict[str, Any]] = []
    for idx, bad_case in enumerate(sorted(dedup.values(), key=lambda row: row["trace_id"]), 1):
        # 候选只进入人工标注池，不会自动回灌训练集，避免把坏答案直接学进去。
        candidate = {
            "candidate_id": f"golden-candidate-{idx:06d}",
            "source_trace_id": bad_case["trace_id"],
            "query_hash": bad_case["query_hash"],
            "provider": bad_case.get("provider", ""),
            "model_name": bad_case.get("model_name", ""),
            "bad_types": bad_case.get("bad_types", []),
            "severity": bad_case.get("severity", 0),
            "needs_human_label": True,
            "label_fields": ["question", "ground_truth", "expected_sources", "category"],
        }
        if bad_case.get("query"):
            candidate["question"] = bad_case["query"]
        candidates.append(candidate)
    return candidates


def render_report(traces: Sequence[Dict[str, Any]], bad_cases: Sequence[Dict[str, Any]], candidates: Sequence[Dict[str, Any]]) -> str:
    """渲染 Markdown 报告。"""
    type_counts = Counter(t for row in bad_cases for t in row.get("bad_types", []))
    severity_counts = Counter(row.get("severity", 0) for row in bad_cases)
    provider_counts = Counter(row.get("provider", "unknown") for row in bad_cases)

    lines = [
        "# 阶段七 Bad Case 挖掘报告",
        "",
        "## 1. 总览",
        "",
        f"- trace_total={len(traces)}",
        f"- bad_case_total={len(bad_cases)}",
        f"- golden_candidate_total={len(candidates)}",
        "",
        "## 2. Bad Case 类型分布",
        "",
        "| 类型 | 数量 |",
        "|---|---:|",
    ]
    for name, count in type_counts.most_common():
        lines.append(f"| {name} | {count} |")

    lines.extend(["", "## 3. 严重度分布", "", "| 严重度 | 数量 |", "|---:|---:|"])
    for severity, count in sorted(severity_counts.items(), reverse=True):
        lines.append(f"| {severity} | {count} |")

    lines.extend(["", "## 4. Provider 分布", "", "| Provider | 数量 |", "|---|---:|"])
    for provider, count in provider_counts.most_common():
        lines.append(f"| {provider} | {count} |")

    lines.extend(["", "## 5. Top Bad Cases", "", "| trace_id | 类型 | 严重度 | 建议 |", "|---|---|---:|---|"])
    for row in list(bad_cases)[:10]:
        bad_types = ",".join(row.get("bad_types", []))
        suggestion = row.get("suggestion", "")
        lines.append(f"| {row['trace_id'][:20]}... | {bad_types} | {row.get('severity', 0)} | {suggestion} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_TRACE_PATH), help="answer_traces.jsonl 路径。")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出目录。")
    parser.add_argument("--sample", action="store_true", help="使用内置 sample trace。")
    parser.add_argument("--latency-threshold-ms", type=int, default=8000)
    parser.add_argument("--prompt-threshold-chars", type=int, default=8000)
    parser.add_argument("--short-answer-chars", type=int, default=20)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    traces = sample_traces() if args.sample else read_jsonl(input_path)
    if not traces:
        raise SystemExit(f"[stage7] no traces found: {input_path}. Use --sample to validate the pipeline.")

    bad_cases = mine_bad_cases(
        traces,
        latency_threshold_ms=args.latency_threshold_ms,
        prompt_threshold_chars=args.prompt_threshold_chars,
        short_answer_chars=args.short_answer_chars,
    )
    candidates = build_golden_candidates(bad_cases)

    bad_path = out_dir / "bad_cases.jsonl"
    golden_path = out_dir / "golden_candidates.jsonl"
    report_path = out_dir / "_stage7_bad_case_report.md"
    write_jsonl(bad_path, bad_cases)
    write_jsonl(golden_path, candidates)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(traces, bad_cases, candidates), encoding="utf-8")

    print(f"[stage7] traces={len(traces)} bad_cases={len(bad_cases)} golden_candidates={len(candidates)}")
    print(f"[stage7] bad_cases -> {bad_path}")
    print(f"[stage7] golden_candidates -> {golden_path}")
    print(f"[stage7] report -> {report_path}")


if __name__ == "__main__":
    main()
