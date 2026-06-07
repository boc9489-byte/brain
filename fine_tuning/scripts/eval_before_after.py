#!/usr/bin/env python3
"""阶段四 Base vs SFT 离线评估脚本。

输入：
    fine_tuning/data/processed/sft_holdout.jsonl

输出：
    fine_tuning/data/eval/eval_report.md
    fine_tuning/data/eval/_eval_metrics.json
    fine_tuning/data/eval/predictions.jsonl
    fine_tuning/data/eval/bad_cases.jsonl

本地可先用 --stub 验证指标计算与报告生成。真实模型评估需要 GPU/模型环境。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol

import _common as C


REFUSE_SUBTYPES = ["no_recall", "weak_recall", "conflict"]
METRIC_LABELS = [
    ("refusal_recall", "拒答 Recall"),
    ("refusal_precision", "拒答 Precision"),
    ("refusal_f1", "拒答 F1"),
    ("false_refusal", "误拒率"),
    ("recall_no_recall", "no_recall Recall"),
    ("recall_weak_recall", "weak_recall Recall"),
    ("recall_conflict", "conflict Recall"),
    ("citation_validity", "引用有效率"),
    ("faithfulness", "忠实性"),
    ("completeness", "完整性"),
]


class Generator(Protocol):
    """统一模型生成接口。"""

    def generate(self, messages: List[Dict[str, str]], gold: str = "") -> str:
        ...


class NaiveBaseStub:
    """弱基线：永远作答，故意不拒答，用于验证拒答指标。"""

    def generate(self, messages: List[Dict[str, str]], gold: str = "") -> str:
        return "根据资料，相关内容如上所述 [1]。"


class OracleSftStub:
    """理想 SFT：直接复现 gold，用于验证指标上限和报告渲染。"""

    def generate(self, messages: List[Dict[str, str]], gold: str = "") -> str:
        return gold


class HFGenerator:
    """真实模型生成器：base 或 base + LoRA adapter。"""

    def __init__(self, base_model: str, adapter_path: Optional[str] = None, max_new_tokens: int = 256):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise SystemExit("Missing inference dependencies. Install requirements-train.txt in a model environment.") from exc

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise SystemExit("Missing peft. Install requirements-train.txt before loading adapter.") from exc
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def generate(self, messages: List[Dict[str, str]], gold: str = "") -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def holdout_path(cfg: Dict[str, Any], override: Optional[str]) -> Path:
    """定位 holdout 文件。"""
    if override:
        return C.repo_path(override)
    eval_cfg = cfg.get("eval", {}) or {}
    if eval_cfg.get("holdout_path"):
        return C.repo_path(eval_cfg["holdout_path"])
    return C.repo_path(cfg["dataset"]["out_dir"]) / "processed" / "sft_holdout.jsonl"


def output_dir(cfg: Dict[str, Any], override: Optional[str]) -> Path:
    """定位评估输出目录。"""
    if override:
        return C.repo_path(override)
    return C.repo_path((cfg.get("eval") or {}).get("output_dir", "fine_tuning/data/eval"))


def prompt_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """取 system + user，去掉 gold assistant。"""
    return [message for message in sample.get("messages", []) if message.get("role") != "assistant"]


def gold_answer(sample: Dict[str, Any]) -> str:
    """取 gold assistant。"""
    messages = sample.get("messages") or []
    return messages[-1].get("content", "") if messages else ""


def user_content(sample: Dict[str, Any]) -> str:
    """取 user 内容。"""
    messages = sample.get("messages") or []
    return messages[1].get("content", "") if len(messages) > 1 else ""


def available_citations(user: str) -> set[str]:
    """从检索资料中提取可用引用编号。"""
    return set(re.findall(r"^\[(\d+)\]", user or "", flags=re.M))


def make_judge(llm: Optional[Dict[str, Any]]):
    """创建可选 LLM judge。"""
    if not llm:
        return None

    def judge(user: str, answer: str) -> Dict[str, Any]:
        prompt = (
            f"{user}\n\n"
            f"【模型答案】\n{answer}\n\n"
            "请判断：\n"
            "1. faithful：答案每个事实是否都能被检索资料支撑，是则 1，否则 0；\n"
            "2. complete：答案是否覆盖回答问题所需的关键条件，是则 1，否则 0。\n"
            '只输出 JSON：{"faithful":0或1,"complete":0或1}'
        )
        return C.llm_json(llm, "你是严格的 RAG 评估员，只输出 JSON。", prompt) or {}

    return judge


def classify_bad_case(row: Dict[str, Any]) -> Optional[str]:
    """根据单条评估结果分类 bad case。"""
    if row["should_refuse"] and not row["did_refuse"]:
        return "missed_refusal"
    if (not row["should_refuse"]) and row["did_refuse"]:
        return "false_refusal"
    if row.get("citation_ok") is False:
        return "citation_invalid"
    if row.get("faithful") == 0:
        return "unfaithful"
    if row.get("complete") == 0:
        return "incomplete"
    return None


def evaluate_model(
    samples: List[Dict[str, Any]],
    generator: Generator,
    model_name: str,
    judge=None,
) -> List[Dict[str, Any]]:
    """运行一个模型并收集逐条评估结果。"""
    rows = []
    for index, sample in enumerate(samples, 1):
        meta = sample.get("meta") or {}
        sample_type = meta.get("type")
        subtype = meta.get("subtype")
        gold = gold_answer(sample)
        user = user_content(sample)
        prediction = generator.generate(prompt_messages(sample), gold=gold) or ""
        should_refuse = sample_type == "refuse"
        did_refuse = C.is_refusal(prediction)

        citation_ok = None
        if not should_refuse and not did_refuse:
            used = C.num_citations_in(prediction)
            citation_ok = bool(used) and used.issubset(available_citations(user))

        faithful = complete = None
        if judge and not should_refuse and not did_refuse:
            judged = judge(user, prediction)
            faithful = judged.get("faithful")
            complete = judged.get("complete")

        row = {
            "id": meta.get("id") or f"sample-{index:06d}",
            "model": model_name,
            "type": sample_type,
            "subtype": subtype,
            "question": user.split("【问题】", 1)[-1].strip() if "【问题】" in user else "",
            "gold": gold,
            "prediction": prediction,
            "should_refuse": should_refuse,
            "did_refuse": did_refuse,
            "citation_ok": citation_ok,
            "faithful": faithful,
            "complete": complete,
        }
        row["bad_case"] = classify_bad_case(row)
        rows.append(row)
    return rows


def safe_div(numerator: float, denominator: float) -> float:
    """安全除法。"""
    return numerator / denominator if denominator else 0.0


def compute_metrics(rows: List[Dict[str, Any]], subtypes: Iterable[str]) -> Dict[str, float]:
    """计算阶段四核心指标。"""
    tp = sum(row["should_refuse"] and row["did_refuse"] for row in rows)
    fn = sum(row["should_refuse"] and not row["did_refuse"] for row in rows)
    fp = sum((not row["should_refuse"]) and row["did_refuse"] for row in rows)
    tn = sum((not row["should_refuse"]) and not row["did_refuse"] for row in rows)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    metrics = {
        "refusal_precision": precision,
        "refusal_recall": recall,
        "refusal_f1": safe_div(2 * precision * recall, precision + recall),
        "false_refusal": safe_div(fp, fp + tn),
    }
    for subtype in subtypes:
        bucket = [row for row in rows if row.get("subtype") == subtype]
        if bucket:
            metrics[f"recall_{subtype}"] = safe_div(sum(row["did_refuse"] for row in bucket), len(bucket))

    citation_rows = [row for row in rows if row.get("citation_ok") is not None]
    metrics["citation_validity"] = safe_div(sum(row["citation_ok"] for row in citation_rows), len(citation_rows))

    faithful_rows = [row for row in rows if row.get("faithful") is not None]
    complete_rows = [row for row in rows if row.get("complete") is not None]
    if faithful_rows:
        metrics["faithfulness"] = safe_div(sum(row["faithful"] for row in faithful_rows), len(faithful_rows))
    if complete_rows:
        metrics["completeness"] = safe_div(sum(row["complete"] for row in complete_rows), len(complete_rows))
    return metrics


def render_metrics_table(base_metrics: Dict[str, float], sft_metrics: Dict[str, float]) -> str:
    """渲染 before/after 指标表。"""
    lines = ["| 指标 | Base | SFT | Δ |", "|---|---:|---:|---:|"]
    for key, label in METRIC_LABELS:
        if key not in base_metrics or key not in sft_metrics:
            continue
        base = base_metrics[key]
        sft = sft_metrics[key]
        lines.append(f"| {label} | {base:.2f} | {sft:.2f} | {sft - base:+.2f} |")
    return "\n".join(lines)


def render_bad_case_summary(rows: List[Dict[str, Any]]) -> str:
    """渲染 bad case 汇总。"""
    counts = Counter(row["bad_case"] for row in rows if row.get("bad_case"))
    if not counts:
        return "无 bad case。"
    lines = ["| Bad Case | 数量 |", "|---|---:|"]
    for name, count in sorted(counts.items()):
        lines.append(f"| {name} | {count} |")
    return "\n".join(lines)


def write_outputs(
    out_dir: Path,
    samples: List[Dict[str, Any]],
    base_rows: List[Dict[str, Any]],
    sft_rows: List[Dict[str, Any]],
    base_metrics: Dict[str, float],
    sft_metrics: Dict[str, float],
) -> None:
    """写评估报告、指标和明细。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = base_rows + sft_rows
    C.write_jsonl(out_dir / "predictions.jsonl", all_rows)
    C.write_jsonl(out_dir / "bad_cases.jsonl", [row for row in all_rows if row.get("bad_case")])
    C.write_json(
        out_dir / "_eval_metrics.json",
        {
            "sample_count": len(samples),
            "base": base_metrics,
            "sft": sft_metrics,
            "bad_case_count": sum(1 for row in all_rows if row.get("bad_case")),
        },
    )
    report = "\n".join(
        [
            "# Base vs SFT 离线评估报告",
            "",
            f"- 样本数: {len(samples)}",
            "",
            "## 总体指标",
            "",
            render_metrics_table(base_metrics, sft_metrics),
            "",
            "## Base Bad Case",
            "",
            render_bad_case_summary(base_rows),
            "",
            "## SFT Bad Case",
            "",
            render_bad_case_summary(sft_rows),
            "",
        ]
    )
    (out_dir / "eval_report.md").write_text(report, encoding="utf-8")


def load_generators(args: argparse.Namespace, cfg: Dict[str, Any]) -> tuple[Generator, Generator]:
    """加载 Base 和 SFT 生成器。"""
    if args.stub:
        print("[eval] stub mode: base=naive, sft=oracle")
        return NaiveBaseStub(), OracleSftStub()

    eval_cfg = cfg.get("eval", {}) or {}
    base_model = args.base or eval_cfg.get("base_model") or (cfg.get("train") or {}).get("base_model")
    adapter = args.adapter or eval_cfg.get("adapter_path") or (cfg.get("train") or {}).get("output_dir")
    if not base_model:
        raise SystemExit("Missing base model. Set eval.base_model or pass --base.")
    if not adapter:
        raise SystemExit("Missing adapter path. Set eval.adapter_path or pass --adapter.")
    adapter_path = C.repo_path(adapter)
    if not adapter_path.exists():
        raise SystemExit(f"Adapter path not found: {adapter_path}. Run stage 3 training first.")
    max_new = int(args.max_new or eval_cfg.get("max_new_tokens", 256))
    return HFGenerator(base_model, None, max_new), HFGenerator(base_model, str(adapter_path), max_new)


def check_only(samples: List[Dict[str, Any]], out: Path) -> None:
    """只检查评估输入输出，不加载模型。"""
    print(f"[eval-check] samples={len(samples)}")
    print(f"[eval-check] output_dir={out}")
    if not samples:
        raise SystemExit("[eval-check] no samples found")
    roles = [message.get("role") for message in samples[0].get("messages", [])]
    print(f"[eval-check] first_roles={roles}")
    if roles != ["system", "user", "assistant"]:
        raise SystemExit("[eval-check] invalid messages schema")
    print("[eval-check] passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=C.DEFAULT_CONFIG_PATH)
    parser.add_argument("--holdout", default=None, help="覆盖 sft_holdout.jsonl 路径")
    parser.add_argument("--out", default=None, help="覆盖 eval 输出目录")
    parser.add_argument("--base", default=None, help="Base 模型路径或名称")
    parser.add_argument("--adapter", default=None, help="LoRA adapter 路径")
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument("--stub", action="store_true", help="离线验证：base 弱基线，sft oracle")
    parser.add_argument("--judge", action="store_true", help="使用配置中的 LLM 做忠实性和完整性 judge")
    parser.add_argument("--check-only", action="store_true", help="只检查输入输出，不加载模型")
    args = parser.parse_args()

    cfg = C.load_config(args.config)
    samples = C.read_jsonl(holdout_path(cfg, args.holdout))
    out = output_dir(cfg, args.out)
    if args.check_only:
        check_only(samples, out)
        return
    if not samples:
        raise SystemExit("No holdout samples found. Run expand_dataset.py first.")

    subtypes = (cfg.get("expand") or {}).get("refuse_subtypes", REFUSE_SUBTYPES)
    judge = make_judge(C.get_llm(cfg)) if args.judge else None
    if args.judge and not judge:
        print("[eval] judge requested but llm is not configured; skip judge metrics.")

    base_gen, sft_gen = load_generators(args, cfg)
    print(f"[eval] samples={len(samples)}")
    print("[eval] running base...")
    base_rows = evaluate_model(samples, base_gen, "base", judge)
    print("[eval] running sft...")
    sft_rows = evaluate_model(samples, sft_gen, "sft", judge)
    base_metrics = compute_metrics(base_rows, subtypes)
    sft_metrics = compute_metrics(sft_rows, subtypes)
    write_outputs(out, samples, base_rows, sft_rows, base_metrics, sft_metrics)

    print(render_metrics_table(base_metrics, sft_metrics))
    print(f"[eval] report -> {out / 'eval_report.md'}")


if __name__ == "__main__":
    main()
