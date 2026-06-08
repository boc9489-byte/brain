#!/usr/bin/env python3
"""阶段四评估指标单测。

只测试纯函数和 stub 生成器，不依赖真实模型。
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "fine_tuning" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import eval_before_after as E  # noqa: E402


def make_sample(sample_type: str, user: str, answer: str, subtype: str | None = None):
    meta = {"id": f"{sample_type}-1", "type": sample_type}
    if subtype:
        meta["subtype"] = subtype
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
        "meta": meta,
    }


def test_stub_metrics_have_expected_direction():
    samples = [
        make_sample(
            "faithful",
            "【检索资料】\n[1] RS-12 使用 9V 电池。\n\n【问题】\nRS-12 使用什么电池？",
            "RS-12 使用 9V 电池 [1]。",
        ),
        make_sample(
            "refuse",
            "【检索资料】\n[1] RS-12 使用 9V 电池。\n\n【问题】\nRS-12 保修多久？",
            "资料未提供保修期限信息，无法回答。建议补充保修条款。",
            subtype="weak_recall",
        ),
    ]
    base_rows = E.evaluate_model(samples, E.NaiveBaseStub(), "base")
    sft_rows = E.evaluate_model(samples, E.OracleSftStub(), "sft")
    base_metrics = E.compute_metrics(base_rows, ["weak_recall"])
    sft_metrics = E.compute_metrics(sft_rows, ["weak_recall"])

    assert base_metrics["refusal_recall"] == 0.0
    assert sft_metrics["refusal_recall"] == 1.0
    assert sft_metrics["false_refusal"] == 0.0
    assert sft_metrics["citation_validity"] == 1.0


def test_refusal_detector_matches_missing_information_wording():
    row = E.evaluate_model(
        [
            make_sample(
                "refuse",
                "【检索资料】\n[1] RS-12 使用 9V 电池。\n\n【问题】\nRS-12 保修多久？",
                "资料中没有提供保修期限和售后政策的信息。",
                subtype="weak_recall",
            )
        ],
        E.OracleSftStub(),
        "sft",
    )[0]

    assert row["did_refuse"] is True
    assert row["bad_case"] is None


def main() -> None:
    test_stub_metrics_have_expected_direction()
    test_refusal_detector_matches_missing_information_wording()
    print("[ok] stage4 eval metrics direction is correct")


if __name__ == "__main__":
    main()
