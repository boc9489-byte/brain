#!/usr/bin/env python3
"""阶段二离线测试。

只测试 local + stub 路径，不依赖 Milvus、强模型或 GPU。这个测试的目标是确认
阶段二新增 schema、五类样本和 messages 校验门禁没有被改坏。
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "fine_tuning" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import expand_dataset as E  # noqa: E402


def sample_chunks():
    """构造小型产品说明书 chunk。"""
    return [
        {
            "chunk_id": 1,
            "content": "RS-12 做短路蜂鸣测试时，将黑色表笔插入 COM 端口，红色表笔插入 Ω 端口。当电阻小于 30Ω 时会蜂鸣。",
            "title": "短路蜂鸣测试",
            "parent_title": "RS-12 使用说明",
            "file_title": "万用表RS-12的使用",
            "item_name": "RS-12 数字万用表",
        },
        {
            "chunk_id": 2,
            "content": "RS-12 的电池为一粒 9V 电池，显示为 3 1/2 数位、2000 位液晶显示。",
            "title": "技术指标",
            "parent_title": "RS-12 使用说明",
            "file_title": "万用表RS-12的使用",
            "item_name": "RS-12 数字万用表",
        },
        {
            "chunk_id": 3,
            "content": "华为擎云 M272Q 可在 OSD 菜单中设置刷新率、游戏准星、画面防撕裂等游戏辅助功能。",
            "title": "游戏辅助",
            "parent_title": "显示器设置",
            "file_title": "华为擎云 M272Q 用户指南",
            "item_name": "华为擎云 M272Q 显示器",
        },
    ]


def test_stage2_stub_builders_cover_all_types():
    chunks = sample_chunks()
    by_item = defaultdict(list)
    for chunk in chunks:
        by_item[E.item_key(chunk)].append(chunk)
    retriever = E.LocalRetriever(chunks)

    samples = []
    for sample_type in ("faithful", "multi_hop", "cite", "format"):
        sample = E.build_grounded(sample_type, chunks, by_item, retriever, None, [], 3)
        assert sample, f"{sample_type} sample was not built"
        ok, reason = E.validate_inline(sample)
        assert ok, f"{sample_type} invalid: {reason}"
        samples.append(sample)

    for subtype in ("no_recall", "weak_recall", "conflict"):
        sample = E.build_refuse(subtype, None, chunks, by_item, None, [])
        assert sample, f"{subtype} sample was not built"
        ok, reason = E.validate_inline(sample)
        assert ok, f"{subtype} invalid: {reason}"
        samples.append(sample)

    types = {sample["meta"]["type"] for sample in samples}
    subtypes = {sample["meta"].get("subtype") for sample in samples if sample["meta"]["type"] == "refuse"}
    assert {"faithful", "multi_hop", "cite", "format", "refuse"} <= types
    assert {"no_recall", "weak_recall", "conflict"} <= subtypes


def main() -> None:
    test_stage2_stub_builders_cover_all_types()
    print("[ok] stage2 local/stub builders cover all types and pass inline validation")


if __name__ == "__main__":
    main()

