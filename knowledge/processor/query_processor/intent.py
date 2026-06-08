"""查询意图识别工具。

当前查询链路没有独立的意图识别节点，这里提供一个轻量、可测试的规则分类器。
上游如果已经通过 LLM 写入 `intent_type`，回答节点会优先使用上游结果；否则使用本模块兜底。
"""

from __future__ import annotations

from typing import Iterable, Optional


INTENT_GENERAL = "general"
INTENT_INSTALL = "install_config"
INTENT_TROUBLESHOOTING = "troubleshooting"
INTENT_PARAMETER = "parameter"
INTENT_OPERATION = "operation"
INTENT_IMAGE = "image_request"
INTENT_COMPARISON = "comparison"
INTENT_AFTER_SALES = "after_sales"

KNOWN_INTENTS = {
    INTENT_GENERAL,
    INTENT_INSTALL,
    INTENT_TROUBLESHOOTING,
    INTENT_PARAMETER,
    INTENT_OPERATION,
    INTENT_IMAGE,
    INTENT_COMPARISON,
    INTENT_AFTER_SALES,
}


def normalize_intent(intent: Optional[str]) -> str:
    """标准化上游传入的 intent，避免脏值污染 prompt 和 trace。"""
    value = (intent or "").strip().lower()
    return value if value in KNOWN_INTENTS else INTENT_GENERAL


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def classify_query_intent(query: str) -> str:
    """按用户问题识别粗粒度意图。

    这里故意保持规则简单透明：先识别高优先级的图片、故障、配置安装等强信号，
    再回落到参数、对比、操作和通用问题。
    """
    text = (query or "").strip().lower()
    if not text:
        return INTENT_GENERAL

    if _contains_any(text, ["图片", "图示", "示意图", "接线图", "外观", "结构图", "照片", "看图"]):
        return INTENT_IMAGE
    if _contains_any(text, ["报错", "错误", "故障", "失败", "无法", "不能", "不显示", "连不上", "异常", "告警"]):
        return INTENT_TROUBLESHOOTING
    if _contains_any(text, ["安装", "配置", "设置", "接线", "连接", "部署", "初始化", "配网"]):
        return INTENT_INSTALL
    if _contains_any(text, ["参数", "规格", "尺寸", "电压", "电流", "功率", "频率", "接口", "支持"]):
        return INTENT_PARAMETER
    if _contains_any(text, ["区别", "对比", "比较", "哪个好", "差异"]):
        return INTENT_COMPARISON
    if _contains_any(text, ["保修", "售后", "维修", "退换", "质保"]):
        return INTENT_AFTER_SALES
    if _contains_any(text, ["怎么用", "如何使用", "操作", "步骤", "测量", "打开", "关闭", "升级"]):
        return INTENT_OPERATION
    return INTENT_GENERAL
