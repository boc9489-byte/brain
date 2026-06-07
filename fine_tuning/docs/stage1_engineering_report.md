# 阶段一工程汇报 - 掌柜智库知识库微调

## 1. 背景

当前掌柜智库已经具备知识库导入能力，可以将 PDF / Markdown 文档解析、切分、向量化并写入 Milvus。微调专项的目标是在 RAG 生成端提升忠实回答、引用标注和资料不足拒答能力。

## 2. 阶段一目标

阶段一聚焦数据闭环，验证现有知识库数据是否可以转化为 SFT 训练样本，并对齐新作战方案中的五类能力目标。

```text
Milvus chunk -> SFT 样本 -> 校验报告 -> messages 训练格式
```

新作战方案能力目标：

```text
faithful / multi_hop / cite / refuse / format
```

## 3. 本阶段交付

| 交付物 | 说明 |
|---|---|
| fine_tuning/configs/config.example.yaml | 配置模板 |
| fine_tuning/scripts/export_kb_chunks.py | 导出知识库 chunk |
| fine_tuning/scripts/build_sft_dataset.py | 构造四类 SFT 样本 |
| fine_tuning/scripts/validate_dataset.py | 训练前数据校验 |
| fine_tuning/scripts/convert_to_messages.py | 转 messages 训练格式 |
| fine_tuning/docs/*.md | 执行计划、代码阅读、进度与汇报文档 |

新增注释覆盖：

```text
配置读取和环境变量兜底
Milvus gRPC 连接超时保护
四类样本与五类能力映射
数据校验的硬错误 / 软告警
messages 格式和 labels mask 的关系
```

## 4. 设计原则

```text
1. 微调模块与现有 knowledge 主链路解耦
2. 本地配置和数据产物不进入 Git
3. 先 stub 跑通数据闭环，再接强模型造正式数据
4. 训练前必须先通过硬校验
```

## 5. 当前状态

代码层面已经完成第一阶段脚手架，并已用真实 Milvus 导出数据跑通：

```text
export -> build -> validate -> convert
```

真实数据验证结果：

```text
collection=kb_chunks
exported=130
empty_content_skipped=0
distinct_items=3
actual_total=59
train=49
holdout=10
dropped_duplicates=7
validate 硬错误=0
messages_train=49
messages_holdout=10
```

测试过程已记录在 `fine_tuning/docs/stage1_test_record.md`。

## 6. 验收命令

```bash
uv run python fine_tuning/scripts/export_kb_chunks.py
uv run python fine_tuning/scripts/build_sft_dataset.py --dry-run
uv run python fine_tuning/scripts/validate_dataset.py
uv run python fine_tuning/scripts/convert_to_messages.py
```

当前验收状态：

| 命令 | 状态 | 说明 |
|---|---|---|
| export_kb_chunks.py | 通过 | 真实 Milvus 导出 130 条 chunk |
| build_sft_dataset.py --dry-run | 通过 | 构造 59 条 stub 样本，丢弃 7 条重复样本 |
| validate_dataset.py | 通过 | 硬错误 0 |
| convert_to_messages.py | 通过 | messages_train=49、messages_holdout=10 |

## 7. 后续计划

```text
阶段一增强：补 format 样本生成器、LLaMA-Factory dataset_info 示例
阶段二：接强模型生成正式 SFT 数据，并人工抽检
阶段三：云 GPU Qwen3-8B QLoRA
阶段四：Base vs SFT + RAGAS / 拒答准确率评测
阶段五：vLLM LoRA 接回业务
```
