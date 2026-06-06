# 阶段一执行计划 - 微调数据闭环增强版

## 1. 阶段目标

阶段一只负责本地数据闭环，不进入训练。按照最新作战方案，阶段一需要从“能生成样本”升级为“能对齐五类微调能力的训练数据基础设施”。

```text
Milvus chunk
  -> kb_chunks.jsonl
  -> train / holdout
  -> 数据校验报告
  -> messages 训练格式
  -> 五类能力映射元数据
```

最终训练目标对应五类能力：

```text
faithful  忠实回答
multi_hop 多片段整合
cite      引用标注
refuse    资料不足/冲突拒答
format    格式适配
```

## 2. 不做范围

```text
不训练模型
不接 vLLM
不改现有导入接口
不改查询链路
不提交真实数据和模型产物
```

## 3. 任务拆解

| 序号 | 任务 | 文件 | 验收 |
|---|---|---|---|
| 1 | 建立 fine_tuning 目录 | fine_tuning/ | 目录结构清晰 |
| 2 | 提供配置模板 | configs/config.example.yaml | 不包含密钥 |
| 3 | 导出 Milvus chunk | scripts/export_kb_chunks.py | 生成 kb_chunks.jsonl |
| 4 | 构造四类冷启动 SFT 样本 | scripts/build_sft_dataset.py | 生成 train / holdout |
| 5 | 数据校验 | scripts/validate_dataset.py | 无硬错误 |
| 6 | 转 messages 格式并写入能力映射 | scripts/convert_to_messages.py | 生成 messages_train / messages_holdout |
| 7 | 工程文档 | docs/*.md | 可阅读、可汇报、可复盘 |
| 8 | 补 format 样本计划 | docs/stage1_execution_plan.md | 明确阶段一增强项 |

## 4. 五类能力对齐

当前脚本先用四类样本跑通冷启动数据闭环：

| 当前 type | 作战方案 type | 说明 |
|---|---|---|
| answerable | faithful | 单片段忠实回答 |
| answerable | cite | 答案必须带 `[C1]` 引用 |
| multi_chunk | multi_hop | 综合 2-3 个片段回答 |
| unanswerable | refuse | 无关/不足资料拒答 |
| conflicting | refuse | 冲突资料拒答 |
| 待新增 | format | 操作步骤、故障处理、参数解释等固定格式 |

阶段一增强项不是立刻训练，而是补齐 `format` 样本生成设计：

```text
输入：真实 chunk + 问题意图
输出：步骤型 / 表格型 / 故障排查型 answer
校验：答案必须包含结构化小标题或步骤编号，且事实仍需引用 C 编号
```

## 5. 执行命令

```bash
cp fine_tuning/configs/config.example.yaml fine_tuning/configs/config.yaml
python fine_tuning/scripts/export_kb_chunks.py
python fine_tuning/scripts/build_sft_dataset.py --dry-run
python fine_tuning/scripts/validate_dataset.py
python fine_tuning/scripts/convert_to_messages.py
```

## 6. 作战方案新增要求

| 要求 | 阶段一处理方式 |
|---|---|
| Qwen3-8B QLoRA | 阶段一不训练，只产出 messages 数据 |
| LLaMA-Factory | messages_train.jsonl 作为后续输入 |
| labels mask | 通过 messages 格式让训练框架只对 assistant 段算 loss |
| RAGAS 评测 | 阶段一只保留 holdout，评估脚本放阶段四 |
| vLLM LoRA 接回 | 阶段一不部署，文档保留后续路径 |
| 五类能力 | 当前四类映射到五类，format 作为增强项补齐 |

## 7. 风险

| 风险 | 影响 | 应对 |
|---|---|---|
| Milvus 中文档少 | unanswerable / multi_chunk 不足 | 先导入 3-5 个不同商品文档 |
| chunk 内容为空 | 数据无法构造 | 查看 `_export_stats.json` |
| stub 数据质量低 | 不能正式训练 | 阶段二接强模型重建数据 |
| 配置泄露 | 安全风险 | `config.yaml` 已加入 `.gitignore` |
| 只做四类样本 | 缺少格式适配能力 | 阶段一增强补 `format` 样本生成器 |
| holdout 太小 | 评估不稳定 | 正式数据扩到 800-1500 条，holdout 保留 100-150 条 |

## 8. 完成标准

```text
四个脚本可运行
数据校验无硬错误
messages 格式可供 LLaMA-Factory / TRL 使用
Git 暂存前不包含 jsonl、config.yaml、模型产物
文档明确五类能力映射和 format 待补项
```

## 9. 阶段一后续增强清单

```text
1. 新增 build_format_samples()，生成操作步骤 / 故障排查 / 参数解释类样本
2. validate_dataset.py 增加 format 样本结构校验
3. convert_to_messages.py 保留 battle_capabilities 分桶
4. 增加 LLaMA-Factory dataset_info 示例
5. 增加 eval/refusal_accuracy.py 设计文档，为阶段四做准备
```
