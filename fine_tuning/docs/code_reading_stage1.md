# 代码阅读文档 - fine_tuning 阶段一

## 1. 模块定位

`fine_tuning` 是掌柜智库的微调专项模块，和现有 `knowledge` 导入、查询链路解耦。

它读取现有知识库导入链路已经写入 Milvus 的 chunk，构造生成模型 SFT 所需的数据。

## 2. 主流程

```text
export_kb_chunks.py
  -> build_sft_dataset.py
  -> validate_dataset.py
  -> convert_to_messages.py
```

主流程的设计重点是“先校验中间结构，再转换训练格式”。这样比直接生成 messages 更容易检查引用越界、拒答失败、重复样本和上下文长度问题。

## 3. 文件说明

| 文件 | 职责 |
|---|---|
| configs/config.example.yaml | 本地配置模板，支持从 knowledge/.env 兜底读取 Milvus 配置 |
| scripts/_common.py | 配置加载、jsonl 读写、引用识别、拒答识别、LLM 客户端封装 |
| scripts/export_kb_chunks.py | 从 Milvus 导出 chunk，输出 `data/raw/kb_chunks.jsonl` |
| scripts/build_sft_dataset.py | 基于 chunk 构造四类 SFT 样本 |
| scripts/validate_dataset.py | 校验引用、拒答、重复、长度、类型分布 |
| scripts/convert_to_messages.py | 转换为 SFT 训练所需 messages 格式 |

## 3.1 注释阅读重点

代码中中文注释主要覆盖：

```text
配置兜底逻辑
Milvus 连接超时保护
四类样本构造边界
拒答 / 冲突样本生成原因
硬错误 / 软告警划分
messages 与 assistant-only loss 的关系
```

## 4. 数据 schema

构造阶段使用：

```json
{
  "id": "kb-000001",
  "type": "answerable",
  "question": "...",
  "contexts": [
    {"cid": "C1", "source": "...", "text": "..."}
  ],
  "answer": "... [C1]",
  "meta": {
    "item_name": "...",
    "source_chunk_ids": [],
    "synthetic": false
  }
}
```

训练阶段转换为：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "【检索资料】...【问题】..."},
    {"role": "assistant", "content": "..."}
  ],
  "meta": {}
}
```

## 5. 四类样本与作战方案五类能力

| type | 目标 |
|---|---|
| answerable | 单片段忠实回答 |
| multi_chunk | 多片段综合回答 |
| unanswerable | 资料不足拒答 |
| conflicting | 合成冲突拒答 |

与新作战方案对应：

| 当前 type | 作战方案能力 |
|---|---|
| answerable | faithful + cite |
| multi_chunk | multi_hop |
| unanswerable | refuse |
| conflicting | refuse |
| 待新增 | format |

## 6. 与主项目关系

输入来自：

```text
knowledge 导入链路 -> Milvus kb_chunks
```

输出用于：

```text
Qwen3-8B / Qwen2.5 SFT 训练
```

训练阶段通过 LLaMA-Factory 或 TRL 读取 `messages_train.jsonl`，并只对 assistant 段计算 loss。这个设计对应作战方案里的 labels mask 要求。

阶段一不会影响：

```text
knowledge/api/import_router.py
knowledge/processor/import_processor
knowledge/processor/query_processor
```
