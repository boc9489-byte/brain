# 阶段二执行计划 - 强模型正式造数

## 1. 一句话定位

阶段二负责把阶段一验证过的 Milvus chunk 数据闭环，升级为可训练前验收的正式 SFT 数据构造链路。

```text
阶段一证明流程能跑通。
阶段二证明数据能用于后续训练前准备。
```

阶段二仍然不训练模型，不接 vLLM，不修改线上导入链路和查询链路。

## 2. 背景与目标

### 2.1 背景

阶段一已经完成：

```text
Milvus chunk -> kb_chunks.jsonl -> stub 样本 -> validate -> messages
```

但阶段一样本通过 `--dry-run` 模板生成，只能验证工程流程，不能作为正式训练数据。

### 2.2 阶段二目标

阶段二目标是：

```text
基于真实知识库 chunk，接入强模型生成高质量 messages SFT 数据，
并用自动校验 + 人工抽检入口控制数据质量。
```

本阶段覆盖五类能力：

| 能力 | 样本 type | 说明 |
|---|---|---|
| faithful | `faithful` | 单片段忠实回答 |
| multi_hop | `multi_hop` | 多片段综合回答 |
| cite | `cite` | 引用标注稳定性 |
| refuse | `refuse` | 资料不足、无关、冲突拒答 |
| format | `format` | 步骤、定义、故障排查等格式化输出 |

### 2.3 非目标

```text
1. 不做 QLoRA 训练；
2. 不合并 LoRA；
3. 不接 vLLM；
4. 不改 knowledge 主链路；
5. 不提交 sft_train.jsonl / sft_holdout.jsonl；
6. 不把 file/files 参考实现整包覆盖到当前项目。
```

## 3. 总体架构

```text
Milvus kb_chunks
  -> export_kb_chunks.py
  -> data/raw/kb_chunks.jsonl
  -> expand_dataset.py
       -> LocalRetriever / MilvusRetriever
       -> seed few-shot
       -> LLM question/answer generation
       -> inline validation
       -> stratified split
  -> data/processed/sft_train.jsonl
  -> data/processed/sft_holdout.jsonl
  -> validate_messages_dataset.py
  -> _messages_validation_report.md
```

阶段一 bootstrap 链路继续保留：

```text
build_sft_dataset.py -> validate_dataset.py -> convert_to_messages.py
```

阶段二新增正式造数链路：

```text
expand_dataset.py -> validate_messages_dataset.py
```

## 4. 核心模块

| 模块 | 职责 | 输入 | 输出 | 依赖 |
|---|---|---|---|---|
| `expand_dataset.py` | 强模型造数主脚本 | `kb_chunks.jsonl`、`seeds.jsonl`、config | `sft_train.jsonl`、`sft_holdout.jsonl` | OpenAI-compatible LLM，可选 Milvus |
| `LocalRetriever` | 离线关键词召回 | query、raw chunks | top-k chunks | 无 |
| `MilvusRetriever` | 生产向量召回 | query、Milvus collection | top-k chunks | pymilvus、BGE-M3 |
| `validate_messages_dataset.py` | messages 数据质量校验 | `sft_train.jsonl`、`sft_holdout.jsonl` | 校验报告 | `_common.py` |
| `seeds.jsonl` | few-shot 种子样本 | 产品说明书样例 | 生成提示示例 | 无 |
| `_common.py` | 公共工具增强 | config、jsonl、LLM | 通用函数 | pyyaml、openai |

## 5. 数据流设计

### 5.1 主流程

```text
读取 raw chunk
  -> 按 file_title / item_name 分组
  -> 读取产品说明书 seed
  -> 按配比规划 target 数量
  -> 按 type 构造样本
  -> 生成即校验
  -> 去重
  -> 分层切分 train / holdout
  -> 写入 messages JSONL
  -> 写入 _expand_stats.json
```

### 5.2 构造策略

| type | 构造方式 | 校验重点 |
|---|---|---|
| `faithful` | 单 chunk 生成问题和答案 | 不能拒答，必须有引用 |
| `multi_hop` | 同 item 采 2-3 个 chunk | 多片段整合，引用不越界 |
| `cite` | 单或多 chunk 强制引用 | 答案必须出现 `[1]` 等来源 |
| `format` | 识别步骤、定义、故障排查场景 | 结构化表达且保留引用 |
| `refuse/no_recall` | 问 A，给 B 上下文 | 必须拒答 |
| `refuse/weak_recall` | 给同主题但缺关键字段 | 必须说明缺少信息 |
| `refuse/conflict` | 合成数值冲突 | 必须指出冲突并拒答 |

### 5.3 异常流程

```text
LLM 返回空 -> 丢弃并计数
引用越界 -> 丢弃并计数
grounded 样本拒答 -> 丢弃并计数
refuse 样本未拒答 -> 丢弃并计数
样本过长 -> validate 阶段报告软告警
Milvus 检索失败 -> 使用 local dry-run 验证，不作为正式造数
```

## 6. 数据模型

阶段二直接输出 messages 格式：

```json
{
  "messages": [
    {"role": "system", "content": "你是掌柜智库的产品知识库问答助手..."},
    {"role": "user", "content": "【检索资料】\n[1] ...\n\n【问题】..."},
    {"role": "assistant", "content": "答案...[1]。"}
  ],
  "meta": {
    "id": "sft-000001",
    "type": "faithful",
    "subtype": "steps",
    "source": "万用表RS-12的使用",
    "synthetic": true,
    "battle_capabilities": ["faithful", "cite"]
  }
}
```

## 7. 配置设计

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `expand.total` | `300` | 阶段二第一轮造数目标 |
| `expand.holdout_ratio` | `0.12` | holdout 比例 |
| `expand.retriever` | `local` | 本地验证默认 local，正式可切 milvus |
| `expand.top_k` | `4` | 检索上下文数量 |
| `expand.ratios` | 五类配比 | 控制样本类型分布 |
| `expand.refuse_subtypes` | 三类拒答 | `no_recall`、`weak_recall`、`conflict` |
| `milvus.vector_field` | `dense_vector` | 当前项目 chunk collection 的稠密向量字段 |
| `milvus.metric_type` | `COSINE` | 当前项目 dense_vector 索引度量 |

## 8. 验收命令

本地离线验证：

```bash
python fine_tuning/scripts/expand_dataset.py --retriever local --dry-run
python fine_tuning/scripts/validate_messages_dataset.py
```

正式强模型造数：

```bash
cp fine_tuning/configs/config.example.yaml fine_tuning/configs/config.yaml
# 填写 llm.base_url / llm.api_key / llm.model
python fine_tuning/scripts/expand_dataset.py --retriever local
python fine_tuning/scripts/validate_messages_dataset.py
```

Milvus 召回验证：

```bash
python fine_tuning/scripts/expand_dataset.py --retriever milvus
```

注意：Milvus 召回依赖本机 BGE-M3 环境，字段必须使用 `dense_vector`。

## 9. 阶段二通过标准

```text
1. sft_train.jsonl / sft_holdout.jsonl 成功生成；
2. 五类 type 全覆盖：faithful / multi_hop / cite / refuse / format；
3. refuse 三子类全覆盖：no_recall / weak_recall / conflict；
4. validate_messages_dataset.py 硬错误 = 0；
5. meta 保留 battle_capabilities；
6. _expand_stats.json 记录 kept / dropped / by_type；
7. 正式数据完成至少 10% 人工抽检；
8. 真实 sft 数据不进入 Git。
```

## 10. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 强模型质量不稳 | 样本引用错、拒答失败 | 生成即校验 + 人工抽检 |
| local 召回太弱 | 与真实 RAG 噪声不一致 | 正式阶段切 MilvusRetriever |
| Milvus 字段不匹配 | 向量检索失败 | 使用 `dense_vector`，不使用参考实现默认 `embedding` |
| seeds 领域偏移 | 问答风格偏电商 | 使用产品说明书种子 |
| 样本比例不均 | 模型偏向作答或拒答 | `_expand_stats.json` 观察并调 ratios |
| 长上下文超限 | 训练截断或丢重点 | validate 阶段报告长度软告警 |

## 11. 开发里程碑

| 里程碑 | 交付物 | 验收 |
|---|---|---|
| M1 设计 | `stage2_execution_plan.md` | 目标、非目标、数据流清晰 |
| M2 造数脚本 | `expand_dataset.py` | local dry-run 可生成五类样本 |
| M3 校验脚本 | `validate_messages_dataset.py` | 硬错误可拦截 |
| M4 种子样本 | `data/seed/seeds.jsonl` | 产品说明书领域示例 |
| M5 配置 | `config.example.yaml` | 包含 expand/milvus 字段 |
| M6 验证 | dry-run + validate | 硬错误 0 |

