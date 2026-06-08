# 阶段八执行计划 - Query Intent Analyzer 与检索路由优化

## 1. 一句话定位

阶段八负责把当前轻量 `intent_type` 能力升级为 RAG 查询入口的 Query Analyzer。

```text
阶段六/七解决“回答质量如何观测和复盘”。
阶段八解决“用户问题如何被理解、确认商品、澄清歧义，并路由到更合适的检索策略”。
```

阶段八不是替代 RAG 主链路，而是在查询入口把自然语言问题结构化，降低后续召回、融合、重排和回答生成跑偏的概率。

## 2. 背景与目标

### 2.1 背景

当前查询链路已经补充了轻量意图识别：

```text
1. 查询 prompt 要求上游 LLM 输出 intent_type；
2. 回答节点优先使用 state.intent_type；
3. 缺失或非法时使用本地规则分类兜底；
4. intent_type 写入回答 prompt 和 answer trace；
5. trace 可按 intent_type 分桶分析 bad case。
```

这解决了“意图可见、可观测、可测试”的问题，但还没有形成完整 Query Analyzer：

```text
1. 没有独立 item_name_confirmed_node；
2. 查询阶段没有把 LLM 候选商品名与 Milvus 商品名集合做向量对齐；
3. 没有高/中/低置信度阈值过滤；
4. 没有多候选澄清流程；
5. intent_type 还没有驱动 hybrid / HyDE / web / image-aware 检索路由。
```

### 2.2 阶段八目标

```text
1. 新增 QueryIntentAnalyzer，统一负责查询理解；
2. 从 original_query + history 中抽取候选商品名和 intent_type；
3. 用 Milvus 商品名集合对齐真实商品名；
4. 基于 high / mid / score_gap 阈值确认商品或触发澄清；
5. 生成 rewritten_query 和 retrieval_strategy；
6. 将结构化结果写入 QueryGraphState；
7. 按 intent_type 调整检索策略和上下文排序；
8. 扩展 trace 和 bad case 挖掘字段，支持按意图、商品确认状态、路由策略复盘。
```

### 2.3 非目标

```text
1. 不重写文档导入链路；
2. 不改 Milvus chunk 入库 schema 的已有字段含义；
3. 不在第一版引入复杂训练模型分类器；
4. 不让 LLM 候选商品名直接作为最终商品名；
5. 不自动把线上 bad case 写入训练集；
6. 不强制所有查询都走 WebSearch。
```

## 3. 前后对比

| 维度 | 当前轻量实现 | 阶段八目标实现 |
|---|---|---|
| 意图识别形态 | `intent_type` 枚举 + 规则兜底 | Query Analyzer 结构化查询理解 |
| 商品名确认 | 主要依赖 prompt 结果 | LLM 候选 + Milvus 对齐 + 阈值过滤 |
| 歧义处理 | 未独立实现 | 多候选分数接近时触发澄清 |
| 查询改写 | `rewritten_query` 由 prompt 输出 | 结合商品确认、历史回填和意图生成 |
| 检索路由 | intent 仅进入 prompt / trace | intent 驱动 hybrid / HyDE / web / image-aware 策略 |
| 观测字段 | `intent_type` | intent、商品置信度、候选项、澄清状态、路由策略 |
| 测试重点 | 分类规则与 trace 字段 | 商品确认、澄清、路由、召回命中率 |

## 4. 总体架构

```text
用户输入 original_query
  -> load_history
  -> item_name_confirmed_node / QueryIntentAnalyzer
       -> LLM 提取 candidate_items / intent_type / rewritten_query
       -> Milvus item_name_collection 向量对齐
       -> 阈值过滤与 score_gap 判断
       -> 历史回填与指代消解
       -> need_clarification 判断
  -> conditional route
       -> need_clarification=true
            -> clarification_answer_node
            -> END
       -> need_clarification=false
            -> multi_search
                 -> hybrid_vector_search
                 -> hyde_vector_search
                 -> optional web_search
                 -> optional image-aware boost
            -> rrf_node
            -> reranker_node
            -> answer_output_node
            -> record_answer_trace
```

## 5. 核心模块

| 模块 | 职责 | 输入 | 输出 | 依赖 |
|---|---|---|---|---|
| `QueryIntentAnalyzer` | 查询理解编排 | query / history | `QueryIntentResult` | LLM / Milvus / config |
| `llm_extract_query_intent` | 抽取候选商品、意图、改写问题 | query / history | candidate_items / intent_type / rewritten_query | ChatOpenAI |
| `item_name_alignment` | 对齐真实商品名 | candidate_items | aligned_items / scores | Milvus / BGE-M3 |
| `item_confidence_filter` | 阈值过滤和候选排序 | aligned_items | confirmed_items / options | config |
| `clarification_router` | 判断是否需要澄清 | confirmed / options / score_gap | route | LangGraph |
| `retrieval_strategy_planner` | 生成检索策略 | intent / item_names | strategy list | config |
| `answer_trace` 扩展 | 记录查询理解结果 | state | JSONL trace | file system |

## 6. 状态模型设计

建议扩展查询状态字段：

```python
class QueryGraphState(TypedDict, total=False):
    session_id: str
    message_id: str
    task_id: str
    original_query: str
    query: str
    is_stream: bool
    history: list

    intent_type: str
    query_type: str
    item_names: list[str]
    candidate_items: list[dict]
    need_clarification: bool
    clarification_options: list[str]
    rewritten_query: str
    retrieval_strategy: list[str]
    intent_confidence: float

    embedding_chunks: list
    hyde_embedding_chunks: list
    web_search_docs: list
    rrf_chunks: list
    reranked_docs: list
    answer: str
```

核心字段说明：

| 字段 | 说明 |
|---|---|
| `intent_type` | 操作、参数、故障、安装、图片、对比、售后、通用 |
| `query_type` | `knowledge_qa` / `clarification` / `fallback` |
| `item_names` | 最终确认后的真实商品名 |
| `candidate_items` | LLM 和 Milvus 对齐后的候选商品及分数 |
| `need_clarification` | 是否需要用户澄清 |
| `clarification_options` | 返回给用户选择的候选商品 |
| `retrieval_strategy` | 本轮应执行的检索策略 |
| `intent_confidence` | 查询理解总体置信度 |

## 7. 商品名确认策略

### 7.1 LLM 候选抽取

LLM 只负责候选提取，不直接决定最终商品名：

```json
{
  "candidate_items": ["万用表", "RS-12"],
  "intent_type": "operation",
  "rewritten_query": "RS-12 数字万用表如何测量电阻"
}
```

提示词约束：

```text
1. 不要把操作短语当商品名；
2. 优先提取品牌 + 型号；
3. 允许空 candidate_items；
4. 多轮代词要结合 history 回填；
5. 只输出 JSON。
```

### 7.2 Milvus 商品名对齐

```text
candidate item
  -> BGE-M3 embedding
  -> item_name_collection hybrid search
  -> topK real item names with score
```

对齐结果示例：

```json
[
  {"item_name": "RS-12 数字万用表", "score": 0.86, "source": "milvus"},
  {"item_name": "UT890D+ 数字万用表", "score": 0.62, "source": "milvus"}
]
```

### 7.3 阈值过滤

建议配置：

```text
ITEM_NAME_HIGH_CONFIDENCE=0.75
ITEM_NAME_MID_CONFIDENCE=0.45
ITEM_NAME_SCORE_GAP=0.08
ITEM_NAME_MAX_OPTIONS=3
```

| 分数情况 | 动作 |
|---|---|
| top1 >= high | 直接确认 top1 |
| top1 >= mid 且 top1 - top2 >= gap | 确认 top1 |
| 多个候选 >= mid 且分数接近 | 触发澄清 |
| 全部 < mid | 不加商品过滤，保留通用检索 |

## 8. 检索路由策略

| `intent_type` | 检索策略 | 上下文偏好 |
|---|---|---|
| `operation` | hybrid + HyDE | 操作步骤、使用说明 |
| `install_config` | hybrid + HyDE | 安装、接线、配置、初始化 |
| `troubleshooting` | hybrid + HyDE | 报错、故障、排障、异常处理 |
| `parameter` | hybrid | 参数表、规格、电压、电流、接口 |
| `image_request` | hybrid + image-aware boost | 带图片 URL、结构图、接线图 |
| `comparison` | multi-item hybrid | 多商品并行召回、差异字段 |
| `after_sales` | hybrid | 售后、保修、维修、退换 |
| `general` | hybrid + optional HyDE | 默认知识问答 |

第一版不需要实现复杂学习型路由，可以先基于配置规则完成。

## 9. Trace 与 Bad Case 扩展

建议 answer trace 增加字段：

```json
{
  "intent_type": "operation",
  "query_type": "knowledge_qa",
  "item_names": ["RS-12 数字万用表"],
  "candidate_item_count": 2,
  "need_clarification": false,
  "retrieval_strategy": ["hybrid_vector", "hyde_vector"],
  "intent_confidence": 0.86
}
```

阶段七 bad case 挖掘可以新增分桶：

| 类型 | 触发规则 | 典型根因 |
|---|---|---|
| `intent_unknown_high_freq` | `intent_type=general` 频繁出现 | 意图规则或 prompt 覆盖不足 |
| `clarification_missing` | 多候选接近但未澄清 | score_gap 判断缺失 |
| `item_alignment_low_confidence` | 商品对齐长期低分 | 商品名库不全或 LLM 候选差 |
| `image_intent_no_image_context` | 图片意图但上下文无图片 | 图片 chunk 召回不足 |
| `parameter_intent_no_spec_context` | 参数意图但未召回规格内容 | 参数表解析或召回策略不足 |

## 10. 测试与验收

### 10.1 单元测试

```text
1. intent_type 规则分类；
2. LLM JSON 解析失败兜底；
3. Milvus 对齐结果阈值过滤；
4. 多候选 score_gap 澄清判断；
5. retrieval_strategy 生成；
6. trace 字段脱敏和写入。
```

### 10.2 集成测试

```text
1. “RS-12 怎么测电阻” -> operation + RS-12 + hybrid/hyde；
2. “这个怎么接线” + history -> install_config + 历史商品回填；
3. “A 和 B 有什么区别” -> comparison + 多商品检索；
4. “有没有接线图” -> image_request + 图片 chunk 优先；
5. 多个商品候选接近 -> clarification answer；
6. 商品名低置信 -> 不强行过滤，走通用知识检索。
```

### 10.3 指标

| 指标 | 目标 |
|---|---:|
| `intent_accuracy` | >= 0.85 |
| `item_match_accuracy` | >= 0.80 |
| `clarification_precision` | >= 0.80 |
| `retrieval_hit_rate@5` | 较当前基线提升 |
| `no_context_answered` | 较当前基线下降 |
| `image_request_hit_image_context` | >= 0.70 |

## 11. 开发里程碑

| 阶段 | 目标 | 交付物 | 验收 |
|---|---|---|---|
| 8.1 | 抽象 QueryIntentResult 和配置 | schema / config / 单测 | 类型和阈值测试通过 |
| 8.2 | 实现 LLM 候选抽取 | prompt / parser / fallback | JSON 解析失败可兜底 |
| 8.3 | 实现 Milvus 商品名对齐 | alignment service | 阈值过滤测试通过 |
| 8.4 | 接入 LangGraph 查询入口 | item_name_confirmed_node | 澄清/继续检索路由正确 |
| 8.5 | 按 intent 调整检索策略 | strategy planner | 不同意图策略可观测 |
| 8.6 | 扩展 trace 和 bad case | trace fields / mining rules | 可按意图分桶复盘 |
| 8.7 | 回归评估 | Golden Set / 报告 | 关键指标不退化 |

## 12. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| LLM 抽取商品名误判 | 商品过滤错，召回跑偏 | 只作为候选，必须经过 Milvus 对齐 |
| 商品名集合不全 | 正确商品无法确认 | 低置信时不强行过滤，并记录 bad case |
| 澄清过多 | 用户体验变差 | 只在多个候选分数接近时澄清 |
| 规则路由过硬 | 某些问题召回变差 | 保留默认 hybrid 兜底 |
| trace 泄露业务文本 | 合规风险 | 默认只记录 hash 和结构化字段 |
| 阈值不适配数据 | 误确认或误澄清 | 将阈值配置化，用 Golden Set 调参 |

## 13. 面试表达

```text
我们把意图识别设计成 RAG 查询入口的 Query Analyzer，而不是简单文本分类器。
用户问题进入系统后，先由 LLM 从 query 和 history 中提取候选商品名、intent_type 和 rewritten_query。
候选商品名不会直接使用，而是送到 Milvus 商品名集合做向量对齐，再结合 high/mid 阈值和 score_gap 判断是否确认商品或触发澄清。
确认后的 item_names、rewritten_query、intent_type 和 retrieval_strategy 会写入 QueryGraphState，后续驱动 hybrid 检索、HyDE、图片优先召回、RRF、Reranker 和答案生成。
这个设计同时利用了 LLM 的语义理解能力和向量库的实体约束能力，可以降低商品误判导致的整条 RAG 链路跑偏。
```

## 14. 一句话总结

```text
阶段八的核心是把“意图标签”升级为“查询理解与路由结果”：LLM 做语义候选，Milvus 做实体对齐，阈值和澄清保证可靠性，最终把结构化状态交给多路检索和回答链路。
```
