# 阶段一测试记录 - fine_tuning 数据闭环

## 1. 测试目标

验证 `fine_tuning` 阶段一是否能够从真实 Milvus 知识库导出 chunk，并继续构造、校验、转换 SFT 数据。

阶段一测试链路：

```text
export_kb_chunks.py
  -> build_sft_dataset.py --dry-run
  -> validate_dataset.py
  -> convert_to_messages.py
```

## 2. 测试环境

```text
项目路径：/Users/bob/PycharmProjects/shopkeeper_brain
uv 环境：.venv
Milvus 地址：http://10.211.55.4:19530
Milvus collection：kb_chunks
配置来源：fine_tuning/configs/config.example.yaml + knowledge/.env 环境变量兜底
```

启动环境：

```bash
cd /Users/bob/PycharmProjects/shopkeeper_brain
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r fine_tuning/requirements-runtime.txt
```

## 3. 测试用例 1：导出真实 Milvus chunk

测试命令：

```bash
uv run python fine_tuning/scripts/export_kb_chunks.py
```

实际输出：

```text
[config] fine_tuning/configs/config.yaml not found, using config.example.yaml with environment fallbacks.
[export] config loaded: uri=http://10.211.55.4:19530 collection=kb_chunks
[export] connecting Milvus...
[export] connected. collection=kb_chunks fields=['chunk_id', 'content', 'title', 'parent_title', 'file_title', 'item_name']
[export] exported=130 -> /Users/bob/PycharmProjects/shopkeeper_brain/fine_tuning/data/raw/kb_chunks.jsonl
[export] empty_content_skipped=0; distinct_items=3
```

结论：

```text
通过
```

说明：

```text
Milvus gRPC 连接已恢复
kb_chunks collection 可查询
字段映射正确
导出 chunk 数：130
空 content 数：0
覆盖 item 数：3
```

验收重点：

```text
1. exported > 0
2. empty_content_skipped = 0 或占比很低
3. distinct_items >= 2，便于构造 unanswerable 样本
```

## 4. 测试用例 2：构造 dry-run SFT 数据

测试命令：

```bash
uv run python fine_tuning/scripts/build_sft_dataset.py --dry-run
```

实际输出：

```text
[config] fine_tuning/configs/config.yaml not found, using config.example.yaml with environment fallbacks.
[build] mode=stub chunks=130 items=3
[build] {'requested_total': 120, 'actual_total': 59, 'train': 49, 'holdout': 10, 'by_type': {'answerable': 35, 'conflicting': 18, 'unanswerable': 3, 'multi_chunk': 3}, 'dropped_duplicates': 7, 'synthetic': 59, 'mode': 'stub'}
[build] stub data is only for pipeline verification. Use a strong model before training.
```

预期产物：

```text
fine_tuning/data/processed/train.jsonl
fine_tuning/data/processed/holdout.jsonl
fine_tuning/data/processed/_build_stats.json
```

结论：

```text
通过
```

说明：

```text
已基于真实导出的 130 条 chunk 构造 dry-run 样本。
构造阶段发现并丢弃 7 条重复样本，避免 train / holdout 泄漏。
当前 dry-run 样本数为 59，其中 train=49，holdout=10。
```

验收重点：

```text
1. mode = stub
2. actual_total > 0
3. by_type 至少包含 answerable / multi_chunk / unanswerable
4. conflicting 数量取决于 chunk 中是否存在可修改数字
```

注意：

```text
--dry-run 只验证流水线，不能直接作为正式训练数据。
正式训练前要配置强模型，重新生成高质量 SFT 样本。
```

## 5. 测试用例 3：校验数据质量

测试命令：

```bash
uv run python fine_tuning/scripts/validate_dataset.py
```

预期产物：

```text
fine_tuning/data/processed/_validation_report.md
```

查看报告：

```bash
cat fine_tuning/data/processed/_validation_report.md
```

实际输出摘要：

```text
- train=49 holdout=10 total=59
- 类型分布: answerable=35(59%), conflicting=18(31%), multi_chunk=3(5%), unanswerable=3(5%)
- 硬错误: 0
- 软告警: 5
```

结论：

```text
通过
```

验收重点：

```text
硬错误: 0
```

硬错误包括：

```text
引用编号越界
可回答样本没有引用
拒答样本没有拒答
conflicting 样本没有指出冲突
train / holdout 重复
非法 type 或缺字段
```

软告警可以暂不阻塞，但需要记录原因。

本次软告警：

```text
1. 1 条样本 rendered length=4003，超过 max_chars=3000；
2. answerable / conflicting / multi_chunk / unanswerable 类型比例偏离目标比例。
```

原因说明：

```text
当前是 stub dry-run，小样本下类型分布受真实 chunk 数量、item 分组和可合成冲突数字影响。
这不阻塞阶段一数据闭环，但阶段二正式造数前要补强模型生成和人工抽检。
```

## 6. 测试用例 4：转换 messages 训练格式

测试命令：

```bash
uv run python fine_tuning/scripts/convert_to_messages.py
```

实际输出：

```text
[config] fine_tuning/configs/config.yaml not found, using config.example.yaml with environment fallbacks.
[convert] /Users/bob/PycharmProjects/shopkeeper_brain/fine_tuning/data/processed/train.jsonl -> /Users/bob/PycharmProjects/shopkeeper_brain/fine_tuning/data/processed/messages_train.jsonl rows=49
[convert] /Users/bob/PycharmProjects/shopkeeper_brain/fine_tuning/data/processed/holdout.jsonl -> /Users/bob/PycharmProjects/shopkeeper_brain/fine_tuning/data/processed/messages_holdout.jsonl rows=10
```

预期产物：

```text
fine_tuning/data/processed/messages_train.jsonl
fine_tuning/data/processed/messages_holdout.jsonl
fine_tuning/data/processed/_messages_stats.json
```

结论：

```text
通过
```

messages 统计：

```text
messages_train=49
messages_holdout=10
battle_capability_map 已写入 _messages_stats.json
```

抽查命令：

```bash
head -n 1 fine_tuning/data/processed/messages_train.jsonl
cat fine_tuning/data/processed/_messages_stats.json
```

验收重点：

```text
1. 每条样本包含 system / user / assistant 三段 messages
2. user 中包含【检索资料】和【问题】
3. assistant 中是标准答案
4. meta 中保留 type 和 battle_capabilities
```

## 7. 当前测试状态

| 测试项 | 状态 | 说明 |
|---|---|---|
| export_kb_chunks.py | 通过 | 真实 Milvus 导出 130 条 chunk |
| build_sft_dataset.py --dry-run | 通过 | 构造 59 条 stub 样本，丢弃 7 条重复样本 |
| validate_dataset.py | 通过 | 硬错误 0，软告警 5 |
| convert_to_messages.py | 通过 | 生成 messages_train=49、messages_holdout=10 |

## 8. 后续测试建议

```text
1. 阶段一数据闭环已通过，可以继续人工抽查 messages_train.jsonl
2. 如果样本类型分布偏斜，检查 _build_stats.json
3. 如果 unanswerable / multi_chunk 不足，继续导入更多不同商品文档
4. 如果 dry-run 通过，再进入强模型正式造数阶段
5. 阶段二前必须记住：当前 stub 数据不能直接训练
```

## 9. Git 注意事项

以下文件是测试产物，不应提交：

```text
fine_tuning/data/raw/kb_chunks.jsonl
fine_tuning/data/raw/_export_stats.json
fine_tuning/data/processed/*.jsonl
fine_tuning/data/processed/*.json
fine_tuning/data/processed/*.md
fine_tuning/configs/config.yaml
```

这些已由 `.gitignore` 保护。
