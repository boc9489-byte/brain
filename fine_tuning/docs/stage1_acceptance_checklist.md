# 阶段一验收清单 - fine_tuning 数据闭环

## 1. 验收范围

阶段一验收只覆盖数据工程闭环：

```text
Milvus chunk
  -> kb_chunks.jsonl
  -> train / holdout
  -> validation report
  -> messages_train / messages_holdout
```

不验收训练效果、不验收 vLLM 部署、不验收线上查询效果。

## 2. 环境验收

| 检查项 | 标准 | 状态 |
|---|---|---|
| uv 环境 | `.venv` 可运行 fine_tuning 脚本 | 已通过 |
| Milvus 地址 | `MILVUS_URL` 可访问 | 已通过：`http://10.211.55.4:19530` |
| collection | `CHUNKS_COLLECTION` 指向 `kb_chunks` 或实际 chunk collection | 已通过：`kb_chunks` |
| 配置文件 | `config.example.yaml` 不含密钥 | 已通过 |
| 本地配置 | `config.yaml` 被 `.gitignore` 忽略 | 已通过 |

## 3. 脚本验收

| 脚本 | 命令 | 验收标准 | 状态 |
|---|---|---|---|
| 导出 chunk | `uv run python fine_tuning/scripts/export_kb_chunks.py` | exported > 0 | 已通过 |
| 构造样本 | `uv run python fine_tuning/scripts/build_sft_dataset.py --dry-run` | actual_total > 0 | 已通过：actual_total=59 |
| 校验数据 | `uv run python fine_tuning/scripts/validate_dataset.py` | 硬错误 = 0 | 已通过：硬错误 0 |
| 转 messages | `uv run python fine_tuning/scripts/convert_to_messages.py` | 生成 messages_train / messages_holdout | 已通过：49 / 10 |

## 4. 数据产物验收

| 文件 | 是否应生成 | 是否可提交 | 状态 |
|---|---|---|---|
| `fine_tuning/data/raw/kb_chunks.jsonl` | 是 | 否 | 已生成 |
| `fine_tuning/data/raw/_export_stats.json` | 是 | 否 | 已生成 |
| `fine_tuning/data/processed/train.jsonl` | 是 | 否 | 已生成 |
| `fine_tuning/data/processed/holdout.jsonl` | 是 | 否 | 已生成 |
| `fine_tuning/data/processed/_build_stats.json` | 是 | 否 | 已生成 |
| `fine_tuning/data/processed/_validation_report.md` | 是 | 否 | 已生成 |
| `fine_tuning/data/processed/messages_train.jsonl` | 是 | 否 | 已生成 |
| `fine_tuning/data/processed/messages_holdout.jsonl` | 是 | 否 | 已生成 |
| `fine_tuning/data/processed/_messages_stats.json` | 是 | 否 | 已生成 |

## 5. 数据质量门禁

| 门禁 | 标准 | 状态 |
|---|---|---|
| 导出数量 | `exported > 0` | 已通过：130 |
| 空内容 | `empty_content_skipped = 0` 或占比很低 | 已通过：0 |
| item 覆盖 | `distinct_items >= 2` | 已通过：3 |
| 样本总数 | `actual_total > 0` | 已通过：59 |
| 类型覆盖 | 至少覆盖 answerable / multi_chunk / unanswerable | 已通过：四类均覆盖 |
| 校验硬错误 | `硬错误: 0` | 已通过 |
| messages 格式 | 每条包含 system / user / assistant | 已通过 |
| 能力映射 | meta 包含 `battle_capabilities` | 已通过 |

## 6. Git 安全门禁

提交前必须确认以下文件不会进入 Git：

```text
fine_tuning/configs/config.yaml
fine_tuning/data/raw/kb_chunks.jsonl
fine_tuning/data/raw/_export_stats.json
fine_tuning/data/processed/*.jsonl
fine_tuning/data/processed/*.json
fine_tuning/data/processed/*.md
fine_tuning/outputs/
*.safetensors
checkpoint-*/
```

检查命令：

```bash
git status --short --ignored fine_tuning .gitignore
git ls-files --others --exclude-standard fine_tuning
```

可提交范围建议：

```bash
git add .gitignore fine_tuning/README.md fine_tuning/configs/config.example.yaml fine_tuning/scripts fine_tuning/docs fine_tuning/data/raw/.gitkeep fine_tuning/data/processed/.gitkeep fine_tuning/eval/.gitkeep fine_tuning/train/.gitkeep fine_tuning/screenshots/.gitkeep
```

## 7. 阶段一通过标准

阶段一验收通过需要同时满足：

```text
1. export_kb_chunks.py 成功导出真实 chunk；
2. build_sft_dataset.py --dry-run 成功生成 train / holdout；
3. validate_dataset.py 报告硬错误为 0；
4. convert_to_messages.py 成功生成 messages_train / messages_holdout；
5. messages 样本包含 battle_capabilities；
6. 真实 jsonl / config.yaml / 模型产物未进入 Git；
7. 代码阅读、执行计划、工程汇报、测试记录、企业工程总方案齐全。
```

当前结论：

```text
阶段一数据闭环已通过。

保留风险：
1. 当前为 stub dry-run 数据，只能验证工程流程，不能直接训练；
2. multi_chunk 和 unanswerable 各 3 条，样本比例偏低；
3. 1 条样本 rendered length 超过 max_chars，需要阶段二正式造数时控制长度。
```

## 8. 未通过时的处理

| 失败点 | 排查方向 |
|---|---|
| Milvus 连接失败 | 检查 `MILVUS_URL`、容器状态、端口映射、pymilvus 版本 |
| 导出数量为 0 | 检查 collection 名称和字段映射 |
| empty_content 过高 | 回查导入链路的 Markdown 解析和 chunk 切分 |
| unanswerable 不足 | 导入更多不同 item 文档 |
| conflicting 不足 | chunk 中缺少可修改数字，后续可增强合成策略 |
| 引用越界 | 检查样本生成 prompt 和 validate 规则 |
| train / holdout 重复 | 检查 fingerprint 去重逻辑 |
| messages 缺字段 | 检查 `convert_to_messages.py` |

## 9. 下一阶段入口条件

进入阶段二“强模型正式造数”前必须满足：

```text
1. 阶段一通过；
2. 至少 3 个 item 已导入；
3. dry-run 样本校验硬错误为 0；
4. format 样本设计已明确；
5. 本地数据产物确认不会提交 Git。
```
