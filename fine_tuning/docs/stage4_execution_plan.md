# 阶段四执行计划 - Base vs SFT 离线评估

## 1. 一句话定位

阶段四负责证明阶段三训练出的 LoRA adapter 是否真的改善了 RAG 生成行为。

```text
阶段三产出 adapter。
阶段四用 holdout 对比 Base 与 SFT，判断 adapter 值不值得接回业务。
```

阶段四只做离线评估，不接线上，不改查询链路，不部署 vLLM。

## 2. 背景与目标

### 2.1 背景

阶段二已经生成：

```text
sft_train.jsonl
sft_holdout.jsonl
```

阶段三计划输出：

```text
fine_tuning/outputs/kb-sft
```

阶段四使用 `sft_holdout.jsonl` 作为固定评估集，对比：

```text
Base model
Base model + LoRA adapter
```

### 2.2 阶段四目标

```text
1. 提供 Base vs SFT 离线评估脚本；
2. 重点评估拒答、引用、忠实性和完整性；
3. 输出 before/after 指标表；
4. 输出 bad case 明细；
5. 支持 stub 模式在本地验证指标计算；
6. 支持可选 LLM judge 评估 faithfulness / completeness。
```

### 2.3 非目标

```text
1. 不训练模型；
2. 不合并 LoRA；
3. 不接 vLLM；
4. 不改 RAG 查询链路；
5. 不做线上 A/B；
6. 不把评估产物提交 Git。
```

## 3. 总体架构

```text
sft_holdout.jsonl
  -> eval_before_after.py
       -> BaseGenerator
       -> SFTGenerator
       -> metrics
       -> bad case classifier
  -> data/eval/eval_report.md
  -> data/eval/_eval_metrics.json
  -> data/eval/bad_cases.jsonl
  -> data/eval/predictions.jsonl
```

本地验证路径：

```text
sft_holdout.jsonl
  -> stub base: 永远作答
  -> stub sft: 复现 gold
  -> 验证指标计算和报告渲染
```

## 4. 指标设计

| 指标 | 含义 | 期望 |
|---|---|---|
| `refusal_recall` | 应拒答样本中真正拒答的比例 | SFT 上升 |
| `refusal_precision` | 模型拒答中确实该拒答的比例 | SFT 不下降 |
| `refusal_f1` | 拒答 precision/recall 综合 | SFT 上升 |
| `false_refusal` | 可回答样本被误拒的比例 | SFT 不升高 |
| `citation_validity` | 可回答样本引用存在且不越界 | SFT 上升 |
| `recall_no_recall` | 无召回拒答 recall | SFT 上升 |
| `recall_weak_recall` | 弱召回拒答 recall | SFT 上升 |
| `recall_conflict` | 冲突资料拒答 recall | SFT 上升 |
| `faithfulness` | 可回答答案是否由上下文支撑 | 可选 judge |
| `completeness` | 答案是否覆盖关键条件 | 可选 judge |

## 5. Bad Case 分类

| 类型 | 触发条件 | 说明 |
|---|---|---|
| `missed_refusal` | 应拒答但模型作答 | 资料不足仍硬答 |
| `false_refusal` | 应作答但模型拒答 | 误拒 |
| `citation_invalid` | 应作答且未拒答，但引用缺失或越界 | 引用不可信 |
| `unfaithful` | judge 判定不忠实 | 答案无依据 |
| `incomplete` | judge 判定不完整 | 漏关键条件 |

## 6. 数据模型

输入沿用阶段二 messages schema：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "【检索资料】...【问题】..."},
    {"role": "assistant", "content": "gold answer"}
  ],
  "meta": {
    "id": "sft-holdout-000001",
    "type": "refuse",
    "subtype": "no_recall"
  }
}
```

输出预测记录：

```json
{
  "id": "sft-holdout-000001",
  "model": "base",
  "type": "refuse",
  "prediction": "...",
  "gold": "...",
  "should_refuse": true,
  "did_refuse": false,
  "bad_case": "missed_refusal"
}
```

## 7. 验收命令

本地 stub 验证：

```bash
python fine_tuning/scripts/eval_before_after.py --stub
```

真实模型评估：

```bash
python fine_tuning/scripts/eval_before_after.py \
  --base Qwen/Qwen2.5-3B-Instruct \
  --adapter fine_tuning/outputs/kb-sft
```

可选 judge：

```bash
python fine_tuning/scripts/eval_before_after.py \
  --base Qwen/Qwen2.5-3B-Instruct \
  --adapter fine_tuning/outputs/kb-sft \
  --judge
```

## 8. 阶段四通过标准

```text
1. --stub 模式可生成 eval_report.md；
2. _eval_metrics.json 包含 base / sft 指标；
3. bad_cases.jsonl 能输出问题样本；
4. 真实模型评估可加载 base + adapter；
5. 评估产物写入 fine_tuning/data/eval/，不进入 Git；
6. 只有 SFT 指标确实改善，才进入阶段五接入。
```

## 9. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| holdout 来自 dry-run | 指标无意义 | 正式评估前必须用强模型正式造数 |
| SFT 过度拒答 | false_refusal 升高 | 调拒答样本比例、回退 adapter |
| 引用有效但事实不忠实 | citation_validity 虚高 | 增加 LLM judge 和人工 bad case |
| judge 不稳定 | 指标波动 | 固定 judge 模型和 prompt，保留人工抽检 |
| 评估集太小 | 结论不稳 | holdout 至少 100-150 条 |

## 10. 阶段五入口条件

```text
1. SFT refusal_recall 明显高于 Base；
2. false_refusal 没有明显上升；
3. citation_validity 不下降；
4. 核心 bad case 有人工复核；
5. adapter 路径、评估报告、回滚方式明确。
```

