# 发布验收清单 - 掌柜智库 SFT

## 1. Git 检查

```bash
git status --short
git log --oneline --decorate -5
git remote -v
```

通过标准：

```text
1. 工作区干净；
2. 当前分支为 main 或发布分支；
3. 最新提交已推送到远程；
4. 不包含 data、outputs、.env、checkpoint。
```

## 2. 本地静态检查

```bash
uv run python -m py_compile \
  knowledge/utils/client/answer_model_config.py \
  knowledge/utils/client/ai_clients.py \
  knowledge/utils/observability/answer_trace.py \
  knowledge/processor/query_processor/nodes/answer_output_node.py \
  fine_tuning/scripts/check_stage5_serving.py \
  fine_tuning/scripts/check_stage6_observability.py \
  fine_tuning/scripts/mine_stage7_bad_cases.py
```

通过标准：

```text
命令退出码为 0。
```

## 3. fine_tuning 单测

```bash
uv run python fine_tuning/tests/test_stage5_answer_config.py
uv run python fine_tuning/tests/test_stage6_answer_trace.py
uv run python fine_tuning/tests/test_stage7_bad_case_mining.py
```

通过标准：

```text
[ok] stage5 answer model config
[ok] stage6 answer trace
[ok] stage7 bad case mining
```

## 4. GPU 服务检查

```bash
nvidia-smi
curl http://127.0.0.1:8000/v1/models
```

通过标准：

```text
1. GPU 可见；
2. vLLM 进程存在；
3. /v1/models 返回 base model；
4. 如果启用 SFT，/v1/models 返回 kb-sft。
```

## 5. 业务配置检查

```bash
uv run python fine_tuning/scripts/check_stage5_serving.py --check-only
uv run python fine_tuning/scripts/check_stage5_serving.py --health
uv run python fine_tuning/scripts/check_stage6_observability.py --check-only
```

通过标准：

```text
1. provider 显示为预期的 base 或 sft；
2. base_url 指向 GPU vLLM；
3. model_name 正确；
4. trace 默认脱敏；
5. health 检查通过。
```

## 6. 业务服务检查

```bash
curl http://127.0.0.1:8000/hello
```

通过标准：

```text
返回 {"flag":"success"}。
```

前端检查：

```text
http://SERVER_IP:8000/front/import.html
```

通过标准：

```text
页面可打开，文件可导入，状态可查询。
```

## 7. Base / SFT 切换验收

Base：

```bash
ANSWER_MODEL_PROVIDER=base
uv run python fine_tuning/scripts/check_stage5_serving.py --check-only
```

SFT：

```bash
ANSWER_MODEL_PROVIDER=sft
uv run python fine_tuning/scripts/check_stage5_serving.py --check-only
```

通过标准：

```text
1. base 模式 model_name 为 ANSWER_BASE_MODEL；
2. sft 模式 model_name 为 ANSWER_SFT_MODEL；
3. SFT 只在阶段四真实评估通过后默认开启。
```

## 8. 线上观测验收

写入 sample trace：

```bash
ANSWER_TRACE_ENABLED=true \
ANSWER_TRACE_PATH=/private/tmp/stage6_answer_traces.jsonl \
uv run python fine_tuning/scripts/check_stage6_observability.py --write-sample
```

通过标准：

```text
1. 能写入 JSONL；
2. 默认不包含完整 query / answer；
3. 包含 provider / model / latency / citation / refusal。
```

## 9. Bad Case 挖掘验收

```bash
uv run python fine_tuning/scripts/mine_stage7_bad_cases.py --sample
```

通过标准：

```text
1. bad_cases.jsonl 生成；
2. golden_candidates.jsonl 生成；
3. _stage7_bad_case_report.md 生成；
4. 输出目录 fine_tuning/data/online/ 被 Git 忽略。
```

## 10. 回滚验收

回滚命令：

```bash
ANSWER_MODEL_PROVIDER=base
```

通过标准：

```text
1. 重启业务服务后恢复 base；
2. check_stage5_serving.py 显示 provider=base；
3. trace 继续记录；
4. 保留 SFT bad case 供复盘。
```
