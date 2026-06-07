# 阶段五测试记录 - vLLM LoRA 接回业务

## 1. 测试目标

验证阶段五新增的回答模型切换层是否能在本地完成静态检查：

```text
环境变量
  -> AnswerModelSettings
  -> AIClients.get_answer_llm_client
  -> AnswerOutPutNode
  -> check_stage5_serving.py
```

本次测试不启动 vLLM，不访问真实模型服务。

## 2. 测试环境

```text
项目路径：/Users/bob/PycharmProjects/shopkeeper_brain
uv 环境：.venv
默认 provider：base
SFT alias：kb-sft
```

## 3. 已执行命令

语法检查：

```bash
uv run python -m py_compile \
  knowledge/utils/client/answer_model_config.py \
  knowledge/utils/client/ai_clients.py \
  knowledge/processor/query_processor/nodes/answer_output_node.py \
  fine_tuning/scripts/check_stage5_serving.py \
  fine_tuning/tests/test_stage5_answer_config.py
```

实际结果：

```text
通过
```

配置单测：

```bash
uv run python fine_tuning/tests/test_stage5_answer_config.py
```

实际输出：

```text
[ok] stage5 answer model config
```

默认 base 检查：

```bash
uv run python fine_tuning/scripts/check_stage5_serving.py --check-only
```

实际输出：

```text
[stage5] answer model settings:
{
  "provider": "base",
  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "model_name": "qwen-flash",
  "base_model": "qwen-flash",
  "sft_model": "kb-sft",
  "temperature": 0.0,
  "max_tokens": 1024,
  "timeout_sec": 60,
  "api_key_set": true
}
[stage5] check passed
```

SFT 配置检查：

```bash
ANSWER_MODEL_PROVIDER=sft \
ANSWER_OPENAI_API_BASE=http://127.0.0.1:8000/v1 \
ANSWER_SFT_MODEL=kb-sft \
uv run python fine_tuning/scripts/check_stage5_serving.py --check-only
```

实际输出：

```text
[stage5] answer model settings:
{
  "provider": "sft",
  "base_url": "http://127.0.0.1:8000/v1",
  "model_name": "kb-sft",
  "base_model": "qwen-flash",
  "sft_model": "kb-sft",
  "temperature": 0.0,
  "max_tokens": 1024,
  "timeout_sec": 60,
  "api_key_set": true
}
[stage5] check passed
```

可选健康检查：

```bash
uv run python fine_tuning/scripts/check_stage5_serving.py --health
```

实际结果：

```text
未执行。当前未启动 vLLM LoRA 服务，本地只验证配置解析和业务接入代码。
```

## 4. 验收标准

```text
1. 语法检查退出码为 0；
2. 配置单测通过；
3. base 模式能解析 base_url / model / timeout；
4. sft 模式能把 active_model 切到 ANSWER_SFT_MODEL；
5. 未启动模型服务时，不执行 --health 不应失败；
6. 评估产物、adapter、虚拟环境不进入 Git。
```

## 5. 正式接入前置条件

```text
1. 阶段二已使用强模型正式造数；
2. 阶段三已训练出 LoRA adapter；
3. 阶段四 Base vs SFT 离线评估通过；
4. vLLM 已使用 --enable-lora 启动；
5. /v1/models 中能看到 ANSWER_SFT_MODEL；
6. 业务 API 支持重启回滚到 ANSWER_MODEL_PROVIDER=base。
```
