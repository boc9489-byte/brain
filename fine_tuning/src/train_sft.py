#!/usr/bin/env python3
"""阶段三 QLoRA 训练入口。

输入：
    fine_tuning/data/processed/sft_train.jsonl

输出：
    fine_tuning/outputs/kb-sft/

本脚本支持 --check-only，用于无 GPU / 未安装训练依赖的本地环境做提交前检查。
真实训练时再导入 torch、transformers、trl、peft 等重依赖。
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SCRIPTS_DIR = REPO_ROOT / "fine_tuning" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import _common as C  # noqa: E402


DEFAULT_ASSISTANT_TEMPLATE = "<|im_start|>assistant\n"


def train_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """读取训练配置并补默认值。"""
    raw = cfg.setdefault("train", {})
    return {
        "base_model": raw.get("base_model", "Qwen/Qwen2.5-3B-Instruct"),
        "output_dir": raw.get("output_dir", "fine_tuning/outputs/kb-sft"),
        "max_seq_len": int(raw.get("max_seq_len", 2048)),
        "load_in_4bit": bool(raw.get("load_in_4bit", True)),
        "lora_r": int(raw.get("lora_r", 16)),
        "lora_alpha": int(raw.get("lora_alpha", 32)),
        "lora_dropout": float(raw.get("lora_dropout", 0.05)),
        "learning_rate": float(raw.get("learning_rate", 2e-4)),
        "epochs": float(raw.get("epochs", 3)),
        "per_device_batch": int(raw.get("per_device_batch", 1)),
        "grad_accum": int(raw.get("grad_accum", 16)),
        "warmup_ratio": float(raw.get("warmup_ratio", 0.03)),
        "logging_steps": int(raw.get("logging_steps", 10)),
        "save_strategy": raw.get("save_strategy", "epoch"),
        "seed": int(raw.get("seed", cfg.get("dataset", {}).get("seed", 42))),
        "assistant_template": raw.get("assistant_template", DEFAULT_ASSISTANT_TEMPLATE),
        "target_modules": raw.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    }


def dataset_paths(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """定位阶段二训练数据。"""
    processed_dir = C.repo_path(cfg["dataset"]["out_dir"]) / "processed"
    return {
        "train": processed_dir / "sft_train.jsonl",
        "holdout": processed_dir / "sft_holdout.jsonl",
    }


def validate_messages_rows(rows: List[Dict[str, Any]]) -> List[str]:
    """训练前做轻量 schema 检查，避免把坏数据送到 GPU。"""
    errors: List[str] = []
    if not rows:
        return ["training dataset is empty"]
    for idx, row in enumerate(rows[:100], 1):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            errors.append(f"row {idx}: messages must have 3 turns")
            continue
        roles = [message.get("role") for message in messages]
        if roles != ["system", "user", "assistant"]:
            errors.append(f"row {idx}: invalid roles {roles}")
        if not messages[2].get("content"):
            errors.append(f"row {idx}: empty assistant content")
    return errors


def check_only(cfg: Dict[str, Any], training: Dict[str, Any]) -> None:
    """本地检查模式：不导入训练依赖，不下载模型。"""
    paths = dataset_paths(cfg)
    rows = C.read_jsonl(paths["train"])
    errors = validate_messages_rows(rows)
    print(f"[train-check] train_path={paths['train']}")
    print(f"[train-check] rows={len(rows)}")
    print(f"[train-check] base_model={training['base_model']}")
    print(f"[train-check] output_dir={C.repo_path(training['output_dir'])}")
    print(f"[train-check] max_seq_len={training['max_seq_len']}")
    print(f"[train-check] load_in_4bit={training['load_in_4bit']}")
    if errors:
        for error in errors:
            print(f"[train-check][error] {error}")
        raise SystemExit("[train-check] failed")
    print("[train-check] passed")


def write_training_summary(output_dir: Path, cfg: Dict[str, Any], training: Dict[str, Any], train_rows: int) -> None:
    """保存训练摘要，便于阶段四复盘和面试说明。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "base_model": training["base_model"],
        "train_rows": train_rows,
        "max_seq_len": training["max_seq_len"],
        "load_in_4bit": training["load_in_4bit"],
        "lora": {
            "r": training["lora_r"],
            "alpha": training["lora_alpha"],
            "dropout": training["lora_dropout"],
            "target_modules": training["target_modules"],
        },
        "optimizer": {
            "learning_rate": training["learning_rate"],
            "epochs": training["epochs"],
            "per_device_batch": training["per_device_batch"],
            "grad_accum": training["grad_accum"],
            "warmup_ratio": training["warmup_ratio"],
        },
        "dataset_out_dir": cfg["dataset"]["out_dir"],
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def train(cfg: Dict[str, Any], training: Dict[str, Any], base_override: str | None, out_override: str | None) -> None:
    """执行真实 QLoRA 训练。"""
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install fine_tuning/requirements-train.txt in a GPU environment."
        ) from exc

    if base_override:
        training["base_model"] = base_override
    if out_override:
        training["output_dir"] = out_override

    paths = dataset_paths(cfg)
    train_rows = C.read_jsonl(paths["train"])
    errors = validate_messages_rows(train_rows)
    if errors:
        raise SystemExit("[train] dataset check failed: " + "; ".join(errors[:5]))

    output_dir = C.repo_path(training["output_dir"])
    base_model = training["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def to_text(example: Dict[str, Any]) -> Dict[str, str]:
        """把 messages 渲染为模型原生 chat template 文本。"""
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    dataset = load_dataset("json", data_files={"train": str(paths["train"])})["train"]
    original_columns = list(dataset.column_names)
    dataset = dataset.map(to_text, remove_columns=original_columns)

    model_kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if training["load_in_4bit"]:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    collator = DataCollatorForCompletionOnlyLM(
        response_template=training["assistant_template"],
        tokenizer=tokenizer,
    )
    lora = LoraConfig(
        r=training["lora_r"],
        lora_alpha=training["lora_alpha"],
        lora_dropout=training["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=training["target_modules"],
    )

    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=training["epochs"],
        per_device_train_batch_size=training["per_device_batch"],
        gradient_accumulation_steps=training["grad_accum"],
        learning_rate=training["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=training["warmup_ratio"],
        logging_steps=training["logging_steps"],
        save_strategy=training["save_strategy"],
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=training["max_seq_len"],
        seed=training["seed"],
        dataset_text_field="text",
    )

    print(f"[train] base={base_model}")
    print(f"[train] rows={len(dataset)} output={output_dir}")
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": dataset,
        "data_collator": collator,
        "peft_config": lora,
    }
    # TRL/Transformers 的 Trainer 参数名有过迁移：新版本偏 processing_class，
    # 旧版本常见 tokenizer。这里做签名判断，降低训练环境版本漂移带来的失败率。
    trainer_params = inspect.signature(SFTTrainer).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    write_training_summary(output_dir, cfg, training, len(train_rows))
    print(f"[train] adapter saved -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=C.DEFAULT_CONFIG_PATH)
    parser.add_argument("--base", default=None, help="覆盖 train.base_model")
    parser.add_argument("--out", default=None, help="覆盖 train.output_dir")
    parser.add_argument("--check-only", action="store_true", help="只检查数据和配置，不加载模型。")
    args = parser.parse_args()

    cfg = C.load_config(args.config)
    training = train_cfg(cfg)
    if args.base:
        training["base_model"] = args.base
    if args.out:
        training["output_dir"] = args.out

    if args.check_only:
        check_only(cfg, training)
        return
    train(cfg, training, args.base, args.out)


if __name__ == "__main__":
    main()
