#!/usr/bin/env python3
"""阶段三 LoRA 合并脚本。

训练时保存的是 LoRA adapter。后续如果需要导出完整模型，可用本脚本把 adapter
合并回 base model。合并动作通常在 GPU 或大内存机器上执行。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SCRIPTS_DIR = REPO_ROOT / "fine_tuning" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import _common as C  # noqa: E402
from train_sft import train_cfg  # noqa: E402


def merge_cfg(cfg: Dict[str, Any], adapter_override: str | None, out_override: str | None) -> Dict[str, str]:
    """读取合并配置。"""
    training = train_cfg(cfg)
    adapter = adapter_override or training["output_dir"]
    out_dir = out_override or f"{adapter.rstrip('/')}-merged"
    return {
        "base_model": training["base_model"],
        "adapter_dir": adapter,
        "output_dir": out_dir,
    }


def check_only(settings: Dict[str, str]) -> None:
    """检查合并输入输出路径，不加载模型。"""
    adapter_path = C.repo_path(settings["adapter_dir"])
    print(f"[merge-check] base_model={settings['base_model']}")
    print(f"[merge-check] adapter_dir={adapter_path}")
    print(f"[merge-check] output_dir={C.repo_path(settings['output_dir'])}")
    if not adapter_path.exists():
        print("[merge-check] adapter does not exist yet; run train_sft.py first.")
    else:
        print("[merge-check] adapter exists")
    print("[merge-check] passed")


def merge(settings: Dict[str, str], base_override: str | None) -> None:
    """执行 LoRA merge_and_unload。"""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing merge dependencies. Install fine_tuning/requirements-train.txt in a GPU environment."
        ) from exc

    base_model = base_override or settings["base_model"]
    adapter_dir = C.repo_path(settings["adapter_dir"])
    output_dir = C.repo_path(settings["output_dir"])
    if not adapter_dir.exists():
        raise SystemExit(f"Adapter not found: {adapter_dir}")

    print(f"[merge] base={base_model}")
    print(f"[merge] adapter={adapter_dir}")
    print(f"[merge] output={output_dir}")
    # 不在 4-bit 量化模型上 merge，避免量化误差被固化进合并权重。
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model, trust_remote_code=True).save_pretrained(str(output_dir))
    print(f"[merge] merged model saved -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=C.DEFAULT_CONFIG_PATH)
    parser.add_argument("--base", default=None, help="覆盖 train.base_model")
    parser.add_argument("--adapter", default=None, help="覆盖 adapter 目录")
    parser.add_argument("--out", default=None, help="覆盖合并后输出目录")
    parser.add_argument("--check-only", action="store_true", help="只检查路径，不加载模型。")
    args = parser.parse_args()

    cfg = C.load_config(args.config)
    settings = merge_cfg(cfg, args.adapter, args.out)
    if args.base:
        settings["base_model"] = args.base

    if args.check_only:
        check_only(settings)
        return
    merge(settings, args.base)


if __name__ == "__main__":
    main()

