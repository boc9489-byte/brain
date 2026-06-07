#!/usr/bin/env python3
"""从 Milvus 导出知识库 chunk，作为 SFT 数据构造的真实知识来源。

在仓库根目录执行：
    uv run python fine_tuning/scripts/export_kb_chunks.py

输出：
    fine_tuning/data/raw/kb_chunks.jsonl
    fine_tuning/data/raw/_export_stats.json

阶段一原则：先确认真实 chunk 质量，再谈训练；不要直接手写脱离知识库的样本。
"""

from __future__ import annotations

import argparse
import contextlib
import signal
from typing import Any, Dict, Iterable, List

import _common as C


class MilvusOperationTimeout(TimeoutError):
    """Milvus 连接或查询超时。

    端口能通不代表 pymilvus 的 gRPC 连接可用，所以这里单独封装超时异常，
    方便用户区分“网络层可达”和“Milvus 客户端不可用”。
    """
    pass


@contextlib.contextmanager
def operation_timeout(seconds: int, label: str):
    """给 Milvus 连接/查询加硬超时，避免命令一直挂起。"""
    if seconds <= 0:
        yield
        return

    def _raise_timeout(_signum, _frame):
        raise MilvusOperationTimeout(f"{label} timed out after {seconds}s")

    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def normalize_row(row: Dict[str, Any], fields: Dict[str, str]) -> Dict[str, Any]:
    """把 Milvus 原始字段映射成微调侧统一字段名。"""
    rec = {logical: row.get(actual) for logical, actual in fields.items()}
    for key in ("content", "title", "parent_title", "file_title", "item_name"):
        if rec.get(key) is None:
            rec[key] = ""
    return rec


def iter_query_batches(client: Any, collection: str, expr: str, output_fields: List[str], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    """分批读取 Milvus 数据。

    优先使用 query_iterator，避免一次 query 命中 Milvus 默认行数上限。
    个别 pymilvus 版本不支持该方法时，降级为一次 bounded query。
    """
    try:
        iterator = client.query_iterator(
            collection_name=collection,
            filter=expr or "",
            output_fields=output_fields,
            batch_size=batch_size,
        )
    except Exception as exc:
        print(f"[export] query_iterator unavailable, fallback to query: {exc}")
        yield client.query(
            collection_name=collection,
            filter=expr or "",
            output_fields=output_fields,
            limit=batch_size,
        )
        return

    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            yield batch
    finally:
        iterator.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=C.DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    cfg = C.load_config(args.config)
    milvus_cfg = cfg["milvus"]
    fields = milvus_cfg["fields"]
    uri = milvus_cfg.get("uri")
    collection = milvus_cfg.get("collection")
    # 这里提前打印配置，是为了排查“配置没读到”和“连接 Milvus 卡住”两类问题。
    print(f"[export] config loaded: uri={uri or '<empty>'} collection={collection or '<empty>'}", flush=True)
    if not uri:
        raise SystemExit("Missing Milvus uri. Set milvus.uri or MILVUS_URL in knowledge/.env.")
    if not collection:
        raise SystemExit("Missing Milvus collection. Set milvus.collection or CHUNKS_COLLECTION.")

    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pymilvus. Install it with `uv pip install -r fine_tuning/requirements-runtime.txt`."
        ) from exc

    client_kwargs = {"uri": uri}
    if milvus_cfg.get("token"):
        client_kwargs["token"] = milvus_cfg["token"]
    if milvus_cfg.get("db_name"):
        client_kwargs["db_name"] = milvus_cfg["db_name"]
    timeout_sec = int(milvus_cfg.get("timeout_sec", 20))
    print("[export] connecting Milvus...", flush=True)
    try:
        # pymilvus 在服务异常时可能长时间无输出，因此连接阶段也需要超时保护。
        with operation_timeout(timeout_sec, "Milvus connection"):
            client = MilvusClient(**client_kwargs)
    except MilvusOperationTimeout as exc:
        raise SystemExit(
            f"{exc}. TCP port may be open, but pymilvus did not finish the gRPC connection."
        ) from exc

    output_fields = list(fields.values())
    batch_size = int(milvus_cfg.get("batch_size", 1000))
    expr = milvus_cfg.get("expr", "")
    print(f"[export] connected. collection={collection} fields={output_fields}", flush=True)

    rows: List[Dict[str, Any]] = []
    empty_content = 0
    items = set()
    try:
        # 查询阶段也加超时，避免 collection 未 load 或服务半可用时阻塞。
        with operation_timeout(timeout_sec, "Milvus chunk query"):
            for batch in iter_query_batches(client, collection, expr, output_fields, batch_size):
                for raw in batch:
                    rec = normalize_row(raw, fields)
                    if not (rec.get("content") or "").strip():
                        empty_content += 1
                        continue
                    items.add(rec.get("item_name") or rec.get("file_title") or "UNKNOWN")
                    rows.append(rec)
    except MilvusOperationTimeout as exc:
        raise SystemExit(f"{exc}. Check Milvus service health and collection availability.") from exc

    out_dir = C.repo_path(cfg["dataset"]["out_dir"]) / "raw"
    out_path = out_dir / "kb_chunks.jsonl"
    C.write_jsonl(out_path, rows)
    C.write_json(
        out_dir / "_export_stats.json",
        {
            "collection": collection,
            "total_exported": len(rows),
            "empty_content_skipped": empty_content,
            "distinct_items": len(items),
            "fields": fields,
        },
    )

    print(f"[export] exported={len(rows)} -> {out_path}")
    print(f"[export] empty_content_skipped={empty_content}; distinct_items={len(items)}")
    if len(items) < 2:
        # 多商品/多文档覆盖不足时，拒答样本和 multi-hop 样本都会明显不足。
        print("[warn] distinct item count < 2; unanswerable samples will be limited.")


if __name__ == "__main__":
    main()
