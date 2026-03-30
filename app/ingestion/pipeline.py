"""
Ingestion pipeline — 串联 解析 → 分块 → 存储 的完整流程。
这是 ingestion 模块对外的唯一入口，API 层直接调用这里。
"""

import json
import logging
from pathlib import Path

from app.ingestion.parsers import parse_document
from app.ingestion.chunker import chunk_document
from app.models import ChunkDoc, SupportedFileType

logger = logging.getLogger(__name__)


def ingest_file(
    file_path: Path,
    vector_store,
    processed_dir: Path,
) -> list[ChunkDoc]:
    """
    对单个文件执行完整的 ingestion 流程。

    步骤：
    1. 解析文档，提取文本 + metadata
    2. 分块，生成 ChunkDoc 列表
    3. 把 chunk 保存为 JSONL 文件（方便调试，不需要重新解析）
    4. 把 chunk upsert 到向量库

    Args:
        file_path:     待处理的文件路径
        vector_store:  VectorStore 实例（由 FastAPI lifespan 注入）
        processed_dir: JSONL 中间文件的保存目录

    Returns:
        成功 ingest 的 ChunkDoc 列表
    """
    logger.info(f"开始 ingest: {file_path.name}")

    # 第一步：解析
    pages, file_type = parse_document(file_path)
    logger.info(f"  解析完成，共 {len(pages)} 个页面/章节")

    # 第二步：分块
    chunks = chunk_document(pages, file_path, file_type)
    logger.info(f"  分块完成，共 {len(chunks)} 个 chunk")

    if not chunks:
        logger.warning(f"  {file_path.name} 没有产生任何 chunk，跳过")
        return []

    # 第三步：保存 JSONL 中间文件
    processed_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = processed_dir / f"{file_path.stem}_chunks.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {"text": chunk.text, **chunk.to_metadata_dict()}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"  JSONL 已保存: {jsonl_path}")

    # 第四步：upsert 到向量库
    vector_store.upsert(chunks)
    logger.info(f"  已 upsert {len(chunks)} 个 chunk 到向量库")

    return chunks


def ingest_directory(
    input_dir: Path,
    vector_store,
    processed_dir: Path,
) -> dict[str, int]:
    """
    批量 ingest 一个目录下所有支持的文件。

    Returns:
        {文件名: chunk 数量} 的字典，chunk 数量为 -1 表示该文件处理失败
    """
    supported_exts = {f".{ft.value}" for ft in SupportedFileType}
    files = [f for f in input_dir.iterdir() if f.suffix.lower() in supported_exts]

    if not files:
        logger.warning(f"目录 {input_dir} 中没有找到支持的文件")
        return {}

    results: dict[str, int] = {}
    for file_path in sorted(files):
        try:
            chunks = ingest_file(file_path, vector_store, processed_dir)
            results[file_path.name] = len(chunks)
        except Exception as e:
            logger.error(f"  {file_path.name} 处理失败: {e}")
            results[file_path.name] = -1

    return results
