"""
分块模块 — 把解析出的文本切成小块，每块保留完整 metadata。
"""

import hashlib
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models import ChunkDoc, SupportedFileType
from app.ingestion.parsers import ParsedPage


CHUNK_SIZE    = 512  # 每块最大字符数
CHUNK_OVERLAP = 64   # 相邻块重叠字符数，防止关键信息被切断


def _make_chunk_id(source: str, page_num: Optional[int], char_start: int) -> str:
    """
    生成确定性的 chunk_id。
    同一个文件、同一个位置，每次生成的 ID 都相同。
    这样重复 ingest 同一文件时可以 upsert，不会产生重复数据。
    """
    raw = f"{source}:{page_num}:{char_start}"
    hash_str = hashlib.md5(raw.encode()).hexdigest()[:8]
    source_stem = Path(source).stem[:20]
    page_tag = f"p{page_num}" if page_num is not None else "p0"
    return f"{source_stem}_{page_tag}_{hash_str}"


def _get_splitter(file_type: SupportedFileType) -> RecursiveCharacterTextSplitter:
    """
    不同格式用不同的分隔符策略。
    表格类（XLSX/CSV）优先按换行切；
    Markdown 按标题层级切；
    其他文档按段落 → 句子 → 词的顺序切。
    """
    if file_type in (SupportedFileType.XLSX, SupportedFileType.CSV):
        separators = ["\n", " ", ""]
    elif file_type == SupportedFileType.MD:
        separators = ["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    else:
        separators = ["\n\n", "\n", "。", ".", " ", ""]

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separators,
        length_function=len,
    )


def chunk_pages(
    pages: list[ParsedPage],
    source: str,
    file_type: SupportedFileType,
) -> list[ChunkDoc]:
    """
    把解析出的页面列表切成 ChunkDoc 列表。

    Args:
        pages:     parsers.parse_document() 的输出
        source:    原始文件名，用于引用显示
        file_type: 决定分隔符策略
    """
    splitter = _get_splitter(file_type)
    chunks: list[ChunkDoc] = []

    for page_text, page_num, section_title in pages:
        if not page_text.strip():
            continue

        sub_texts = splitter.split_text(page_text)

        char_cursor = 0
        for sub_text in sub_texts:
            try:
                pos = page_text.index(sub_text, max(0, char_cursor - CHUNK_OVERLAP))
            except ValueError:
                pos = char_cursor

            char_start = pos
            char_end   = pos + len(sub_text)
            char_cursor = char_end

            chunk = ChunkDoc(
                chunk_id      = _make_chunk_id(source, page_num, char_start),
                text          = sub_text.strip(),
                source        = source,
                file_type     = file_type,
                page_num      = page_num,
                section_title = section_title,
                char_start    = char_start,
                char_end      = char_end,
            )
            chunks.append(chunk)

    return chunks


def chunk_document(
    pages: list[ParsedPage],
    file_path: Path,
    file_type: SupportedFileType,
) -> list[ChunkDoc]:
    """对外统一入口，用文件名作为 source 标识。"""
    return chunk_pages(pages, file_path.name, file_type)
