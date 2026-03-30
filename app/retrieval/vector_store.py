"""
向量存储模块 — 封装 ChromaDB + SentenceTransformer。
负责：嵌入生成、chunk upsert、相似度检索。
"""

import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.models import ChunkDoc, RetrievedChunk

logger = logging.getLogger(__name__)

# ── 配置 ───────────────────────────────────────────────────────────────────────

DEFAULT_COLLECTION = "rag_documents"
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 中文文档换成 bge-small-zh-v1.5
DEFAULT_TOP_K      = 5
SCORE_THRESHOLD    = 0.35  # 低于这个分数 → 触发拒答


class VectorStore:
    """
    对外暴露三个方法：
    - upsert(chunks)   存入 chunk
    - search(query)    检索相关 chunk
    - delete_by_source 删除某个文档的所有 chunk
    """

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.model_name      = model_name
        self.collection_name = collection_name

        # 加载嵌入模型（只在服务启动时加载一次，开销大）
        logger.info(f"加载嵌入模型: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        logger.info("嵌入模型加载完成")

        # ChromaDB 本地持久化客户端
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # 获取或创建 collection，使用余弦相似度
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{collection_name}' 就绪，"
            f"当前 chunk 数量: {self.collection.count()}"
        )

    # ── 写入 ───────────────────────────────────────────────────────────────────

    def upsert(self, chunks: list[ChunkDoc]) -> None:
        """
        把 chunk 列表嵌入并存入 ChromaDB。
        upsert = 有则更新，无则插入，重复 ingest 同一文件不会产生重复数据。
        """
        if not chunks:
            return

        texts     = [c.text for c in chunks]
        ids       = [c.chunk_id for c in chunks]
        metadatas = [c.to_metadata_dict() for c in chunks]

        # 批量嵌入，normalize=True 是余弦相似度的必要条件
        embeddings = self.encoder.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # ── 检索 ───────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filter_source: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        语义检索，返回最相关的 top_k 个 chunk。

        Args:
            query:         用户问题
            top_k:         返回结果数量
            filter_source: 只在指定文件中检索（可选）

        Returns:
            RetrievedChunk 列表，按相似度从高到低排列
        """
        if self.collection.count() == 0:
            logger.warning("向量库为空，请先 ingest 文档")
            return []

        query_embedding = self.encoder.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        where_filter = {"source": filter_source} if filter_source else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []

        for rank, (doc, meta, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ), start=1):
            # ChromaDB 返回的是余弦距离（0=完全相同，2=完全相反）
            # 转换成相似度分数 [0, 1]
            score = 1.0 - (distance / 2.0)

            chunk = ChunkDoc(
                chunk_id      = meta["chunk_id"],
                text          = doc,
                source        = meta["source"],
                file_type     = meta["file_type"],
                page_num      = meta["page_num"] if meta["page_num"] != -1 else None,
                section_title = meta["section_title"] or None,
                char_start    = meta["char_start"],
                char_end      = meta["char_end"],
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=score, rank=rank))

        return retrieved

    # ── 工具方法 ───────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self.collection.count()

    def delete_by_source(self, source: str) -> None:
        """删除某个文档的所有 chunk，用于重新 ingest 前清理旧数据。"""
        self.collection.delete(where={"source": source})
        logger.info(f"已删除文档 '{source}' 的所有 chunk")
