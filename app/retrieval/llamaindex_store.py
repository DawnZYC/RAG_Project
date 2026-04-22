"""
LlamaIndex 检索后端 — 使用 ChromaVectorStore + VectorStoreIndex。

与 native 后端的区别：
- 嵌入：HuggingFaceEmbedding（本地，与 native 使用相同模型，无需联网）
- 检索策略：支持 similarity / mmr 两种模式，通过 RETRIEVAL_MODE 环境变量或参数切换
- 存储：同样是 ChromaDB，但通过 LlamaIndex 的封装层操作

接口与 VectorStoreBackend Protocol 完全一致，上层代码无需感知后端类型。

检索模式说明：
- similarity（默认）：纯向量余弦相似度，与 native 后端行为一致
- mmr：Maximal Marginal Relevance，在相关性基础上引入多样性惩罚，
        减少返回结果中内容高度重叠的情况。
        由 mmr_threshold 控制平衡点（0.0=最多样，1.0=最相关，默认 0.7）
"""

import logging
from pathlib import Path
from typing import Literal, Optional

from app.models import ChunkDoc, RetrievedChunk
from app.retrieval.vector_store import DEFAULT_COLLECTION, DEFAULT_MODEL_NAME, DEFAULT_TOP_K

logger = logging.getLogger(__name__)

# 支持的检索模式类型
RetrievalMode = Literal["similarity", "mmr"]

DEFAULT_RETRIEVAL_MODE: RetrievalMode = "similarity"
DEFAULT_MMR_THRESHOLD = 0.7   # lambda 值：0.0=最多样，1.0=最相关


class LlamaIndexVectorStore:
    """
    LlamaIndex 检索后端。

    内部结构：
      ChromaDB (持久化) ──> chromadb_collection
                               │
                    LlamaIndex ChromaVectorStore
                               │
                    VectorStoreIndex（管理 embedding + 检索）
                               │
                    VectorIndexRetriever（执行查询，支持多种 query_mode）
    """

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL_NAME,
        retrieval_mode: RetrievalMode = DEFAULT_RETRIEVAL_MODE,
        mmr_threshold: float = DEFAULT_MMR_THRESHOLD,
    ):
        self.collection_name  = collection_name
        self.model_name       = model_name
        self.retrieval_mode   = retrieval_mode
        self.mmr_threshold    = mmr_threshold

        # ── 延迟导入，避免未安装 llama-index 时报错 ──────────────────────────────
        import chromadb
        from chromadb.config import Settings
        from llama_index.core import VectorStoreIndex, StorageContext
        from llama_index.core import Settings as LISettings
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        # ── 嵌入模型（与 native 同款，本地运行）──────────────────────────────────
        logger.info(f"[LlamaIndex] 加载嵌入模型: {model_name}")
        embed_model = HuggingFaceEmbedding(model_name=model_name)
        LISettings.embed_model = embed_model
        LISettings.llm = None  # 检索层不使用 LLM，关闭自动加载
        logger.info(f"[LlamaIndex] 嵌入模型加载完成，检索模式: {retrieval_mode}")

        # ── ChromaDB 连接 ─────────────────────────────────────────────────────
        persist_dir.mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        chroma_collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # ── LlamaIndex 向量存储 + 索引 ────────────────────────────────────────
        vector_store    = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 从已有 ChromaDB 数据加载索引（若为空则建空索引）
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )

        # 保存引用，upsert / delete 时需要
        self._vector_store      = vector_store
        self._storage_context   = storage_context
        self._embed_model       = embed_model
        self._chroma_collection = chroma_collection

        logger.info(
            f"[LlamaIndex] ChromaDB collection '{collection_name}' 就绪，"
            f"当前 chunk 数量: {chroma_collection.count()}"
        )

    # ── 写入 ───────────────────────────────────────────────────────────────────

    def upsert(self, chunks: list[ChunkDoc]) -> None:
        """
        将 ChunkDoc 列表转换为 LlamaIndex TextNode 并写入向量存储。
        insert_nodes 内部自动生成嵌入并写入 ChromaDB，chunk_id 作为 node id 保证幂等。
        """
        if not chunks:
            return

        from llama_index.core.schema import TextNode

        nodes = [
            TextNode(
                id_=c.chunk_id,
                text=c.text,
                metadata=c.to_metadata_dict(),
            )
            for c in chunks
        ]

        self._index.insert_nodes(nodes)
        logger.info(f"[LlamaIndex] upsert {len(nodes)} 个 chunk 完成")

    # ── 检索 ───────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filter_source: Optional[str] = None,
        retrieval_mode: Optional[RetrievalMode] = None,
        mmr_threshold: Optional[float] = None,
    ) -> list[RetrievedChunk]:
        """
        语义检索，返回最相关的 top_k 个 chunk。

        Args:
            query:          用户问题
            top_k:          返回结果数量
            filter_source:  只在指定文件中检索（可选）
            retrieval_mode: 本次检索模式，覆盖初始化时的默认值（可选）
            mmr_threshold:  MMR 模式下的多样性权重，覆盖初始化时的默认值（可选）

        检索模式：
            similarity — 纯余弦相似度排序（等同 native）
            mmr        — 相关性 + 多样性平衡，减少结果内容重叠
        """
        if self._chroma_collection.count() == 0:
            logger.warning("[LlamaIndex] 向量库为空，请先 ingest 文档")
            return []

        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
        from llama_index.core.vector_stores.types import VectorStoreQueryMode

        # 本次检索使用的参数：优先用调用方传入的，否则用初始化时的默认值
        mode      = retrieval_mode or self.retrieval_mode
        threshold = mmr_threshold  if mmr_threshold is not None else self.mmr_threshold

        # 把字符串模式映射到 LlamaIndex 的枚举值
        query_mode = (
            VectorStoreQueryMode.MMR
            if mode == "mmr"
            else VectorStoreQueryMode.DEFAULT
        )

        # 构建元数据过滤器（可选）
        filters = None
        if filter_source:
            filters = MetadataFilters(
                filters=[ExactMatchFilter(key="source", value=filter_source)]
            )

        # MMR 模式下需要多取候选集，再从中做多样性筛选
        # fetch_k 是初始候选数量，top_k 是最终返回数量
        # 候选池越大，MMR 的多样性效果越好，但速度稍慢
        fetch_k = top_k * 4 if mode == "mmr" else top_k

        retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=fetch_k,
            vector_store_query_mode=query_mode,
            # MMR 专属参数：lambda 值控制相关性与多样性的权重
            # 只在 mmr 模式下生效，similarity 模式忽略此参数
            vector_store_kwargs={"mmr_threshold": threshold} if mode == "mmr" else {},
            filters=filters,
        )

        nodes_with_scores = retriever.retrieve(query)

        # MMR 模式下候选集更多，截取前 top_k 个
        nodes_with_scores = nodes_with_scores[:top_k]

        retrieved: list[RetrievedChunk] = []
        for rank, node_with_score in enumerate(nodes_with_scores, start=1):
            node  = node_with_score.node
            meta  = node.metadata
            score = float(node_with_score.score or 0.0)

            chunk = ChunkDoc(
                chunk_id      = meta.get("chunk_id", node.node_id),
                text          = node.get_content(),
                source        = meta.get("source", ""),
                file_type     = meta.get("file_type", ""),
                page_num      = meta.get("page_num") if meta.get("page_num") != -1 else None,
                section_title = meta.get("section_title") or None,
                char_start    = meta.get("char_start", 0),
                char_end      = meta.get("char_end", 0),
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=score, rank=rank))

        logger.debug(
            f"[LlamaIndex] search 完成，模式={mode}，"
            f"返回 {len(retrieved)} 个 chunk"
        )
        return retrieved

    # ── 工具方法 ───────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self._chroma_collection.count()

    def delete_by_source(self, source: str) -> None:
        """删除某个文档的所有 chunk。直接操作 ChromaDB，绕过 LlamaIndex 封装。"""
        self._chroma_collection.delete(where={"source": source})
        logger.info(f"[LlamaIndex] 已删除文档 '{source}' 的所有 chunk")
