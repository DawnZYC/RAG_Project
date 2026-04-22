"""
检索后端工厂 — 根据 RAG_BACKEND 环境变量决定创建哪个后端实例。
调用方（main.py）只需调用 create_vector_store()，不需要知道后端是哪个。

注意：LangChain 不用于检索层，而是用于生成层（streaming、memory、LCEL）。
检索层只支持 native 和 llamaindex 两个后端。
"""

import logging
from pathlib import Path

from app.retrieval.vector_store import VectorStore, DEFAULT_COLLECTION, DEFAULT_MODEL_NAME
from app.retrieval.llamaindex_store import DEFAULT_RETRIEVAL_MODE, DEFAULT_MMR_THRESHOLD

logger = logging.getLogger(__name__)


def create_vector_store(
    backend: str,
    persist_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    collection_name: str = DEFAULT_COLLECTION,
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
    mmr_threshold: float = DEFAULT_MMR_THRESHOLD,
):
    """
    工厂函数：根据 backend 参数创建对应的检索后端。

    Args:
        backend:         "native" / "llamaindex"
        persist_dir:     ChromaDB 数据存储路径
        model_name:      嵌入模型名称
        collection_name: ChromaDB collection 名称
        retrieval_mode:  LlamaIndex 检索模式："similarity"（默认）/ "mmr"
        mmr_threshold:   MMR 多样性权重，0.0=最多样，1.0=最相关（默认 0.7）

    Returns:
        满足 VectorStoreBackend 接口的实例
    """

    if backend == "native":
        logger.info("使用 native 检索后端（sentence-transformers + ChromaDB）")
        return VectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
            model_name=model_name,
        )

    elif backend == "llamaindex":
        logger.info(
            f"使用 LlamaIndex 检索后端（VectorStoreIndex + ChromaDB），"
            f"检索模式: {retrieval_mode}"
        )
        from app.retrieval.llamaindex_store import LlamaIndexVectorStore
        return LlamaIndexVectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
            model_name=model_name,
            retrieval_mode=retrieval_mode,
            mmr_threshold=mmr_threshold,
        )

    else:
        raise ValueError(
            f"未知的 RAG_BACKEND: '{backend}'，"
            f"可选值：native / llamaindex"
        )
