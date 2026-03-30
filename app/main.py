"""
FastAPI 应用入口。
管理共享资源（VectorStore、Generator、QueryLogger）的生命周期。
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.retrieval.vector_store import VectorStore
from app.generation.generator import Generator
from app.logging.db import QueryLogger

# 加载 .env 文件
load_dotenv()

# ── 日志配置 ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── 读取环境变量 ───────────────────────────────────────────────────────────────

def get_config() -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("环境变量 OPENAI_API_KEY 未设置，请检查 .env 文件")

    return {
        "openai_api_key":  api_key,
        "llm_model":       os.getenv("LLM_MODEL",       "gpt-4o-mini"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        "chroma_dir":      Path(os.getenv("CHROMA_DIR",      "data/chroma")),
        "processed_dir":   Path(os.getenv("PROCESSED_DIR",   "data/processed")),
        "db_path":         Path(os.getenv("DB_PATH",         "data/logs/queries.db")),
    }


# ── Lifespan：服务启动/关闭时执行 ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    服务启动时初始化所有共享资源，挂到 app.state 上。
    这样所有请求共用同一个 VectorStore 和 Generator 实例，
    避免每次请求都重新加载嵌入模型（非常慢）。
    """
    logger.info("RAG 服务启动中...")
    config = get_config()

    app.state.vector_store = VectorStore(
        persist_dir=config["chroma_dir"],
        model_name=config["embedding_model"],
    )
    app.state.generator = Generator(
        api_key=config["openai_api_key"],
        model=config["llm_model"],
    )
    app.state.query_logger = QueryLogger(db_path=config["db_path"])
    app.state.processed_dir = config["processed_dir"]

    logger.info("RAG 服务就绪")
    yield
    logger.info("RAG 服务关闭")


# ── 创建 FastAPI 应用 ──────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Service",
    description="基于检索增强生成的知识问答系统",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由，所有接口统一加 /api/v1 前缀
from app.api.routes import router
app.include_router(router, prefix="/api/v1")
