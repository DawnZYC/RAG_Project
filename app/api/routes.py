"""
FastAPI 路由定义。
只负责接收请求、调用模块、返回结果，不写业务逻辑。
"""

import time
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from pydantic import BaseModel, Field

from app.models import SupportedFileType
from app.retrieval.vector_store import DEFAULT_TOP_K

logger = logging.getLogger(__name__)
router = APIRouter()


# ── 请求 / 响应格式 ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:      str = Field(..., min_length=1, max_length=2000)
    top_k:         int = Field(DEFAULT_TOP_K, ge=1, le=20)
    filter_source: str | None = Field(None, description="只在指定文件中检索")


class SourceItem(BaseModel):
    source:        str
    page_num:      int | None
    section_title: str | None
    score:         float
    rank:          int
    excerpt:       str  # chunk 前 200 字符，用于前端展示


class QueryResponse(BaseModel):
    answer:        str
    refused:       bool
    refuse_reason: str | None
    sources:       list[SourceItem]
    latency_ms:    float


class IngestResponse(BaseModel):
    filename:     str
    chunks_added: int
    status:       str


# ── 路由 ───────────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/stats")
def stats(request: Request):
    vs = request.app.state.vector_store
    return {
        "total_chunks": vs.count(),
        "collection":   vs.collection_name,
        "model":        vs.model_name,
    }


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: Request, file: UploadFile = File(...)):
    """上传文档并 ingest 到向量库。"""
    from app.ingestion.pipeline import ingest_file

    vector_store  = request.app.state.vector_store
    processed_dir = request.app.state.processed_dir

    # 校验文件类型
    suffix = Path(file.filename).suffix.lower()
    supported = {f".{ft.value}" for ft in SupportedFileType}
    if suffix not in supported:
        raise HTTPException(
            status_code=415,
            detail=f"不支持的文件类型: {suffix}，支持: {sorted(supported)}"
        )

    # 把上传的文件流保存到临时文件
    content = await file.read()
    named_path = processed_dir / file.filename
    named_path.parent.mkdir(parents=True, exist_ok=True)
    named_path.write_bytes(content)

    try:
        chunks = ingest_file(
            file_path=named_path,
            vector_store=vector_store,
            processed_dir=processed_dir,
        )
    except Exception as e:
        logger.exception(f"Ingest 失败: {file.filename}")
        raise HTTPException(status_code=500, detail=f"Ingest 失败: {str(e)}")

    return IngestResponse(
        filename=file.filename,
        chunks_added=len(chunks),
        status="success",
    )


@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest, request: Request):
    """提问，返回带引用的回答。"""
    vector_store = request.app.state.vector_store
    generator    = request.app.state.generator
    query_logger = request.app.state.query_logger

    t0 = time.perf_counter()

    # 检索
    retrieved = vector_store.search(
        query=body.question,
        top_k=body.top_k,
        filter_source=body.filter_source,
    )

    # 生成
    response = generator.generate(
        query=body.question,
        retrieved_chunks=retrieved,
    )

    latency_ms = (time.perf_counter() - t0) * 1000

    # 记录日志
    query_logger.log(
        query=body.question,
        response=response,
        latency_ms=latency_ms,
        model=generator.model,
    )

    # 格式化来源列表
    sources = [
        SourceItem(
            source=rc.chunk.source,
            page_num=rc.chunk.page_num,
            section_title=rc.chunk.section_title,
            score=round(rc.score, 4),
            rank=rc.rank,
            excerpt=rc.chunk.text[:200],
        )
        for rc in response.sources
    ]

    return QueryResponse(
        answer=response.answer,
        refused=response.refused,
        refuse_reason=response.refuse_reason,
        sources=sources,
        latency_ms=round(latency_ms, 2),
    )


@router.delete("/document/{filename}")
def delete_document(filename: str, request: Request):
    """删除某个文档的所有 chunk。"""
    request.app.state.vector_store.delete_by_source(filename)
    return {"status": "deleted", "filename": filename}
