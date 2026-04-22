"""
FastAPI 路由定义。
只负责接收请求、调用模块、返回结果，不写业务逻辑。

Session / Memory 管理在这里：
- app.state.sessions 是一个字典，key=session_id，value=消息历史列表
- 路由负责读取历史、传给 generator、把新 Q&A 写回历史
- generator 只接收 history 列表，不感知 session 概念
"""

import json
import time
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.models import SupportedFileType, RAGResponse
from app.retrieval.vector_store import DEFAULT_TOP_K

logger = logging.getLogger(__name__)
router = APIRouter()

# 单个 Session 最多保留多少轮历史（每轮 = 1条user + 1条assistant）
MEMORY_WINDOW_K    = 5
MAX_CONTEXT_CHUNKS = 5  # 最多放进 prompt 的 chunk 数，与 generator.py 保持一致


# ── 请求 / 响应格式 ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:      str      = Field(..., min_length=1, max_length=2000)
    top_k:         int      = Field(DEFAULT_TOP_K, ge=1, le=20)
    filter_source: str | None = Field(None, description="只在指定文件中检索")
    session_id:    str | None = Field(
        None,
        description=(
            "多轮对话的会话 ID（UUID）。"
            "不传 = 单次问答，行为与原来完全一致；"
            "传了 = 多轮模式，服务器会记住历史并自动改写追问。"
        )
    )


class SourceItem(BaseModel):
    source:        str
    page_num:      int | None
    section_title: str | None
    score:         float
    rank:          int
    excerpt:       str


class QueryResponse(BaseModel):
    answer:        str
    refused:       bool
    refuse_reason: str | None
    sources:       list[SourceItem]
    latency_ms:    float
    session_id:    str | None = None   # 返回给前端，方便前端存储


class IngestResponse(BaseModel):
    filename:     str
    chunks_added: int
    status:       str


# ── Session 工具函数 ───────────────────────────────────────────────────────────

def _get_history(request: Request, session_id: str | None) -> list[dict]:
    """
    从 app.state.sessions 读取指定 session 的历史消息列表。
    session 不存在时自动创建空列表。
    没有 session_id 时返回空列表（单次问答模式）。
    """
    if not session_id:
        return []
    sessions = request.app.state.sessions
    if session_id not in sessions:
        sessions[session_id] = []
    return sessions[session_id]


def _save_to_history(
    request: Request,
    session_id: str | None,
    question: str,
    answer: str,
) -> None:
    """
    把本轮的问题和回答追加到 Session 历史。
    超过 MEMORY_WINDOW_K 轮时，从头删除最旧的一轮（2条消息）。
    """
    if not session_id:
        return

    sessions = request.app.state.sessions
    history  = sessions.setdefault(session_id, [])

    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": answer})

    # 每轮 = 2条消息，超出窗口就删最旧的一轮
    max_messages = MEMORY_WINDOW_K * 2
    if len(history) > max_messages:
        sessions[session_id] = history[-max_messages:]


def _format_sources(retrieved_chunks) -> list[SourceItem]:
    """把 RetrievedChunk 列表转成 SourceItem 列表。"""
    return [
        SourceItem(
            source=rc.chunk.source,
            page_num=rc.chunk.page_num,
            section_title=rc.chunk.section_title,
            score=round(rc.score, 4),
            rank=rc.rank,
            excerpt=rc.chunk.text[:200],
        )
        for rc in retrieved_chunks
    ]


# ── 路由 ───────────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/stats")
def stats(request: Request):
    vs = request.app.state.vector_store
    return {
        "total_chunks":   vs.count(),
        "collection":     vs.collection_name,
        "model":          vs.model_name,
        "active_sessions": len(request.app.state.sessions),
    }


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: Request, file: UploadFile = File(...)):
    """上传文档并 ingest 到向量库。"""
    from app.ingestion.pipeline import ingest_file

    vector_store  = request.app.state.vector_store
    processed_dir = request.app.state.processed_dir

    suffix    = Path(file.filename).suffix.lower()
    supported = {f".{ft.value}" for ft in SupportedFileType}
    if suffix not in supported:
        raise HTTPException(
            status_code=415,
            detail=f"不支持的文件类型: {suffix}，支持: {sorted(supported)}"
        )

    content    = await file.read()
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
    """
    提问，返回带引用的回答。

    多轮对话流程（传 session_id 时）：
      1. 读取该 session 的历史
      2. 用历史改写追问（Query Rewriting）
      3. 用改写后的问题检索
      4. 生成时把历史注入 prompt
      5. 把本轮 Q&A 写回历史

    单次问答流程（不传 session_id）：
      直接检索 → 生成，和原来完全一致
    """
    vector_store = request.app.state.vector_store
    generator    = request.app.state.generator
    query_logger = request.app.state.query_logger

    t0         = time.perf_counter()
    session_id = body.session_id

    # ── 1. 读取历史 ───────────────────────────────────────────────────────────
    history = _get_history(request, session_id)

    # ── 2. Query Rewriting（有历史时才改写）──────────────────────────────────
    search_query = await generator.rewrite_query(body.question, history)

    # ── 3. 检索（用改写后的问题）─────────────────────────────────────────────
    retrieved = vector_store.search(
        query=search_query,
        top_k=body.top_k,
        filter_source=body.filter_source,
    )

    # ── 4. 生成（带入历史）───────────────────────────────────────────────────
    response = generator.generate(
        query=search_query,
        retrieved_chunks=retrieved,
        history=history,
    )

    latency_ms = (time.perf_counter() - t0) * 1000

    # ── 5. 写回历史（拒答时只记录问题，answer 置空避免污染历史）───────────────
    if session_id:
        _save_to_history(
            request=request,
            session_id=session_id,
            question=body.question,           # 存原始问题，不存改写版
            answer=response.answer if not response.refused else "[拒答]",
        )

    # ── 记录日志 ──────────────────────────────────────────────────────────────
    query_logger.log(
        query=body.question,
        response=response,
        latency_ms=latency_ms,
        model=generator.model,
    )

    return QueryResponse(
        answer=response.answer,
        refused=response.refused,
        refuse_reason=response.refuse_reason,
        sources=_format_sources(response.sources),
        latency_ms=round(latency_ms, 2),
        session_id=session_id,
    )


@router.post("/stream")
async def stream_query(body: QueryRequest, request: Request):
    """
    流式提问（SSE），支持多轮对话。

    SSE 事件格式：
      {"type": "token",  "content": "..."}
      {"type": "done",   "refused": false, "sources": [...], "session_id": "..."}
      {"type": "done",   "refused": true,  "reason": "..."}
      {"type": "error",  "message": "..."}
    """
    vector_store = request.app.state.vector_store
    generator    = request.app.state.generator
    query_logger = request.app.state.query_logger

    t0         = time.perf_counter()
    session_id = body.session_id

    # 读取历史、改写问题、检索 —— 这三步在流式开始前完成
    history      = _get_history(request, session_id)
    search_query = await generator.rewrite_query(body.question, history)
    retrieved    = vector_store.search(
        query=search_query,
        top_k=body.top_k,
        filter_source=body.filter_source,
    )

    async def event_generator():
        full_answer = ""
        final_event = None

        async for event in generator.stream(search_query, retrieved, history=history):
            if event["type"] == "token":
                full_answer += event["content"]
            elif event["type"] == "done":
                final_event = event
                # 把 session_id 附在 done 事件里，前端存起来供下轮使用
                event = {**event, "session_id": session_id}

            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        # 流结束后：写回历史 + 写日志
        if final_event:
            if session_id:
                _save_to_history(
                    request=request,
                    session_id=session_id,
                    question=body.question,
                    answer=full_answer if not final_event.get("refused") else "[拒答]",
                )

            latency_ms = (time.perf_counter() - t0) * 1000
            fake_response = RAGResponse(
                answer=full_answer if not final_event.get("refused") else "",
                sources=retrieved[:MAX_CONTEXT_CHUNKS] if not final_event.get("refused") else [],
                refused=final_event.get("refused", False),
                refuse_reason=final_event.get("reason"),
            )
            query_logger.log(
                query=body.question,
                response=fake_response,
                latency_ms=latency_ms,
                model=generator.model,
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/session/{session_id}")
def delete_session(session_id: str, request: Request):
    """
    清空指定会话的历史记录。
    前端点"新对话"时调用，下次提问就是全新对话。
    """
    sessions = request.app.state.sessions
    if session_id in sessions:
        del sessions[session_id]
        logger.info(f"Session {session_id} 已清除")
        return {"status": "cleared", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


@router.get("/session/{session_id}")
def get_session(session_id: str, request: Request):
    """
    查看指定会话的历史记录（调试用）。
    """
    sessions = request.app.state.sessions
    history  = sessions.get(session_id, [])
    return {
        "session_id": session_id,
        "turn_count": len(history) // 2,
        "history":    history,
    }


@router.delete("/document/{filename}")
def delete_document(filename: str, request: Request):
    """删除某个文档的所有 chunk。"""
    request.app.state.vector_store.delete_by_source(filename)
    return {"status": "deleted", "filename": filename}


