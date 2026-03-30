"""
查询日志模块 — 把每次 RAG 请求记录到 SQLite。
用于后续评估检索质量和排查问题。
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

from app.models import RAGResponse

logger = logging.getLogger(__name__)

Base = declarative_base()


class QueryLog(Base):
    """每次请求对应一条记录。"""
    __tablename__ = "query_log"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    timestamp     = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    query         = Column(Text, nullable=False)
    answer        = Column(Text)
    refused       = Column(Boolean, default=False)
    refuse_reason = Column(Text)
    top_score     = Column(Float)    # 最高检索分数
    num_sources   = Column(Integer)  # 引用了几个 chunk
    sources_json  = Column(Text)     # 引用详情，JSON 格式
    latency_ms    = Column(Float)    # 整个请求耗时
    model         = Column(String(64))


class QueryLogger:

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(engine)
        self.SessionLocal = sessionmaker(bind=engine)
        logger.info(f"查询日志初始化完成: {db_path}")

    def log(
        self,
        query: str,
        response: RAGResponse,
        latency_ms: float,
        model: str,
    ) -> None:
        """记录一次 RAG 请求。"""
        sources_data = [
            {
                "source":  rc.chunk.source,
                "page":    rc.chunk.page_num,
                "section": rc.chunk.section_title,
                "score":   round(rc.score, 4),
                "rank":    rc.rank,
            }
            for rc in response.sources
        ]

        record = QueryLog(
            query         = query,
            answer        = response.answer or None,
            refused       = response.refused,
            refuse_reason = response.refuse_reason,
            top_score     = response.sources[0].score if response.sources else None,
            num_sources   = len(response.sources),
            sources_json  = json.dumps(sources_data, ensure_ascii=False),
            latency_ms    = round(latency_ms, 2),
            model         = model,
        )

        with self.SessionLocal() as session:
            session.add(record)
            session.commit()
