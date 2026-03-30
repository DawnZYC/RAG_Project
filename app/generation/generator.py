"""
生成模块 — 调用 OpenAI，生成带引用的回答，并处理拒答逻辑。

设计原则：
- 这个模块只负责生成，不碰向量库
- 拒答有两道判断：调 LLM 之前（分数阈值）和调 LLM 之后（LLM 自己说证据不足）
"""

import logging
from typing import Optional

from openai import OpenAI

from app.models import RAGResponse, RetrievedChunk
from app.retrieval.vector_store import SCORE_THRESHOLD

logger = logging.getLogger(__name__)

# ── 配置 ───────────────────────────────────────────────────────────────────────

DEFAULT_MODEL      = "gpt-4o-mini"
MAX_CONTEXT_CHUNKS = 5    # 最多把几个 chunk 放进 prompt
TEMPERATURE        = 0.2  # 越低越保守，越不容易胡编


# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一个严谨的问答助手。
你的回答必须完全基于下方提供的上下文，不得使用上下文以外的知识。

规则：
1. 每个事实陈述后面必须标注来源，格式：[来源: 文件名, 第X页]
   如果没有页码，格式为：[来源: 文件名]
2. 如果上下文不足以回答问题，直接回复：INSUFFICIENT_EVIDENCE: <原因>
3. 不要推测或补充上下文中没有的内容。
4. 回答简洁，只说有据可查的内容。"""

CONTEXT_TEMPLATE = """上下文（按相关度排列）：

{context_blocks}

---
问题：{question}

回答（每条陈述需标注来源）："""


def _format_context_blocks(chunks: list[RetrievedChunk]) -> str:
    """把检索结果格式化成 prompt 里的上下文文本。"""
    blocks = []
    for rc in chunks:
        c = rc.chunk
        if c.page_num and c.page_num > 0:
            citation = f"{c.source}, 第{c.page_num}页"
        else:
            citation = c.source

        if c.section_title:
            citation += f"，章节：{c.section_title}"

        block = (
            f"[{rc.rank}] 来源：{citation}（相关度：{rc.score:.2f}）\n"
            f"{c.text}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


# ── 拒答判断 ───────────────────────────────────────────────────────────────────

def _should_refuse(chunks: list[RetrievedChunk]) -> tuple[bool, Optional[str]]:
    """
    调用 LLM 之前的拒答判断。
    返回 (是否拒答, 原因)。
    """
    if not chunks:
        return True, "知识库中没有找到相关文档，请先上传文档。"

    top_score = chunks[0].score
    if top_score < SCORE_THRESHOLD:
        return True, (
            f"最高匹配分数（{top_score:.2f}）低于置信度阈值（{SCORE_THRESHOLD}），"
            f"证据不足，无法可靠回答。"
        )

    return False, None


# ── Generator ──────────────────────────────────────────────────────────────────

class Generator:

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.client = OpenAI(api_key=api_key)
        self.model  = model
        logger.info(f"Generator 初始化，模型: {model}")

    def generate(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> RAGResponse:
        """
        生成带引用的回答。

        流程：
        1. 调 LLM 前拒答检查（分数阈值）
        2. 格式化上下文，构建 prompt
        3. 调用 OpenAI
        4. 调 LLM 后拒答检查（LLM 返回 INSUFFICIENT_EVIDENCE）
        5. 返回 RAGResponse

        任何步骤出错都会返回 refused=True，不会抛异常。
        """
        # 第一道拒答
        refuse, reason = _should_refuse(retrieved_chunks)
        if refuse:
            logger.info(f"前置拒答: {reason}")
            return RAGResponse(
                answer="",
                sources=[],
                refused=True,
                refuse_reason=reason,
            )

        # 构建 prompt
        context_chunks = retrieved_chunks[:MAX_CONTEXT_CHUNKS]
        context_text   = _format_context_blocks(context_chunks)
        user_message   = CONTEXT_TEMPLATE.format(
            context_blocks=context_text,
            question=query,
        )

        # 调用 OpenAI
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=TEMPERATURE,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI 调用失败: {e}")
            return RAGResponse(
                answer="",
                sources=context_chunks,
                refused=True,
                refuse_reason=f"LLM 调用出错: {str(e)}",
            )

        # 第二道拒答：LLM 自己判断证据不足
        if answer.startswith("INSUFFICIENT_EVIDENCE"):
            reason = answer.replace("INSUFFICIENT_EVIDENCE:", "").strip()
            logger.info(f"LLM 拒答: {reason}")
            return RAGResponse(
                answer="",
                sources=context_chunks,
                refused=True,
                refuse_reason=reason,
            )

        logger.info(f"生成回答，长度 {len(answer)} 字符，引用 {len(context_chunks)} 个来源")
        return RAGResponse(
            answer=answer,
            sources=context_chunks,
            refused=False,
        )
