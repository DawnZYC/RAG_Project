"""
生成模块 — 调用 OpenAI，生成带引用的回答，并处理拒答逻辑。

设计原则：
- 这个模块只负责生成，不碰向量库，不管理 Session
- 拒答有两道判断：Pre-LLM（分数阈值）和 Post-LLM（LLM 返回 INSUFFICIENT_EVIDENCE）
- generate()       : 同步一次性返回，原有接口不变，支持可选 history
- stream()         : 异步生成器，LangChain LCEL 流式输出，支持可选 history
- rewrite_query()  : 把追问改写为独立完整问题，供 routes.py 在检索前调用

Session 管理（创建、存储、读取 Memory）由 routes.py 负责，不在这里。
Generator 只接收 history 列表，不感知 Session 概念。
"""

import logging
from typing import AsyncGenerator, Optional

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

# Query Rewriting Prompt：让 LLM 把追问改写成独立完整的问题
# 注意措辞：如果问题已经完整，直接返回原问题，不做改动
REWRITE_PROMPT = """根据以下对话历史，将用户的追问改写为一个独立完整的问题。
如果问题本身已经完整清晰、不依赖历史，直接返回原问题，不做任何改动。
只返回改写后的问题，不要加任何解释。

对话历史：
{history}

追问：{question}

改写后的完整问题："""


# ── 工具函数 ───────────────────────────────────────────────────────────────────

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


def _format_history_text(history: list[dict]) -> str:
    """
    把历史消息列表转成可读文本，用于 REWRITE_PROMPT。

    输入格式：
      [{"role": "user", "content": "退款政策？"},
       {"role": "assistant", "content": "30天内可退..."}]

    输出格式：
      用户：退款政策？
      助手：30天内可退...
    """
    lines = []
    for msg in history:
        role = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"{role}：{msg['content']}")
    return "\n".join(lines)


def _should_refuse(chunks: list[RetrievedChunk]) -> tuple[bool, Optional[str]]:
    """Pre-LLM 拒答判断，返回 (是否拒答, 原因)。"""
    if not chunks:
        return True, "知识库中没有找到相关文档，请先上传文档。"

    top_score = chunks[0].score
    if top_score < SCORE_THRESHOLD:
        return True, (
            f"最高匹配分数（{top_score:.2f}）低于置信度阈值（{SCORE_THRESHOLD}），"
            f"证据不足，无法可靠回答。"
        )

    return False, None


def _build_messages_with_history(
    system_prompt: str,
    user_message: str,
    history: list[dict],
) -> list[dict]:
    """
    把历史消息注入到 OpenAI messages 列表里。

    结构：
      system prompt
      历史消息（user / assistant 交替）
      当前用户消息（含检索上下文）

    为什么不把历史拼进 user_message 的文本里？
    - 注入为真实 message 对象，OpenAI 能正确理解对话结构
    - LLM 更清楚哪些是历史、哪些是当前上下文
    - 符合 OpenAI chat 接口的设计意图
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)                                   # 注入历史
    messages.append({"role": "user", "content": user_message}) # 当前问题
    return messages


# ── Generator ──────────────────────────────────────────────────────────────────

class Generator:

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.client   = OpenAI(api_key=api_key)
        self.api_key  = api_key
        self.model    = model
        logger.info(f"Generator 初始化，模型: {model}")

    # ── Query Rewriting ────────────────────────────────────────────────────────

    async def rewrite_query(
        self,
        question: str,
        history: list[dict],
    ) -> str:
        """
        用 LangChain LLM 把追问改写为独立完整的问题，供检索前使用。

        为什么在检索前改写？
        - 检索用的是向量相似度，"国际订单呢？"向量很模糊
        - 改写为"国际订单的退款政策是什么？"后向量更精准
        - 检索质量直接决定回答质量

        没有历史时直接返回原问题，不消耗额外 Token。
        """
        if not history:
            return question

        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        history_text = _format_history_text(history)

        prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)
        llm    = ChatOpenAI(model=self.model, temperature=0, api_key=self.api_key)
        chain  = prompt | llm | StrOutputParser()

        try:
            rewritten = await chain.ainvoke({
                "history":  history_text,
                "question": question,
            })
            rewritten = rewritten.strip()
            if rewritten and rewritten != question:
                logger.info(f"Query rewriting: '{question}' → '{rewritten}'")
            return rewritten or question
        except Exception as e:
            # 改写失败不影响主流程，回退到原问题
            logger.warning(f"Query rewriting 失败，使用原问题: {e}")
            return question

    # ── 普通生成 ───────────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        history: Optional[list[dict]] = None,
    ) -> RAGResponse:
        """
        生成带引用的回答（同步，一次性返回）。

        Args:
            query:            用户当前问题（已经过 rewrite_query 改写）
            retrieved_chunks: 检索结果
            history:          对话历史，格式 [{"role":"user","content":"..."},...]
                              None 或空列表 = 单次问答模式，行为与原来完全一致

        流程：
          Pre-LLM 拒答 → 构建 prompt（注入历史）→ 调 OpenAI → Post-LLM 拒答 → 返回
        """
        # 第一道拒答
        refuse, reason = _should_refuse(retrieved_chunks)
        if refuse:
            logger.info(f"前置拒答: {reason}")
            return RAGResponse(
                answer="", sources=[], refused=True, refuse_reason=reason,
            )

        # 构建 prompt
        context_chunks = retrieved_chunks[:MAX_CONTEXT_CHUNKS]
        context_text   = _format_context_blocks(context_chunks)
        user_message   = CONTEXT_TEMPLATE.format(
            context_blocks=context_text,
            question=query,
        )

        # 注入历史（history 为空时等同原来的行为）
        messages = _build_messages_with_history(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            history=history or [],
        )

        # 调用 OpenAI
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI 调用失败: {e}")
            return RAGResponse(
                answer="", sources=context_chunks, refused=True,
                refuse_reason=f"LLM 调用出错: {str(e)}",
            )

        # 第二道拒答
        if answer.startswith("INSUFFICIENT_EVIDENCE"):
            reason = answer.replace("INSUFFICIENT_EVIDENCE:", "").strip()
            logger.info(f"LLM 拒答: {reason}")
            return RAGResponse(
                answer="", sources=context_chunks, refused=True, refuse_reason=reason,
            )

        logger.info(f"生成回答，{len(answer)} 字符，{len(context_chunks)} 个来源")
        return RAGResponse(answer=answer, sources=context_chunks, refused=False)

    # ── 流式生成 ───────────────────────────────────────────────────────────────

    async def stream(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        history: Optional[list[dict]] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        LangChain LCEL 流式生成，yield SSE 事件字典。

        Args:
            query:            用户当前问题（已经过 rewrite_query 改写）
            retrieved_chunks: 检索结果
            history:          对话历史，None 或空列表 = 单次问答模式

        事件类型：
          {"type": "token",  "content": "..."}              — 每个文字片段
          {"type": "done",   "refused": false, "sources": [...]} — 正常结束
          {"type": "done",   "refused": true,  "reason": "..."}  — 拒答
          {"type": "error",  "message": "..."}              — 异常
        """
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        # 第一道拒答
        refuse, reason = _should_refuse(retrieved_chunks)
        if refuse:
            logger.info(f"[stream] 前置拒答: {reason}")
            yield {"type": "done", "refused": True, "reason": reason, "sources": []}
            return

        # 构建上下文
        context_chunks = retrieved_chunks[:MAX_CONTEXT_CHUNKS]
        context_text   = _format_context_blocks(context_chunks)
        user_message   = CONTEXT_TEMPLATE.format(
            context_blocks=context_text,
            question=query,
        )

        # 把 history dict 列表转成 LangChain message 对象
        # LangChain 的 from_messages 支持混合列表：
        # ("system", ...) 元组 + LangChain message 对象 + ("human", ...) 元组
        lc_history = []
        for msg in (history or []):
            if msg["role"] == "user":
                lc_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_history.append(AIMessage(content=msg["content"]))

        # 构建完整 messages 列表
        all_messages = (
            [SystemMessage(content=SYSTEM_PROMPT)]
            + lc_history
            + [HumanMessage(content=user_message)]
        )

        llm   = ChatOpenAI(
            model=self.model, temperature=TEMPERATURE,
            max_tokens=1024, streaming=True, api_key=self.api_key,
        )

        # 直接用 llm.astream(messages)，不需要 ChatPromptTemplate
        # 因为 messages 已经是完整列表，无需再套 prompt 模板
        full_answer = ""
        try:
            async for chunk in llm.astream(all_messages):
                token = chunk.content
                if token:
                    full_answer += token
                    yield {"type": "token", "content": token}

        except Exception as e:
            logger.error(f"[stream] LangChain 流式调用失败: {e}")
            yield {"type": "error", "message": str(e)}
            return

        # 第二道拒答
        if full_answer.strip().startswith("INSUFFICIENT_EVIDENCE"):
            reason = full_answer.replace("INSUFFICIENT_EVIDENCE:", "").strip()
            logger.info(f"[stream] LLM 拒答: {reason}")
            yield {"type": "done", "refused": True, "reason": reason, "sources": []}
            return

        # 正常结束
        sources_data = [
            {
                "source":        rc.chunk.source,
                "page_num":      rc.chunk.page_num,
                "section_title": rc.chunk.section_title,
                "score":         round(rc.score, 4),
                "rank":          rc.rank,
                "excerpt":       rc.chunk.text[:200],
            }
            for rc in context_chunks
        ]
        logger.info(f"[stream] 完成，{len(full_answer)} 字符，{len(sources_data)} 个来源")
        yield {"type": "done", "refused": False, "reason": None, "sources": sources_data}
