# RAG Project

基于检索增强生成（Retrieval-Augmented Generation）的知识问答系统。

上传文档后，系统从文档中检索相关内容，结合 OpenAI 生成带引用来源的回答。证据不足时主动拒答，不胡编。

---

## 项目目标

- 上传文档（PDF、Word、Excel 等），针对文档内容提问
- 回答必须基于文档内容，每条陈述标注来源（文件名 + 页码）
- 证据不足时拒绝回答，给出明确理由
- 支持多轮对话，追问能自动理解上下文
- 支持流式输出，回答逐字出现
- 所有请求记录日志，支持评估检索质量

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        INGEST 流程                           │
│                                                             │
│  文档上传 → 解析（文本+页码+章节）→ 分块（512字符）            │
│         → 嵌入（本地 sentence-transformers）                 │
│         → 存入 ChromaDB（本地持久化）                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        QUERY 流程                            │
│                                                             │
│  用户提问                                                    │
│     │                                                       │
│     ├─[多轮模式]─→ Query Rewriting（LLM 改写追问为完整问题）  │
│     │                                                       │
│     ↓                                                       │
│  向量检索（native: 余弦相似度 / llamaindex: similarity|MMR） │
│     │                                                       │
│     ↓                                                       │
│  Pre-LLM 拒答判断（score < 0.35 → 直接拒答）                 │
│     │                                                       │
│     ↓                                                       │
│  构建 Prompt（上下文 + 对话历史 → OpenAI）                    │
│     │                                                       │
│     ├─[普通模式]─→ 一次性返回                                │
│     └─[流式模式]─→ SSE 逐 token 推送                        │
│                                                             │
│  Post-LLM 拒答判断（INSUFFICIENT_EVIDENCE → 拒答）           │
│     │                                                       │
│     ↓                                                       │
│  返回回答 + 引用列表 + 耗时，写入 SQLite 日志                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 目录结构

```
RAGProject/
├── app/
│   ├── models.py                   # 核心数据结构（ChunkDoc、RAGResponse 等）
│   ├── main.py                     # FastAPI 入口，管理共享资源生命周期
│   ├── ingestion/
│   │   ├── parsers.py              # 各格式解析器（PDF/DOCX/XLSX/CSV/TXT/MD）
│   │   ├── chunker.py              # 文本分块 + chunk_id 生成
│   │   └── pipeline.py             # 串联 解析→分块→存储 的完整流程
│   ├── retrieval/
│   │   ├── vector_store.py         # native 后端：SentenceTransformer + ChromaDB
│   │   ├── llamaindex_store.py     # LlamaIndex 后端：HuggingFaceEmbedding + MMR
│   │   └── factory.py              # 工厂函数：根据 RAG_BACKEND 创建对应后端
│   ├── generation/
│   │   └── generator.py            # 生成模块：同步/流式/Query Rewriting/拒答逻辑
│   ├── api/
│   │   └── routes.py               # FastAPI 路由：接收请求，调用模块，管理 Session
│   └── logging/
│       └── db.py                   # SQLite 查询日志（SQLAlchemy）
├── evaluation/
│   ├── eval.py                     # 评估脚本（Hit@k、MRR、拒答准确率）
│   └── qa_set.json                 # 测试问题集（需手动准备）
├── tests/                          # 单元测试（待补充）
├── data/
│   ├── raw/                        # 原始文档（PDF、DOCX 等）
│   ├── processed/                  # 分块后的 JSONL 中间文件（调试用）
│   ├── chroma/                     # ChromaDB 向量库（自动生成）
│   └── logs/                       # SQLite 查询日志（自动生成）
├── frontend.html                   # 前端测试面板（Chat UI，无需额外依赖）
├── architecture.html               # 系统架构可视化
├── Dockerfile                      # 多阶段构建，预缓存嵌入模型
├── docker-compose.yml              # 含持久化 volume 配置
├── requirements.txt                # Python 依赖
├── .env.example                    # 环境变量模板
└── .gitignore
```

---

## 支持的文档格式

| 格式 | 状态 | 说明 |
|------|------|------|
| PDF  | ✅ 支持 | pdfplumber 解析，PyMuPDF 兜底 |
| DOCX | ✅ 支持 | 保留 Heading 层级作为章节信息 |
| XLSX | ✅ 支持 | 每个 Sheet 单独处理，行列转文本 |
| CSV  | ✅ 支持 | 每 100 行打包一页 |
| TXT  | ✅ 支持 | 纯文本直接读取 |
| MD   | ✅ 支持 | 按 H1/H2 标题切分章节 |
| PPTX | 🔜 计划中 | |
| HTML | 🔜 计划中 | |

---

## 快速开始

### 环境准备

```bash
git clone https://github.com/DawnZYC/RAG_Project.git
cd RAG_Project

# 推荐 Python 3.11
conda create -n rag python=3.11 -y
conda activate rag

pip install -r requirements.txt
```

### 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 OpenAI API Key
```

最简配置（其他都有默认值）：

```env
OPENAI_API_KEY=sk-...
```

### 本地运行

```bash
uvicorn app.main:app --reload
```

启动成功后：
- API 文档：`http://localhost:8000/docs`
- 前端面板：直接用浏览器打开 `frontend.html`（无需 serve，双击即可）

### Docker 运行

```bash
docker-compose up --build
```

首次 build 会把嵌入模型打进镜像（约 130MB），之后启动秒级就绪。

---

## 前端使用

打开 `frontend.html`，功能包括：

- **上传文档**：拖拽或点击，支持 PDF/DOCX/XLSX/CSV/TXT/MD
- **提问**：`Cmd+Enter`（Mac）或 `Ctrl+Enter` 快速提交
- **普通模式** vs **流式模式**：流式模式下回答逐字出现（SSE）
- **多轮对话**：勾选"开启多轮对话"后自动生成 Session UUID，追问能理解上下文
- **新对话**：清空当前 Session，重新开始
- **引用来源**：点击折叠按钮展开，查看文件名 + 页码 + 相关度分数
- **拒答提示**：证据不足时显示橙色告警和具体原因

---

## API 接口

### 上传文档

```bash
POST /api/v1/ingest

curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@report.pdf"
```

```json
{
  "filename": "report.pdf",
  "chunks_added": 42,
  "status": "success"
}
```

### 提问（普通模式）

```bash
POST /api/v1/query

# 单次问答
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "退款政策是什么？", "top_k": 5}'

# 多轮对话（传 session_id 保持上下文）
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "国际订单呢？", "session_id": "550e8400-e29b-41d4-a716-446655440000"}'

# 只在指定文件中检索
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "filter_source": "report.pdf"}'
```

正常回答：

```json
{
  "answer": "退款政策规定... [来源: report.pdf, 第3页]",
  "refused": false,
  "refuse_reason": null,
  "sources": [
    {
      "source": "report.pdf",
      "page_num": 3,
      "section_title": "退款条款",
      "score": 0.82,
      "rank": 1,
      "excerpt": "..."
    }
  ],
  "latency_ms": 1243.5,
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

拒答时：

```json
{
  "answer": "",
  "refused": true,
  "refuse_reason": "最高匹配分数（0.21）低于置信度阈值（0.35），证据不足，无法可靠回答。",
  "sources": [],
  "latency_ms": 45.2
}
```

### 提问（流式模式）

```bash
POST /api/v1/stream
```

请求体与 `/query` 相同，响应为 SSE 流：

```
data: {"type": "token", "content": "退款"}
data: {"type": "token", "content": "政策"}
...
data: {"type": "done", "refused": false, "sources": [...], "session_id": "..."}
```

拒答时：

```
data: {"type": "done", "refused": true, "reason": "证据不足..."}
```

异常时：

```
data: {"type": "error", "message": "..."}
```

### 其他接口

```bash
GET    /api/v1/health                    # 健康检查
GET    /api/v1/stats                     # 向量库统计（chunk 总数、活跃 Session 数）
DELETE /api/v1/document/{filename}       # 删除某个文档的所有 chunk
DELETE /api/v1/session/{session_id}      # 清空会话历史（前端"新对话"时调用）
GET    /api/v1/session/{session_id}      # 查看会话历史（调试用）
```

---

## 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `OPENAI_API_KEY` | **必填** | OpenAI API Key |
| `LLM_MODEL` | `gpt-4o-mini` | 生成用的 LLM 模型 |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | 嵌入模型，中文文档换 `bge-small-zh-v1.5` |
| `RAG_BACKEND` | `native` | 检索后端：`native` / `llamaindex` |
| `RETRIEVAL_MODE` | `similarity` | 检索模式（仅 llamaindex）：`similarity` / `mmr` |
| `MMR_THRESHOLD` | `0.7` | MMR 多样性权重（仅 mmr 模式）：0.0=最多样，1.0=最相关 |
| `CHROMA_DIR` | `data/chroma` | ChromaDB 存储路径 |
| `PROCESSED_DIR` | `data/processed` | JSONL 中间文件路径 |
| `DB_PATH` | `data/logs/queries.db` | SQLite 日志路径 |

---

## 检索后端对比

通过 `RAG_BACKEND` 切换，API 接口和生成层完全不变：

| 能力 | native | llamaindex |
|------|--------|------------|
| 嵌入模型 | SentenceTransformer（本地） | HuggingFaceEmbedding（本地，同款模型） |
| 检索策略 | 纯余弦相似度 | similarity / MMR 可切换 |
| 结果多样性控制 | 无 | MMR 减少重复结果 |
| 元数据过滤 | ChromaDB where 条件 | LlamaIndex MetadataFilters |
| 学习价值 | 理解 RAG 原理 | 体验框架抽象和扩展性 |

**切换后端注意**：两个后端的 embedding 格式不兼容，切换后必须清空向量库并重新 ingest：

```bash
rm -rf data/chroma/*
# 重启服务后重新上传文档
```

---

## 多轮对话说明

### 为什么需要 Query Rewriting

没有改写时，追问的向量语义很模糊：

```
Q1: "退款政策是什么？"  →  检索正常
Q2: "国际订单呢？"      →  向量库里没有"国际订单呢"，检索失败
Q3: "那第二条规则呢？"  →  "第二条"完全没有语义，必然检索失败
```

加了 Query Rewriting 后，LLM 在检索前先把追问改写为完整问题：

```
历史: Q1="退款政策？" A1="30天内可退..."
追问: "国际订单呢？"
  ↓ rewrite_query()（LLM 改写，temperature=0）
完整问题: "国际订单的退款政策是什么？"
  ↓ 用完整问题检索，命中准确
```

### 实现细节

- Session 用 UUID 标识，历史存在 `app.state.sessions`（内存字典，服务重启后清空）
- 滑动窗口 k=5 轮：手动 list 切片，超出就删最旧一轮（2条消息）
- 历史注入为真实 OpenAI message 对象（SystemMessage / HumanMessage / AIMessage），不是文本拼接
- 无历史时 `rewrite_query()` 直接返回原问题，不调用 LLM，节省 Token
- 改写失败时 fallback 到原问题，不影响主流程

---

## 拒答机制

系统有两道独立的拒答判断：

**第一道：Pre-LLM（分数阈值）**

检索完成后立即判断，不调用 LLM，响应极快：
- 知识库为空 → 拒答
- `top_score < 0.35` → 拒答（`SCORE_THRESHOLD` in `vector_store.py`）

**第二道：Post-LLM（语义判断）**

LLM 生成后检查回答前缀：
- 回答以 `INSUFFICIENT_EVIDENCE:` 开头 → 拒答
- 由 System Prompt 指导 LLM 在上下文不足时自行判断

---

## 评估

准备好测试问题集后运行（格式见 `evaluation/eval.py` 开头注释）：

```bash
# 评估 native 后端
python -m evaluation.eval --qa-file evaluation/qa_set.json

# 评估 LlamaIndex 后端
python -m evaluation.eval --qa-file evaluation/qa_set.json --backend llamaindex

# 对比两个后端（side-by-side 表格）
python -m evaluation.eval --qa-file evaluation/qa_set.json --compare

# 只跑检索评估，跳过 LLM 调用（省 Token）
python -m evaluation.eval --qa-file evaluation/qa_set.json --skip-generation
```

输出指标：
- **Hit@k**：top-k 结果中包含正确来源文件的比例
- **MRR**：正确来源排名的平均倒数（越高说明正确结果排得越靠前）
- **拒答准确率**：TP/FP/FN 分类，分析拒答是否合理

---

## 技术栈

| 模块 | 技术选型 | 说明 |
|------|---------|------|
| API 框架 | FastAPI + Uvicorn | 异步，lifespan 管理共享资源 |
| 文档解析 | pdfplumber / PyMuPDF / python-docx / openpyxl | 各格式独立解析器 |
| 文本分块 | langchain-text-splitters | RecursiveCharacterTextSplitter，format-aware |
| 嵌入模型 | sentence-transformers / LlamaIndex HuggingFaceEmbedding | 本地运行，无 API 费用 |
| 向量存储 | ChromaDB（本地持久化） | 余弦相似度，cosine space |
| 检索层 | native（直连）/ LlamaIndex（MMR 等高级策略） | 工厂模式切换 |
| 生成层 | OpenAI SDK + LangChain LCEL | 同步生成 + 流式生成 |
| 流式输出 | LangChain `astream()` + FastAPI `StreamingResponse` + SSE | token 级推送 |
| 多轮 Memory | 手动 list 滑动窗口（k=5）+ LLM Query Rewriting | 无额外依赖 |
| 查询日志 | SQLite + SQLAlchemy | 每次请求完整记录 |
| 容器化 | Docker 多阶段构建 + docker-compose | 嵌入模型预缓存进镜像 |

---

## 学习路径与进度

本项目按三个阶段推进，每个阶段对应不同技术深度：

### 第一阶段：native MVP ✅

掌握 RAG pipeline 每一步细节：文档解析 → 分块 → 本地嵌入 → ChromaDB → 余弦相似度检索 → 拒答 → 引用来源。全部自己实现，没有框架封装。

核心文件：`parsers.py` / `chunker.py` / `pipeline.py` / `vector_store.py` / `generator.py`

### 第二阶段：LlamaIndex 检索优化 ✅

在检索层引入 LlamaIndex，对比不同检索策略（similarity vs MMR）对结果质量的影响。通过 `RAG_BACKEND` 环境变量热切换后端，API 接口和生成层完全不变。

核心文件：`llamaindex_store.py` / `factory.py`

### 第三阶段：LangChain 生成层增强 ✅

LangChain 用于生成层（不用于检索）：LCEL 流式输出、多轮对话 Memory、LLM Query Rewriting。

| 功能 | 状态 |
|------|------|
| 流式输出（SSE） | ✅ 完成 |
| 多轮对话 Memory | ✅ 完成 |
| Query Rewriting | ✅ 完成 |
| 前端 Chat UI | ✅ 完成 |

核心文件：`generator.py` / `routes.py` / `frontend.html`

### 后续计划

- [ ] 单元测试（parsers / chunker / 拒答逻辑）
- [ ] 构建 `qa_set.json`，跑评估并对比两个后端
- [ ] 支持 PPTX / HTML 格式
- [ ] Reranker（Cross-Encoder 重排序）
- [ ] 持久化 Session（Redis，解决服务重启后历史清空问题）
