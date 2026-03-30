# RAG Project

基于检索增强生成（Retrieval-Augmented Generation）的知识问答系统。
系统从你的文档中检索相关内容，结合 OpenAI 生成带引用来源的回答，并在证据不足时主动拒答。

---

## 项目目标

- 上传文档（PDF、Word、Excel 等）后，能够针对文档内容提问
- 回答必须基于文档内容，每条陈述都标注来源（文件名 + 页码）
- 证据不足时拒绝回答，而不是胡编
- 所有请求记录日志，支持后续评估检索质量

---

## 整体架构

```
文档上传
   ↓
解析（提取文本 + 页码 + 章节）
   ↓
分块（切成 512 字符小块，保留 metadata）
   ↓
嵌入（sentence-transformers 转向量）
   ↓
存储（ChromaDB 本地向量库）

用户提问
   ↓
检索（向量相似度，返回 top-k 块）
   ↓
拒答判断（分数低于阈值 → 直接拒答）
   ↓
生成（OpenAI GPT，Prompt 要求引用来源）
   ↓
返回回答 + 引用列表 + 耗时
```

---

## 目录结构

```
RAGProject/
├── app/
│   ├── models.py              # 核心数据结构（ChunkDoc、RAGResponse 等）
│   ├── main.py                # FastAPI 入口，管理共享资源的生命周期
│   ├── ingestion/
│   │   ├── parsers.py         # 各格式文档解析器
│   │   ├── chunker.py         # 文本分块 + chunk_id 生成
│   │   └── pipeline.py        # 串联解析→分块→存储的完整流程
│   ├── retrieval/
│   │   └── vector_store.py    # ChromaDB 封装 + 嵌入模型
│   ├── generation/
│   │   └── generator.py       # OpenAI 调用 + 拒答逻辑
│   ├── api/
│   │   └── routes.py          # FastAPI 路由定义
│   └── logging/
│       └── db.py              # SQLite 查询日志
├── data/
│   ├── raw/                   # 放原始文档（PDF、DOCX 等）
│   └── processed/             # 分块后的 JSONL 中间文件（调试用）
├── evaluation/
│   ├── eval.py                # 评估脚本（Hit@k、MRR、拒答率）
│   └── qa_set.json            # 测试问题集
├── tests/                     # 单元测试
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 支持的文档格式

| 格式 | 状态 | 说明 |
|------|------|------|
| PDF  | ✅ 支持 | pdfplumber 解析，PyMuPDF 兜底 |
| DOCX | ✅ 支持 | 保留标题层级作为章节信息 |
| XLSX | ✅ 支持 | 每个 Sheet 单独处理 |
| CSV  | ✅ 支持 | 每 100 行一页 |
| TXT  | ✅ 支持 | 纯文本直接读取 |
| MD   | ✅ 支持 | 按 H1/H2 标题切分章节 |
| PPTX | 🔜 计划中 | |
| HTML | 🔜 计划中 | |
| 图片（OCR）| 🔜 计划中 | |

---

## 快速开始

### 1. 克隆项目，创建环境

```bash
git clone <your-repo-url>
cd RAGProject

conda create -n rag python=3.11 -y
conda activate rag

pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 OpenAI API Key
```

### 3. 本地直接运行

```bash
uvicorn app.main:app --reload
```

### 4. 或者用 Docker 启动

```bash
docker-compose up --build
```

服务启动后访问 `http://localhost:8000/docs` 可以看到交互式 API 文档。

---

## API 接口

### 上传文档

```bash
POST /api/v1/ingest

curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@你的文件.pdf"
```

返回：
```json
{
  "filename": "你的文件.pdf",
  "chunks_added": 42,
  "status": "success"
}
```

### 提问

```bash
POST /api/v1/query

curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "文档里说了什么？", "top_k": 5}'
```

返回（正常回答）：
```json
{
  "answer": "文档指出... [Source: 你的文件.pdf, Page 3]",
  "refused": false,
  "refuse_reason": null,
  "sources": [
    {
      "source": "你的文件.pdf",
      "page_num": 3,
      "section_title": "第一章",
      "score": 0.82,
      "rank": 1,
      "excerpt": "..."
    }
  ],
  "latency_ms": 1243.5
}
```

返回（证据不足，拒答）：
```json
{
  "answer": "",
  "refused": true,
  "refuse_reason": "最高匹配分数 0.21 低于置信度阈值，证据不足。",
  "sources": [],
  "latency_ms": 45.2
}
```

### 其他接口

```bash
GET  /api/v1/health              # 健康检查
GET  /api/v1/stats               # 向量库统计（chunk 总数等）
DELETE /api/v1/document/{filename}  # 删除某个文档的所有 chunk
```

---

## 环境变量说明

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `OPENAI_API_KEY` | 必填 | OpenAI API Key |
| `LLM_MODEL` | `gpt-4o-mini` | 使用的 LLM 模型 |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | 嵌入模型（中文文档换 bge-small-zh-v1.5） |
| `CHROMA_DIR` | `/data/chroma` | 向量库存储路径 |
| `PROCESSED_DIR` | `/data/processed` | 分块中间文件路径 |
| `DB_PATH` | `/data/logs/queries.db` | SQLite 日志路径 |

---

## 评估

准备好测试问题集后运行：

```bash
python -m evaluation.eval --qa-file evaluation/qa_set.json
```

输出指标：
- **Hit@k**：top-k 结果中包含正确来源的比例
- **MRR**：正确来源排名的平均倒数
- **拒答率**：系统主动拒绝回答的比例

---

## 技术栈

| 模块 | 技术选型 |
|------|---------|
| API 框架 | FastAPI + Uvicorn |
| 文档解析 | pdfplumber、PyMuPDF、python-docx、openpyxl |
| 文本分块 | langchain-text-splitters |
| 嵌入模型 | sentence-transformers（本地运行） |
| 向量库 | ChromaDB（本地持久化） |
| LLM | OpenAI GPT（gpt-4o-mini） |
| 日志存储 | SQLite + SQLAlchemy |
| 容器化 | Docker + docker-compose |
