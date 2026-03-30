# ── 第一阶段：安装依赖 ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu


# ── 第二阶段：运行环境 ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# 从第一阶段复制已安装的包
COPY --from=builder /install /usr/local

# 在 build 阶段预下载嵌入模型并缓存进镜像
# 这样容器启动时不需要重新下载，启动速度快
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

# 复制应用代码
COPY app/ ./app/

# 创建数据目录
RUN mkdir -p /data/chroma /data/processed /data/logs

# 使用非 root 用户运行，提高安全性
RUN useradd -m -u 1000 appuser && chown -R appuser /app /data
USER appuser

EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
