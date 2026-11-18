# 运行镜像：自带示例语料，起容器就能问
#
# 换成自己的文档：把 RAG_DOCS_DIR 指到挂载进来的目录即可
#   docker run -p 8000:8000 -v /your/docs:/data/docs -e RAG_DOCS_DIR=/data/docs rag-assistant

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAG_DOCS_DIR=/app/examples/docs \
    RAG_INDEX_PATH=/app/.cache/index.json.gz

WORKDIR /app

# 先拷依赖描述，利用镜像层缓存：只改代码时不必重装依赖
COPY pyproject.toml README.md ./
COPY rag ./rag
COPY static ./static
COPY examples ./examples

RUN pip install -e ".[server]"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "rag.app:app", "--host", "0.0.0.0", "--port", "8000"]
