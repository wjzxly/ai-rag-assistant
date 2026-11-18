"""HTTP 服务层。

设计取向上刻意保守：**启动即建索引**，不做后台增量、不做多租户、不做鉴权。
这个项目的定位是"单机/内网可用、一个人能讲清楚每行代码"，加一层分布式
只会让面试官多问十个答不上来的问题，不会加分。

真要上生产，需要补的是：索引持久化到对象存储、按文档版本增量重建、
并发下的只读快照——这些写进 README 的「已知局限」里，比硬凑实现更诚实。
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .pipeline import RAGPipeline, build_pipeline_from_env

DOCS_DIR = os.environ.get("RAG_DOCS_DIR", "examples/docs")
INDEX_PATH = os.environ.get("RAG_INDEX_PATH", ".cache/index.json.gz")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="索引尚未就绪")
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载或构建索引。

    有缓存就读缓存（毫秒级），没有才现场建。这样容器重启不会每次都重新嵌入，
    接上付费嵌入模型后这一点很关键。
    """
    global _pipeline
    pipe = build_pipeline_from_env(DOCS_DIR)
    cache = Path(INDEX_PATH)
    if cache.exists():
        try:
            pipe.load(cache)
        except RuntimeError:
            stats = pipe.ingest()
            pipe.save(cache)
            print(f"[startup] 缓存不可用，已重建：{stats.chunks} 个片段")
    else:
        stats = pipe.ingest()
        pipe.save(cache)
        print(f"[startup] 已建索引：{stats.docs} 篇文档 / {stats.chunks} 个片段")
    _pipeline = pipe
    yield
    _pipeline = None


app = FastAPI(
    title="文档 RAG 问答服务",
    description="混合检索（BM25 + 向量，RRF 融合）+ 引用溯源",
    version="0.1.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(5, ge=1, le=20)


class Citation(BaseModel):
    rank: int
    source: str
    heading: str
    start_line: int
    end_line: int
    score: float
    preview: str


class AskResponse(BaseModel):
    question: str
    answer: str
    mode: str
    elapsed_ms: float
    warning: Optional[str] = None
    citations: List[Citation]


@app.get("/api/health")
def health():
    pipe = get_pipeline()
    return {"status": "ok", "embedder": pipe.embedder.name, "chunks": len(pipe.index.chunks) if pipe.index else 0}


@app.get("/api/stats")
def stats():
    return get_pipeline().stats()


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    pipe = get_pipeline()
    ans = pipe.query(req.question, top_k=req.top_k)
    return AskResponse(
        question=ans.question,
        answer=ans.text,
        mode=ans.mode,
        elapsed_ms=ans.elapsed_ms,
        warning=ans.warning,
        citations=[
            Citation(
                rank=h.rank,
                source=h.chunk.source,
                heading=h.chunk.heading,
                start_line=h.chunk.start_line,
                end_line=h.chunk.end_line,
                score=round(h.score, 6),
                preview=h.chunk.preview(),
            )
            for h in ans.hits
        ],
    )


@app.post("/api/ingest")
def ingest():
    """手动重建索引。文档更新后调用，或改由定时任务触发。"""
    pipe = get_pipeline()
    stats_ = pipe.ingest()
    pipe.save(INDEX_PATH)
    return {"docs": stats_.docs, "chunks": stats_.chunks, "elapsed_ms": stats_.elapsed_ms}


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    page = STATIC_DIR / "index.html"
    if not page.exists():
        return {"message": "前端未安装，直接调用 POST /api/ask"}
    return FileResponse(page)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "rag.app:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
