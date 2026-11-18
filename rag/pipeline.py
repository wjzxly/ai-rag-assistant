"""把加载、分块、索引、生成串成一条流水线。

对外只暴露两个动作：``ingest()`` 建库、``query()`` 提问。
CLI 与 HTTP 服务都只是这层的薄壳，保证两种入口行为一致。
"""

from __future__ import annotations

import gzip
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .chunker import DEFAULT_MAX_CHARS, chunk_documents
from .embedder import Embedder, build_embedder
from .generator import LLMGenerator, build_messages, extractive_answer
from .index import HybridIndex
from .loader import load_dir
from .models import Answer, Chunk, Hit

INDEX_VERSION = 1


@dataclass
class IngestStats:
    docs: int
    chunks: int
    chars: int
    elapsed_ms: float


class RAGPipeline:
    """一个可独立跑起来的 RAG 实例。"""

    def __init__(
        self,
        docs_dir: str | Path,
        embedder_kind: str = "hashing",
        dim: int = 1024,
        max_chars: int = DEFAULT_MAX_CHARS,
        top_k: int = 5,
        use_llm: bool = True,
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self.embedder: Embedder = build_embedder(embedder_kind, dim=dim)
        self.max_chars = max_chars
        self.top_k = top_k
        self.generator = LLMGenerator() if use_llm else None
        self.index: Optional[HybridIndex] = None

    # ---------- 建库 ----------

    def ingest(self) -> IngestStats:
        started = time.perf_counter()
        docs = load_dir(self.docs_dir)
        chunks = chunk_documents(docs, max_chars=self.max_chars)
        self.index = HybridIndex(chunks, self.embedder).build()
        elapsed = (time.perf_counter() - started) * 1000
        return IngestStats(
            docs=len(docs),
            chunks=len(chunks),
            chars=sum(len(d.text) for d in docs),
            elapsed_ms=round(elapsed, 1),
        )

    # ---------- 检索 ----------

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[Hit]:
        if self.index is None:
            raise RuntimeError("索引尚未构建，请先调用 ingest() 或 load()")
        return self.index.search(question, top_k=top_k or self.top_k)

    # ---------- 问答 ----------

    def query(self, question: str, top_k: Optional[int] = None) -> Answer:
        started = time.perf_counter()
        hits = self.retrieve(question, top_k=top_k)

        if not hits:
            return Answer(
                question=question,
                text="知识库中没有检索到相关内容。请换一种问法，或确认文档已导入。",
                hits=(),
                mode="extractive",
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
                warning="no_hits",
            )

        warning = None
        if self.generator is not None and self.generator.available:
            try:
                text = self.generator.chat(build_messages(question, hits))
                mode = "llm"
            except RuntimeError as exc:
                # 接口挂了不能整个服务不可用，降级到抽取式
                text = extractive_answer(question, hits)
                mode = "extractive"
                warning = f"LLM 调用失败，已降级：{exc}"
        else:
            text = extractive_answer(question, hits)
            mode = "extractive"
            warning = "未配置 LLM_API_KEY，使用抽取式回答"

        return Answer(
            question=question,
            text=text,
            hits=tuple(hits),
            mode=mode,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            warning=warning,
        )

    # ---------- 持久化 ----------

    def save(self, path: str | Path) -> Path:
        """把索引落盘。

        存向量而不是"只存文本、加载时重算"：接上付费嵌入模型后，
        重算一次全库是要花钱的，而且离线环境根本重算不了。
        """
        if self.index is None:
            raise RuntimeError("没有可保存的索引")
        path = Path(path)
        payload = {
            "version": INDEX_VERSION,
            "embedder": self.embedder.name,
            "dim": self.embedder.dim,
            "max_chars": self.max_chars,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "heading": c.heading,
                    "text": c.text,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                }
                for c in self.index.chunks
            ],
            # 量化到 6 位小数，体积能省一半以上，检索排序不受影响
            "vectors": [
                [round(v, 6) for v in vec] for vec in self.index._vectors  # noqa: SLF001
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        return path

    def load(self, path: str | Path) -> "RAGPipeline":
        path = Path(path)
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("version") != INDEX_VERSION:
            raise RuntimeError(
                f"索引格式版本不匹配（文件 {data.get('version')}，期望 {INDEX_VERSION}），请重新 ingest"
            )
        saved_embedder = data.get("embedder")
        if saved_embedder and saved_embedder != self.embedder.name:
            # 不同嵌入模型的向量空间不可混用，混了排序会变成噪声
            raise RuntimeError(
                f"索引由 {saved_embedder} 构建，当前配置是 {self.embedder.name}，"
                "两者向量不可混用；请重新 ingest 或改回原来的嵌入后端"
            )
        chunks = [
            Chunk(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                source=c["source"],
                heading=c["heading"],
                text=c["text"],
                start_line=c["start_line"],
                end_line=c["end_line"],
            )
            for c in data["chunks"]
        ]
        self.max_chars = data.get("max_chars", self.max_chars)
        idx = HybridIndex(chunks, self.embedder)
        # 直接注入已存的向量，跳过 embed
        idx._vectors = [list(v) for v in data["vectors"]]  # noqa: SLF001
        idx.build_lexical()
        self.index = idx
        return self

    # ---------- 辅助 ----------

    def stats(self) -> dict:
        if self.index is None:
            return {"chunks": 0, "sources": []}
        sources = sorted({c.source for c in self.index.chunks})
        return {
            "chunks": len(self.index.chunks),
            "sources": sources,
            "embedder": self.embedder.name,
        }


def build_pipeline_from_env(docs_dir: str | Path) -> RAGPipeline:
    """按环境变量构造流水线，供 CLI 与服务共用。"""
    return RAGPipeline(
        docs_dir=docs_dir,
        embedder_kind=os.environ.get("RAG_EMBEDDER", "hashing"),
        dim=int(os.environ.get("RAG_DIM", "1024")),
        max_chars=int(os.environ.get("RAG_MAX_CHARS", str(DEFAULT_MAX_CHARS))),
        top_k=int(os.environ.get("RAG_TOP_K", "5")),
        use_llm=os.environ.get("RAG_NO_LLM", "") not in {"1", "true", "yes"},
    )
