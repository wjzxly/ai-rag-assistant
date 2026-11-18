"""混合检索：BM25 + 向量，用 RRF 融合。

为什么不用单一路径——这是面试里的高频追问，答案很具体：

- **纯向量**在本场景会漏招。财务与供应链文档里大量专有名词、单据编号、
  物料编码（如 ``PO-2024-0912``、``GSK-318``），稠密向量对这类"低频精确串"
  不敏感，反而是 BM25 的强项。而默认的 hashing 嵌入本身就是词袋，
  更不能指望它理解同义改写。
- **纯 BM25** 又解决不了"报销要什么材料"↔"差旅费用需要哪些附件"这类
  字面不重合的问法。

所以两路各自排序，再用 **RRF（Reciprocal Rank Fusion）** 合并：
``score = Σ 1 / (k + rank)``，k 取 60。

选 RRF 而不是加权求和，是因为两路分数量纲完全不同（BM25 无上界、余弦在 [-1,1]），
归一化会引入额外假设；RRF 只用名次，天然免疫量纲问题，且对单路的异常高分不敏感。
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from .embedder import Embedder
from .models import Chunk, Hit
from .text import tokenize, unique_terms

RRF_K = 60

# 相关性门槛默认值。定得低是有意的：误杀一条正确结果（false negative）比
# 多带一条边角片段更伤体验；这个值在示例评测集上验证过，recall 不掉。
DEFAULT_MIN_COVERAGE = 0.15


class BM25Index:
    """标准 Okapi BM25，参数用业界默认值。"""

    def __init__(
        self,
        tokenized_corpus: Sequence[Sequence[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus: List[List[str]] = [list(t) for t in tokenized_corpus]
        self.n_docs = len(self.corpus)
        self.doc_len = [len(t) for t in self.corpus]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0

        self.tf: List[Counter] = [Counter(t) for t in self.corpus]
        df: Counter = Counter()
        for terms in self.corpus:
            df.update(set(terms))
        self.df = df
        self.idf: Dict[str, float] = {
            term: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }

    def search(self, query: Sequence[str], top_k: int = 50) -> List[Tuple[int, float]]:
        if not self.n_docs:
            return []
        q_terms = [t for t in query if t in self.idf]
        if not q_terms:
            return []
        scores: List[Tuple[int, float]] = []
        for i, tf in enumerate(self.tf):
            dl = self.doc_len[i] or 1
            s = 0.0
            for term in q_terms:
                f = tf.get(term)
                if not f:
                    continue
                idf = self.idf[term]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += idf * (f * (self.k1 + 1)) / denom
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


class HybridIndex:
    """把片段索引起来，提供融合检索。"""

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder) -> None:
        self.chunks: List[Chunk] = list(chunks)
        self.embedder = embedder
        self._vectors: List[List[float]] = []
        self._bm25: Optional[BM25Index] = None
        self._doc_terms: List[set] = []
        self._built = False

    # ---------- 构建 ----------

    def _indexable_text(self, chunk: Chunk) -> str:
        """参与索引的文本 = 标题路径 + 正文。

        标题路径算进去，是为了让「发票要求」这个上下文参与匹配，
        否则片段里只剩孤零零的条款，召回和排序都会掉。
        """
        return f"{chunk.heading}\n{chunk.text}"

    def build_lexical(self) -> "HybridIndex":
        """只建词法索引（BM25），向量由调用方决定是否复用已存结果。"""
        texts = [self._indexable_text(c) for c in self.chunks]
        tokenized = [tokenize(t) for t in texts]
        # 缓存词集合，检索时统计 matched_terms 不必反复分词
        self._doc_terms = [set(t) for t in tokenized]
        self._bm25 = BM25Index(tokenized)
        self._built = True
        return self

    def build(self) -> "HybridIndex":
        texts = [self._indexable_text(c) for c in self.chunks]
        # 让嵌入后端在语料上拟合（只有 hashing 后端有 fit）
        fit = getattr(self.embedder, "fit", None)
        if callable(fit):
            fit(texts)
        self._vectors = self.embedder.embed(texts) if texts else []
        self.build_lexical()
        return self

    def _ensure_built(self) -> None:
        if not self._built:
            self.build()

    # ---------- 检索 ----------

    def _vector_ranking(self, query: str, limit: int) -> List[Tuple[int, float]]:
        if not self._vectors:
            return []
        qv = self.embedder.embed([query])[0]
        scored: List[Tuple[int, float]] = []
        for i, vec in enumerate(self._vectors):
            # 向量已 L2 归一化，点积即余弦
            dot = sum(a * b for a, b in zip(qv, vec))
            if dot > 0:
                scored.append((i, dot))
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    def _lexical_ranking(self, query: str, limit: int) -> List[Tuple[int, float]]:
        assert self._bm25 is not None
        return self._bm25.search(tokenize(query), top_k=limit)

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 50,
        source_filter: Optional[str] = None,
        min_coverage: float = DEFAULT_MIN_COVERAGE,
    ) -> List[Hit]:
        """融合检索。

        ``candidate_k`` 是每一路的候选数，通常远大于 ``top_k``。

        ``min_coverage`` 是相关性门槛：问题里的实词至少要有这个比例出现在片段里，
        否则宁可不返回。没有这道门槛时，随便问一句"量子计算怎么制备"都会被
        拼出几段"材料""计算"沾边的片段，然后模型拿着这些垃圾硬答——
        这是 RAG 幻觉最常见的来源，问题不在生成，在检索没学会说"不知道"。
        """
        self._ensure_built()
        if not self.chunks:
            return []

        vec_rank = self._vector_ranking(query, candidate_k)
        lex_rank = self._lexical_ranking(query, candidate_k)

        fused: Dict[int, float] = {}
        detail: Dict[int, Dict[str, float]] = {}
        for rank, (idx, score) in enumerate(vec_rank, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)
            detail.setdefault(idx, {})["vector_score"] = score
        for rank, (idx, score) in enumerate(lex_rank, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)
            detail.setdefault(idx, {})["lexical_score"] = score

        q_terms = set(unique_terms(query))
        # 覆盖率只统计二元组与英文词，不统计单字。
        # 中文单字极容易偶然命中（问"量子计算"，片段里有"材料"就凑上了"材""料"），
        # 把它们算进去会让门槛形同虚设。
        content_terms = sorted((t for t in q_terms if len(t) >= 2), key=len, reverse=True)
        if not content_terms:
            content_terms = list(q_terms)

        # 纯 RRF 只用名次，会丢掉"命中强度"：一个只碰到"周期"两个字的片段，
        # 和一个完整命中「GSK-318 + 采购 + 周期」的片段，在 RRF 眼里名次可能只差两位，
        # 融合分几乎相同，于是前者靠另一路排名靠前就反超了。
        # 这里乘一个覆盖度因子——覆盖越多问题词的片段越相关，最低保留 50% 分数
        # 以免把"字面不重合但语义相关"的向量召回彻底打死。
        def coverage_of(idx: int) -> float:
            if not content_terms:
                return 0.0
            return len(set(content_terms) & self._doc_terms[idx]) / len(content_terms)

        scored: List[Tuple[int, float, float]] = []
        for idx, rrf in fused.items():
            cov = coverage_of(idx)
            if content_terms and cov < min_coverage:
                continue
            if source_filter and source_filter not in self.chunks[idx].source:
                continue
            scored.append((idx, rrf * (0.5 + 0.5 * cov), cov))
        scored.sort(key=lambda t: -t[1])

        hits: List[Hit] = []
        for idx, final, _cov in scored[:top_k]:
            chunk = self.chunks[idx]
            covered = set(content_terms) & self._doc_terms[idx]
            hits.append(
                Hit(
                    chunk=chunk,
                    score=final,
                    vector_score=detail[idx].get("vector_score", 0.0),
                    lexical_score=detail[idx].get("lexical_score", 0.0),
                    rank=len(hits) + 1,
                    matched_terms=tuple(t for t in content_terms if t in covered)[:6],
                )
            )
        return hits

    def __len__(self) -> int:
        return len(self.chunks)
