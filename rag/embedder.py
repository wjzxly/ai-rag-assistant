"""嵌入后端。

项目必须"零成本可跑"，所以默认提供一个纯标准库的 :class:`HashingEmbedder`：

- 用 hashing trick 把词映射到固定维度，签名哈希（±1）抵消部分碰撞偏差；
- 权重取 ``(1 + log tf) * idf``，再 L2 归一化，等价于一个稀疏 TF-IDF 向量；
- 维度 1024，几百个片段时内存和速度都毫无压力。

它当然不如 BGE / text-embedding-3 这类稠密向量模型。但它的作用是**保证 clone 下来
不配任何 key 就能跑通全流程**，而且给了一个可对照的基线：接上真模型之后
评测集上的提升是能量化的，不是"我感觉准多了"。

需要真模型时走 :class:`OpenAIEmbedder`，用标准库 urllib 直接请求
OpenAI 兼容的 ``/v1/embeddings``，不引入 requests / openai SDK。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from collections import Counter
from typing import Dict, List, Optional, Protocol, Sequence

from .text import tokenize


class Embedder(Protocol):
    """嵌入后端需要满足的接口。"""

    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> List[List[float]]: ...


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-12:
        return vec
    return [v / norm for v in vec]


def _signed_hash(token: str, dim: int) -> tuple[int, float]:
    """把词映射到维度下标与符号。用 blake2b 而非内置 hash()——
    后者带进程随机盐，重启后向量全变，索引就废了。"""
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    v = int.from_bytes(h, "big")
    return v % dim, (1.0 if (v >> 63) & 1 else -1.0)


class HashingEmbedder:
    """纯标准库的稀疏词袋嵌入，带可选 IDF 拟合。"""

    def __init__(self, dim: int = 1024) -> None:
        self.name = f"hashing-{dim}"
        self.dim = dim
        self._idf: Dict[str, float] = {}
        self._default_idf = 1.0

    def fit(self, corpus: Sequence[str]) -> "HashingEmbedder":
        """在语料上统计文档频率，得到 IDF。

        不 fit 也能用（退化为纯 TF），但 IDF 能显著压低"公司""员工"这类
        到处都出现的词，让检索更聚焦。
        """
        n = len(corpus)
        if n == 0:
            return self
        df: Counter = Counter()
        for doc in corpus:
            for term in set(tokenize(doc)):
                df[term] += 1
        # 平滑 IDF，避免除零
        self._idf = {
            term: math.log((n + 1) / (cnt + 0.5)) + 1.0 for term, cnt in df.items()
        }
        self._default_idf = math.log(n + 1) + 1.0
        return self

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        counts = Counter(tokenize(text))
        for term, tf in counts.items():
            idx, sign = _signed_hash(term, self.dim)
            idf = self._idf.get(term, self._default_idf)
            vec[idx] += sign * (1.0 + math.log(tf)) * idf
        return _l2_normalize(vec)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]


class OpenAIEmbedder:
    """OpenAI 兼容的稠密嵌入后端。

    ``base_url`` 指向任何兼容服务即可（官方、硅基流动、DeepSeek 托管版等）。
    只依赖标准库，避免为了一个 HTTP POST 引入整个 SDK。
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dim: int = 1536,
        timeout: float = 30.0,
    ) -> None:
        self.name = f"openai:{model}"
        self.model = model
        self.dim = dim
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("EMBEDDING_API_KEY", "")
        self.base_url = (
            base_url
            or os.environ.get("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not self.api_key:
            raise RuntimeError(
                "缺少 EMBEDDING_API_KEY；如需零配置运行请改用 --embedder hashing"
            )
        out: List[List[float]] = []
        # 分批，单次请求过大容易被网关截断
        batch = 32
        for i in range(0, len(texts), batch):
            chunk = list(texts[i : i + batch])
            try:
                data = self._post({"model": self.model, "input": chunk})
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
                raise RuntimeError(f"嵌入接口返回 {exc.code}：{detail}") from exc
            items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
            out.extend(_l2_normalize(list(it["embedding"])) for it in items)
        return out


def build_embedder(kind: str = "hashing", dim: int = 1024) -> Embedder:
    """按名字构造嵌入后端。"""
    kind = (kind or "hashing").lower()
    if kind.startswith("hashing"):
        return HashingEmbedder(dim=dim)
    if kind in {"openai", "api", "dense"}:
        return OpenAIEmbedder(dim=dim)
    raise ValueError(f"未知的嵌入后端：{kind}")
