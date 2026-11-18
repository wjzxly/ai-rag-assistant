"""核心数据结构。

整个项目只围绕三个对象转：文档（Document）→ 切片（Chunk）→ 命中（Hit）。
把它们定成 frozen dataclass，是为了让"检索结果"这类中间产物不可变，
避免下游环节一边读一边被改，调试时对不上账。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class Document:
    """一份被摄入的原始文档。

    Attributes:
        doc_id: 稳定标识，默认取相对路径的哈希，改名不算改内容。
        source: 相对路径，最终要展示给用户，所以保持人类可读。
        title: 文档标题，取首个一级标题，没有就用文件名。
        text: 全文。
        mtime: 文件最后修改时间，用于增量更新判断。
    """

    doc_id: str
    source: str
    title: str
    text: str
    mtime: float = 0.0

    @property
    def line_count(self) -> int:
        return self.text.count("\n") + 1


@dataclass(frozen=True)
class Chunk:
    """检索的最小单位。

    Attributes:
        chunk_id: ``doc_id`` + 序号，保证全局唯一。
        doc_id: 归属文档。
        source: 冗余存一份路径，展示引用时不必回查文档。
        heading: 标题路径（如「差旅报销 > 发票要求」）。这是召回质量的关键：
            单独一句"必须是增值税专用发票"脱离上下文毫无意义，带上标题路径后
            模型和向量都能理解它在讲什么。
        text: 切片正文。
        start_line / end_line: 在原文中的行号范围，用于引用定位。
    """

    chunk_id: str
    doc_id: str
    source: str
    heading: str
    text: str
    start_line: int
    end_line: int

    def citation(self) -> str:
        """人类可读的引用标签。"""
        loc = f"{self.source}"
        if self.heading:
            loc = f"{loc} › {self.heading}"
        return f"{loc} (L{self.start_line}-{self.end_line})"

    def preview(self, limit: int = 160) -> str:
        """给引用卡片用的纯文本摘要。

        Markdown 表格直接截断会露出 ``|---|---|`` 这种分隔行，观感很差，
        所以这里先把表格拆成"单元格 · 单元格"，再折叠空白。
        """
        parts = []
        for raw in self.text.splitlines():
            s = raw.strip()
            if not s:
                continue
            # 表格分隔行：只由 - | : 和空格组成
            if set(s) <= set("-|: "):
                continue
            if "|" in s:
                cells = [c.strip() for c in s.strip("|").split("|") if c.strip()]
                s = " · ".join(cells)
            parts.append(s)
        out = " ".join(parts)
        out = " ".join(out.split())
        return out if len(out) <= limit else out[:limit].rstrip() + "…"


@dataclass(frozen=True)
class Hit:
    """一条检索命中，保留各路子分数便于解释。

    面试里最常被问的是"为什么这条排第一"，所以分数不能被融合成一个黑盒数字。
    """

    chunk: Chunk
    score: float
    vector_score: float = 0.0
    lexical_score: float = 0.0
    rank: int = 0
    matched_terms: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Answer:
    """一次问答的完整结果。"""

    question: str
    text: str
    hits: Tuple[Hit, ...]
    mode: str  # "llm" 或 "extractive"
    elapsed_ms: float = 0.0
    warning: Optional[str] = None
