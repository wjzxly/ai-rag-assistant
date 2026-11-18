"""分块策略。

这是 RAG 里最容易被低估的一环。分块不好，后面 embedding 再贵、模型再强也救不回来。

本模块的两条设计：

1. **按标题切分，并把标题路径带进 chunk**。
   「必须是增值税专用发票」这句话单看是死的，但放进「差旅报销 > 发票要求」里，
   向量和模型都能正确理解它。所以每个 chunk 都冗余一份 heading 路径。

2. **段落优先，不切断句子**。
   固定长度切窗会把一句话拦腰截断，检索到的片段语义残缺。
   这里以空行分段为最小单位，超长段落才退化为按句号切，再退化为硬切。

重叠（overlap）只保留上一块的最后一个段落，而不是按字符滑窗：
字符滑窗会产出大量几乎重复的片段，把 top-k 名额挤掉。
"""

from __future__ import annotations

import csv
import io
import re
from typing import List, Sequence, Tuple

from .models import Chunk, Document

DEFAULT_MAX_CHARS = 600
DEFAULT_MIN_CHARS = 100
DEFAULT_OVERLAP_PARAGRAPHS = 1

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCE_END = re.compile(r"(?<=[。！？；\.\!\?])\s*")


class _Paragraph:
    """带行号的段落，用于回填引用位置。"""

    __slots__ = ("text", "start_line", "end_line")

    def __init__(self, text: str, start_line: int, end_line: int) -> None:
        self.text = text
        self.start_line = start_line
        self.end_line = end_line

    def __len__(self) -> int:
        return len(self.text)


def _split_paragraphs(lines: Sequence[Tuple[int, str]]) -> List[_Paragraph]:
    """把 (行号, 文本) 序列按空行聚成段落。"""
    out: List[_Paragraph] = []
    buf: List[str] = []
    start = end = 0
    for lineno, line in lines:
        if line.strip() == "":
            if buf:
                out.append(_Paragraph("\n".join(buf).strip(), start, end))
                buf = []
            continue
        if not buf:
            start = lineno
        buf.append(line)
        end = lineno
    if buf:
        out.append(_Paragraph("\n".join(buf).strip(), start, end))
    return [p for p in out if p.text]


def _hard_split(para: _Paragraph, max_chars: int) -> List[_Paragraph]:
    """把超长段落切开：先按句末标点，再退化按字符硬切。"""
    text = para.text
    if len(text) <= max_chars:
        return [para]

    pieces: List[str] = []
    for sent in _SENTENCE_END.split(text):
        if not sent:
            continue
        if len(sent) <= max_chars:
            pieces.append(sent)
        else:
            for i in range(0, len(sent), max_chars):
                pieces.append(sent[i : i + max_chars])

    # 把碎片重新拼到 max_chars 以内
    merged: List[str] = []
    cur = ""
    for piece in pieces:
        if len(cur) + len(piece) <= max_chars:
            cur += piece
        else:
            if cur:
                merged.append(cur)
            cur = piece
    if cur:
        merged.append(cur)

    # 行号无法精确对应到句子级，按字符占比线性估算，只用于展示
    span = max(1, para.end_line - para.start_line + 1)
    total = max(1, len(text))
    out: List[_Paragraph] = []
    cursor = 0
    for m in merged:
        offset = cursor
        cursor += len(m)
        s = para.start_line + int(span * offset / total)
        e = para.start_line + int(span * cursor / total)
        out.append(_Paragraph(m, s, max(s, e)))
    return out


def _pack(paragraphs: List[_Paragraph], max_chars: int) -> List[List[_Paragraph]]:
    """把段落装箱成若干块，尽量填满但不超长。"""
    groups: List[List[_Paragraph]] = []
    cur: List[_Paragraph] = []
    cur_len = 0
    for p in paragraphs:
        add = len(p) + (1 if cur else 0)
        if cur and cur_len + add > max_chars:
            groups.append(cur)
            cur = [p]
            cur_len = len(p)
        else:
            cur.append(p)
            cur_len += add
    if cur:
        groups.append(cur)
    return groups


def _section_chunks(
    doc: Document,
    heading: str,
    lines: List[Tuple[int, str]],
    max_chars: int,
    overlap: int,
    counter: List[int],
) -> List[Chunk]:
    paragraphs = _split_paragraphs(lines)
    if not paragraphs:
        return []

    expanded: List[_Paragraph] = []
    for p in paragraphs:
        expanded.extend(_hard_split(p, max_chars))

    groups = _pack(expanded, max_chars)
    chunks: List[Chunk] = []
    prev_tail: List[_Paragraph] = []

    for gi, group in enumerate(groups):
        body = prev_tail + group
        text = "\n\n".join(p.text for p in body).strip()
        # 太短的尾块并到上一块，避免「半句话」成为独立片段
        if (
            len(text) < DEFAULT_MIN_CHARS
            and chunks
            and gi == len(groups) - 1
        ):
            last = chunks.pop()
            merged_text = f"{last.text}\n\n{text}".strip()
            chunks.append(
                Chunk(
                    chunk_id=last.chunk_id,
                    doc_id=last.doc_id,
                    source=last.source,
                    heading=last.heading,
                    text=merged_text,
                    start_line=last.start_line,
                    end_line=body[-1].end_line,
                )
            )
            prev_tail = body[-overlap:] if overlap else []
            continue

        counter[0] += 1
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}-{counter[0]:03d}",
                doc_id=doc.doc_id,
                source=doc.source,
                heading=heading,
                text=text,
                start_line=body[0].start_line,
                end_line=body[-1].end_line,
            )
        )
        prev_tail = body[-overlap:] if overlap else []

    return chunks


def _csv_chunks(doc: Document, max_chars: int, counter: List[int]) -> List[Chunk]:
    """表格按行切分。

    整张表当成一个片段是错的：一行物料和一行完全无关的物料被塞进同一个向量，
    检索时"GSK-318 的采购周期"只能靠整表平均向量去碰运气，命中率和引用精度都会掉。

    这里把每行转成「列名：值」的形式，而不是保留原始逗号行。
    原因是纯逗号串里 ``15`` 这个数字没有语义，模型和检索都看不出它是"采购周期天数"；
    带上列名之后，既好检索也好读。引用仍然指回原始行号。
    """
    try:
        rows = [
            r for r in csv.reader(io.StringIO(doc.text)) if any(c.strip() for c in r)
        ]
    except csv.Error:
        return []
    if len(rows) < 2:
        return []

    header = [h.strip() for h in rows[0]]
    chunks: List[Chunk] = []
    for line_no, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            continue
        pairs = [
            f"{h}：{v.strip()}" for h, v in zip(header, row) if h and v.strip()
        ]
        if not pairs:
            continue
        text = "；".join(pairs)
        if len(text) > max_chars:
            text = text[:max_chars]
        counter[0] += 1
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}-{counter[0]:03d}",
                doc_id=doc.doc_id,
                source=doc.source,
                heading=doc.title,
                text=text,
                start_line=line_no,
                end_line=line_no,
            )
        )
    return chunks


def chunk_document(
    doc: Document,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP_PARAGRAPHS,
) -> List[Chunk]:
    """把一份文档切成引用可定位的片段。"""
    if max_chars < 80:
        raise ValueError("max_chars 过小，会把句子切碎；建议不小于 80")

    counter = [0]
    if doc.source.lower().endswith(".csv"):
        return _csv_chunks(doc, max_chars, counter)

    lines = [(i + 1, ln) for i, ln in enumerate(doc.text.splitlines())]
    chunks: List[Chunk] = []

    stack: List[Tuple[int, str]] = []  # (level, title)
    section_headings: List[Tuple[int, str]] = []
    section_lines: List[Tuple[int, str]] = []

    def flush() -> None:
        if not section_lines:
            return
        # 一级标题就是文档标题，citation 里已经有文件名，重复一遍只会让引用变长
        sub = [t for lvl, t in section_headings if lvl > 1]
        path = " › ".join(sub) or doc.title
        chunks.extend(
            _section_chunks(doc, path, section_lines, max_chars, overlap, counter)
        )

    for lineno, line in lines:
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).strip()))
            section_headings = list(stack)
            section_lines = []
            continue
        section_lines.append((lineno, line))

    flush()

    if not chunks and doc.text.strip():
        # 没有任何标题的纯文本：整篇当一节
        chunks.extend(
            _section_chunks(doc, doc.title, lines, max_chars, overlap, counter)
        )
    return chunks


def chunk_documents(
    docs: Sequence[Document],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP_PARAGRAPHS,
) -> List[Chunk]:
    out: List[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, max_chars=max_chars, overlap=overlap))
    return out
