"""文档加载。

只做一件事：把目录里的文本文件读成 :class:`Document`。

支持的格式刻意做得窄：``.md`` / ``.txt`` / ``.csv``，PDF 走可选依赖。
理由是这个项目的场景是"企业内部制度与业务文档"，绝大多数是 Word/Excel 导出的
纯文本或 Markdown；为了多支持几种格式引入一堆解析器，收益不抵维护成本。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, List

from .models import Document

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".csv", ".log", ".rst"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}

# Markdown 一级标题：`# 标题`
_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def make_doc_id(source: str) -> str:
    """用路径生成稳定 id。

    不用内容哈希：文档改一个字就换 id 的话，增量索引会退化成全量重建。
    """
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]


def _extract_title(path: Path, text: str) -> str:
    if path.suffix.lower() in {".md", ".markdown"}:
        m = _H1.search(text)
        if m:
            return m.group(1).strip()
    # 退化为文件名（去掉扩展名）
    return path.stem


def _decode(path: Path) -> str:
    """按 UTF-8 → GBK 的顺序尝试解码。

    企业环境里大量文档是从 Windows 导出后又存成 GBK 的，直接 UTF-8 打开会抛异常。
    """
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def load_file(path: Path, root: Path) -> Document:
    """加载单个文件。``root`` 用于计算相对路径。"""
    path = Path(path)
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.as_posix()
    text = _decode(path)
    return Document(
        doc_id=make_doc_id(rel),
        source=rel,
        title=_extract_title(path, text),
        text=text,
        mtime=path.stat().st_mtime,
    )


def iter_files(root: Path) -> Iterable[Path]:
    """递归列出可加载文件，跳过虚拟环境与版本库目录。"""
    root = Path(root)
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in TEXT_SUFFIXES:
            yield p


def load_dir(root: Path) -> List[Document]:
    """加载整个目录。返回按路径排序的列表，保证索引顺序可复现。"""
    root = Path(root)
    docs: List[Document] = []
    for f in iter_files(root):
        try:
            docs.append(load_file(f, root))
        except OSError:
            # 单个文件读失败不应该让整个摄入流程挂掉
            continue
    return docs
