"""命令行入口。

四个子命令正好对应 RAG 项目的四个阶段：
``ingest`` 建库 → ``search`` 看检索 → ``ask`` 出答案 → ``eval`` 量效果。

先能单独调试检索、再看生成，是排查 RAG 问题的正确顺序：
答案胡说时，先确认是不是检索就没给到对的材料。
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import List, Optional

from .evaluation import evaluate, load_qrels
from .pipeline import RAGPipeline, build_pipeline_from_env

DEFAULT_INDEX = ".cache/index.json.gz"


def _ensure_utf8_stdout() -> None:
    """Windows 控制台默认 GBK，中文会乱码。能改就改成 UTF-8，改不了就算了。"""
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if enc == "utf8":
            continue
        buf = getattr(stream, "buffer", None)
        if buf is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, io.UnsupportedOperation, ValueError):
            try:
                wrapped = io.TextIOWrapper(buf, encoding="utf-8", errors="replace")
                if stream is sys.stdout:
                    sys.stdout = wrapped
                else:
                    sys.stderr = wrapped
            except Exception:  # noqa: BLE001 - 输出编码不该让程序崩掉
                pass


def _get_pipeline(args) -> RAGPipeline:
    return build_pipeline_from_env(args.docs)


def _ensure_index(args) -> RAGPipeline:
    """优先读缓存索引，没有就现建。"""
    pipe = _get_pipeline(args)
    index_path = Path(args.index)
    if index_path.exists() and not getattr(args, "rebuild", False):
        try:
            pipe.load(index_path)
            return pipe
        except RuntimeError as exc:
            print(f"[warn] 缓存索引不可用，将重建：{exc}", file=sys.stderr)
    stats = pipe.ingest()
    pipe.save(index_path)
    print(
        f"[ingest] 文档 {stats.docs} 篇 / 片段 {stats.chunks} 个 / "
        f"{stats.chars} 字 / 耗时 {stats.elapsed_ms} ms",
        file=sys.stderr,
    )
    return pipe


def cmd_ingest(args) -> int:
    pipe = _get_pipeline(args)
    stats = pipe.ingest()
    out = pipe.save(args.index)
    print(f"文档 {stats.docs} 篇，片段 {stats.chunks} 个，{stats.chars} 字")
    print(f"耗时 {stats.elapsed_ms} ms，嵌入后端 {pipe.embedder.name}")
    print(f"索引已保存：{out}")
    return 0


def cmd_search(args) -> int:
    pipe = _ensure_index(args)
    hits = pipe.retrieve(args.question, top_k=args.k)
    if not hits:
        print("没有命中任何片段。")
        return 0
    for h in hits:
        print(f"\n[{h.rank}] {h.chunk.citation()}")
        print(f"    融合={h.score:.4f} 向量={h.vector_score:.3f} 词法={h.lexical_score:.3f}")
        if h.matched_terms:
            print(f"    命中词：{' '.join(h.matched_terms)}")
        preview = h.chunk.preview(120)
        print(f"    {preview}")
    return 0


def cmd_ask(args) -> int:
    pipe = _ensure_index(args)
    ans = pipe.query(args.question, top_k=args.k)
    print(ans.text)
    print(f"\n--- 引用 ---")
    for h in ans.hits:
        print(f"[{h.rank}] {h.chunk.citation()}")
    print(f"\n模式：{ans.mode} ｜ 耗时 {ans.elapsed_ms} ms")
    if ans.warning:
        print(f"提示：{ans.warning}")
    if args.json:
        payload = {
            "question": ans.question,
            "answer": ans.text,
            "mode": ans.mode,
            "elapsed_ms": ans.elapsed_ms,
            "citations": [
                {"rank": h.rank, "source": h.chunk.source, "heading": h.chunk.heading,
                 "start_line": h.chunk.start_line, "end_line": h.chunk.end_line,
                 "score": round(h.score, 6)}
                for h in ans.hits
            ],
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"JSON 已写入：{args.json}")
    return 0


def cmd_eval(args) -> int:
    pipe = _ensure_index(args)
    cases = load_qrels(args.qrels)
    res = evaluate(pipe, cases, top_k=args.k)
    print(f"用例 {res.total} 条 ｜ recall@{args.k} = {res.recall:.2%}"
          f" ｜ MRR = {res.mrr:.3f} ｜ 关键词命中 {res.keyword_hit:.2%}")
    if args.verbose:
        for d in res.details:
            flag = "OK " if d["rank"] else "MISS"
            print(f"  {flag} rank={d['rank']} {d['question']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag", description="本地文档 RAG 问答：混合检索 + 引用溯源"
    )
    p.add_argument("--docs", default="examples/docs", help="文档目录")
    p.add_argument("--index", default=DEFAULT_INDEX, help="索引缓存路径")
    p.add_argument("--rebuild", action="store_true", help="忽略缓存，强制重建索引")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("ingest", help="构建索引")
    s1.set_defaults(func=cmd_ingest)

    s2 = sub.add_parser("search", help="只做检索，看命中片段")
    s2.add_argument("question")
    s2.add_argument("-k", type=int, default=5)
    s2.set_defaults(func=cmd_search)

    s3 = sub.add_parser("ask", help="完整问答")
    s3.add_argument("question")
    s3.add_argument("-k", type=int, default=5)
    s3.add_argument("--json", help="把结果写成 JSON 文件")
    s3.set_defaults(func=cmd_ask)

    s4 = sub.add_parser("eval", help="跑评测集")
    s4.add_argument("--qrels", default="examples/qrels.json")
    s4.add_argument("-k", type=int, default=5)
    s4.add_argument("-v", "--verbose", action="store_true")
    s4.set_defaults(func=cmd_eval)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    _ensure_utf8_stdout()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
