"""核心链路测试。

重点覆盖四类容易出错的地方：
1. 分词与中文二元组（错了后面全错）
2. 分块的行号与标题路径（引用溯源全靠它）
3. 混合检索的精确串召回（BM25 存在的理由）
4. 索引落盘往返（向量存错会让线上和本地不一致）
"""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.chunker import chunk_document  # noqa: E402
from rag.embedder import HashingEmbedder  # noqa: E402
from rag.index import BM25Index, HybridIndex  # noqa: E402
from rag.loader import load_dir, load_file  # noqa: E402
from rag.models import Chunk, Document  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.text import tokenize  # noqa: E402

DOCS = ROOT / "examples" / "docs"


class TokenizeTest(unittest.TestCase):
    def test_cjk_bigram_included(self):
        toks = tokenize("发票要求")
        self.assertIn("发票", toks)
        self.assertIn("票要", toks)

    def test_stopwords_removed(self):
        self.assertNotIn("的", tokenize("差旅费的报销"))
        self.assertIn("差旅", tokenize("差旅费的报销"))

    def test_ascii_and_code_kept(self):
        toks = tokenize("单号 PO-2026-0912 已生成")
        self.assertIn("po-2026-0912", toks)

    def test_empty(self):
        self.assertEqual(tokenize(""), [])


class ChunkerTest(unittest.TestCase):
    def setUp(self):
        self.doc = Document(
            doc_id="d1",
            source="制度.md",
            title="制度",
            text=(
                "# 制度\n\n"
                "开头一段不算重要。\n\n"
                "## 报销\n\n"
                "### 发票要求\n\n"
                "必须是增值税专用发票，抬头为公司全称。\n\n"
                "### 时限\n\n"
                "15 个工作日内提交。\n"
            ),
        )

    def test_heading_path_recorded(self):
        chunks = chunk_document(self.doc)
        headings = [c.heading for c in chunks]
        self.assertTrue(any("报销 › 发票要求" in h for h in headings), headings)

    def test_line_numbers_are_one_based_and_ordered(self):
        chunks = chunk_document(self.doc)
        for c in chunks:
            self.assertGreaterEqual(c.start_line, 1)
            self.assertLessEqual(c.start_line, c.end_line)

    def test_chunk_id_unique(self):
        chunks = chunk_document(self.doc)
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_respects_max_chars(self):
        long_text = "# 标题\n\n" + "\n\n".join("这是一段测试文本，用来撑长度。" * 3 for _ in range(30))
        doc = Document("d2", "长文.md", "长文", long_text)
        chunks = chunk_document(doc, max_chars=200)
        for c in chunks:
            # 超长只可能来自最后合并的尾块，允许 1.6 倍余量
            self.assertLessEqual(len(c.text), 320, c.text[:80])

    def test_invalid_max_chars_rejected(self):
        with self.assertRaises(ValueError):
            chunk_document(self.doc, max_chars=20)

    def test_empty_doc_no_chunks(self):
        self.assertEqual(chunk_document(Document("d3", "e.md", "e", "")), [])


class EmbedderTest(unittest.TestCase):
    def test_deterministic_across_instances(self):
        a = HashingEmbedder(dim=256).embed(["发票要求"])
        b = HashingEmbedder(dim=256).embed(["发票要求"])
        self.assertEqual(a, b)

    def test_l2_normalized(self):
        vec = HashingEmbedder(dim=128).embed(["差旅报销标准"])[0]
        norm = sum(v * v for v in vec) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_similar_text_scores_higher(self):
        emb = HashingEmbedder(dim=512)
        emb.fit(["差旅报销标准", "供应商准入流程", "发票合规要求"])
        vecs = emb.embed(
            ["差旅报销标准", "差旅报销的标准是什么", "供应商淘汰流程"]
        )
        def cos(x, y):
            return sum(a * b for a, b in zip(x, y))
        self.assertGreater(cos(vecs[0], vecs[1]), cos(vecs[0], vecs[2]))


class BM25Test(unittest.TestCase):
    def test_exact_code_ranks_first(self):
        corpus = [
            tokenize("GSK-318 铝合金电芯支架，采购周期 15 天"),
            tokenize("GSK-402 铜排连接片，采购周期 12 天"),
            tokenize("CEL-2210 磷酸铁锂电芯，采购周期 30 天"),
        ]
        idx = BM25Index(corpus)
        res = idx.search(tokenize("GSK-318 采购周期"), top_k=3)
        self.assertEqual(res[0][0], 0)

    def test_no_match_returns_empty(self):
        idx = BM25Index([tokenize("完全无关的内容")])
        self.assertEqual(idx.search(tokenize("量子计算"), top_k=3), [])


class HybridIndexTest(unittest.TestCase):
    def _chunks(self):
        rows = [
            ("A", "差旅 › 住宿", "一线城市住宿每晚上限 600 元。"),
            ("B", "供应商 › 账期", "A 类供应商标准账期为月结 60 天。"),
            ("C", "物料", "GSK-318 采购周期 15 天，安全库存 2000 件。"),
        ]
        return [
            Chunk(f"d-{i}", "d", f"{s}.md", h, t, 1, 2)
            for i, (s, h, t) in enumerate(rows)
        ]

    def test_hybrid_returns_expected_chunk(self):
        idx = HybridIndex(self._chunks(), HashingEmbedder(dim=512)).build()
        hits = idx.search("GSK-318 的采购周期", top_k=2)
        self.assertTrue(hits)
        self.assertIn("GSK-318", hits[0].chunk.text)

    def test_source_filter(self):
        idx = HybridIndex(self._chunks(), HashingEmbedder(dim=256)).build()
        hits = idx.search("采购周期", top_k=3, source_filter="A.md")
        for h in hits:
            self.assertIn("A.md", h.chunk.source)

    def test_empty_index(self):
        idx = HybridIndex([], HashingEmbedder(dim=64)).build()
        self.assertEqual(idx.search("任何问题"), [])


class LoaderTest(unittest.TestCase):
    def test_gbk_file_decoded(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "gbk.txt"
            p.write_bytes("差旅报销标准".encode("gbk"))
            doc = load_file(p, Path(tmp))
            self.assertIn("差旅报销标准", doc.text)

    def test_skip_dirs_and_non_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ok.md").write_text("# 标题\n内容", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "x.md").write_text("不该读到", encoding="utf-8")
            (root / "bin.png").write_bytes(b"\x89PNG")
            docs = load_dir(root)
            self.assertEqual([d.source for d in docs], ["ok.md"])

    def test_example_docs_load(self):
        docs = load_dir(DOCS)
        self.assertGreaterEqual(len(docs), 4)


class PipelineTest(unittest.TestCase):
    def test_ingest_and_query_without_llm(self):
        pipe = RAGPipeline(DOCS, use_llm=False, top_k=3)
        stats = pipe.ingest()
        self.assertGreater(stats.chunks, 5)
        ans = pipe.query("一线城市住宿标准是多少")
        self.assertEqual(ans.mode, "extractive")
        self.assertIn("600", "".join(h.chunk.text for h in ans.hits))

    def test_index_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "idx.json.gz"
            pipe = RAGPipeline(DOCS, use_llm=False)
            pipe.ingest()
            before = pipe.query("供应商账期").hits[0].chunk.chunk_id
            pipe.save(path)

            other = RAGPipeline(DOCS, use_llm=False)
            other.load(path)
            after = other.query("供应商账期").hits[0].chunk.chunk_id
            self.assertEqual(before, after)

    def test_saved_index_is_gzip_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "i.json.gz"
            pipe = RAGPipeline(DOCS, use_llm=False)
            pipe.ingest()
            pipe.save(path)
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(data["version"], 1)
            self.assertEqual(len(data["chunks"]), len(data["vectors"]))

    def test_embedder_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "i.json.gz"
            first = RAGPipeline(DOCS, use_llm=False, dim=128)
            first.ingest()
            first.save(path)
            other = RAGPipeline(DOCS, use_llm=False, dim=256)
            with self.assertRaises(RuntimeError):
                other.load(path)

    def test_no_hits_answer(self):
        pipe = RAGPipeline(DOCS, use_llm=False)
        pipe.ingest()
        ans = pipe.query("量子计算机的超导材料怎么制备")
        self.assertIn("没有检索到", ans.text)


class EvaluationTest(unittest.TestCase):
    def test_example_qrels_runs(self):
        from rag.evaluation import evaluate, load_qrels

        qrels = load_qrels(ROOT / "examples" / "qrels.json")
        self.assertGreater(len(qrels), 5)
        pipe = RAGPipeline(DOCS, use_llm=False, top_k=5)
        pipe.ingest()
        res = evaluate(pipe, qrels, top_k=5)
        self.assertEqual(res.total, len(qrels))
        # 示例语料很小，recall 不应低于 0.75；低于这个值说明检索链路出问题了
        self.assertGreaterEqual(res.recall, 0.75, res.details)


if __name__ == "__main__":
    unittest.main(verbosity=2)
