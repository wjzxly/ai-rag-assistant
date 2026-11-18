"""HTTP 层测试。

用 FastAPI 的 TestClient 跑真实路由，覆盖启动加载索引、问答、拒绝空问题。
服务层最容易出的问题是"启得来但一问就 500"，所以这里一定走完整链路。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient

    from rag.app import app

    HAS_FASTAPI = True
except ImportError:  # 没装依赖时跳过，不影响纯引擎测试
    HAS_FASTAPI = False


@unittest.skipUnless(HAS_FASTAPI, "未安装 fastapi")
class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_health(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "ok")
        self.assertGreater(body["chunks"], 0)

    def test_stats_lists_sources(self):
        r = self.client.get("/api/stats")
        self.assertEqual(r.status_code, 200)
        self.assertGreaterEqual(len(r.json()["sources"]), 4)

    def test_ask_returns_citations(self):
        r = self.client.post(
            "/api/ask", json={"question": "一线城市住宿标准是多少", "top_k": 3}
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["answer"])
        self.assertGreaterEqual(len(body["citations"]), 1)
        first = body["citations"][0]
        self.assertIn("source", first)
        self.assertLessEqual(first["start_line"], first["end_line"])

    def test_empty_question_rejected(self):
        r = self.client.post("/api/ask", json={"question": "", "top_k": 3})
        self.assertEqual(r.status_code, 422)

    def test_top_k_out_of_range_rejected(self):
        r = self.client.post(
            "/api/ask", json={"question": "账期", "top_k": 999}
        )
        self.assertEqual(r.status_code, 422)

    def test_off_topic_gets_no_citation(self):
        r = self.client.post(
            "/api/ask",
            json={"question": "量子计算机的超导材料怎么制备", "top_k": 3},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["citations"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
