"""评测：用固定问题集量化检索质量。

没有评测集的 RAG 项目只能靠"感觉准"，这在面试里是站不住的。
这里约定的 qrels 格式是 JSON 列表，每条：

```json
{"question": "差旅住宿费上限是多少", "sources": ["差旅报销制度.md"], "keywords": ["500"]}
```

- ``sources`` 是**期望命中的文档**（文件名即可），算 recall@k 与 MRR；
- ``keywords`` 是题面之外必须被检索到的关键事实，用于检查"命中了但内容不对"。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from .pipeline import RAGPipeline


@dataclass
class EvalCase:
    question: str
    sources: List[str]
    keywords: List[str]


@dataclass
class EvalResult:
    total: int
    recall: float
    mrr: float
    keyword_hit: float
    details: List[dict]


def load_qrels(path: str | Path) -> List[EvalCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases: List[EvalCase] = []
    for item in data:
        cases.append(
            EvalCase(
                question=item["question"],
                sources=list(item.get("sources") or []),
                keywords=list(item.get("keywords") or []),
            )
        )
    return cases


def _first_relevant_rank(hits: Sequence, sources: Sequence[str]) -> int:
    """返回第一条命中期望来源的名次，没有则返回 0。"""
    for h in hits:
        if any(s in h.chunk.source for s in sources):
            return h.rank
    return 0


def evaluate(
    pipeline: RAGPipeline,
    cases: Sequence[EvalCase],
    top_k: int = 5,
) -> EvalResult:
    if not cases:
        return EvalResult(0, 0.0, 0.0, 0.0, [])

    recalls = 0
    rr_sum = 0.0
    kw_hits = 0
    details: List[dict] = []

    for case in cases:
        hits = pipeline.retrieve(case.question, top_k=top_k)
        rank = _first_relevant_rank(hits, case.sources)
        if rank:
            recalls += 1
            rr_sum += 1.0 / rank

        joined = "\n".join(h.chunk.text for h in hits)
        kw_ok = all(k in joined for k in case.keywords) if case.keywords else True
        if kw_ok:
            kw_hits += 1

        details.append(
            {
                "question": case.question,
                "rank": rank,
                "keywords_ok": kw_ok,
                "top_source": hits[0].chunk.citation() if hits else None,
            }
        )

    n = len(cases)
    return EvalResult(
        total=n,
        recall=round(recalls / n, 4),
        mrr=round(rr_sum / n, 4),
        keyword_hit=round(kw_hits / n, 4),
        details=details,
    )
