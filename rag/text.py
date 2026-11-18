"""分词。

中文没有空格，零依赖条件下没有 jieba 可用。这里的选择是「单字 + 二元组」：

- 单字保证召回率（"发票" 里的 "票" 也能命中 "票据"）
- 二元组保证精度（"发票" 不会误配到 "发货"）

实测在制度类文档上，二元组已经能把 BM25 的排序拉到可接受水平。
如果要上生产，换成 jieba 或公司自带的词典即可，接口不变。
"""

from __future__ import annotations

import re
from typing import List

_CJK = r"\u4e00-\u9fff"
_ASCII_WORD = re.compile(r"[a-z0-9_]+(?:[.\-][a-z0-9_]+)*")
_CJK_RUN = re.compile(f"[{_CJK}]+")

# 问句里高频但无检索意义的词。只做最小集合，不做完整停用词表——
# 停用词表越长越容易误伤业务术语（比如"差额"在财务文档里是关键词）。
STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "或", "有", "没有", "请问", "什么",
    "怎么", "如何", "哪些", "多少", "是否", "可以", "需要", "我们", "你们",
    "吗", "呢", "吧", "啊", "请", "这个", "那个", "一个", "以及", "对于",
    "the", "a", "an", "is", "are", "of", "to", "and", "or", "in", "on",
    "how", "what", "which", "can", "do", "does", "please",
}


def _split_cjk(run: str) -> List[str]:
    chars = list(run)
    grams = list(chars)
    grams.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
    return grams


def tokenize(text: str, keep_stopwords: bool = False) -> List[str]:
    """切词。返回保留重复的词序列（BM25 需要词频）。"""
    if not text:
        return []
    lowered = text.lower()
    tokens: List[str] = []

    # 先按位置把 ASCII 词和中文段各自抽出，避免相互干扰
    for m in _ASCII_WORD.finditer(lowered):
        tokens.append(m.group(0))
    for m in _CJK_RUN.finditer(lowered):
        tokens.extend(_split_cjk(m.group(0)))

    if keep_stopwords:
        return tokens
    return [t for t in tokens if t not in STOPWORDS and len(t) > 0]


def unique_terms(text: str) -> List[str]:
    """去重且保序，用于展示"命中了哪些词"。"""
    seen = set()
    out = []
    for t in tokenize(text):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
