"""答案生成。

两条路径，必须在没有 API key 的环境里也能给出可用结果：

1. **LLM 路径**：走 OpenAI 兼容的 ``/v1/chat/completions``。提示词强制要求
   只依据检索到的资料作答，并用 ``[1]`` ``[2]`` 标注来源；资料不足时明确说"资料未提及"。
   这是抑制幻觉最有效的一招——比事后校验便宜得多。

2. **抽取式兜底**：没有 key 时，从命中片段里挑出与问题词重合度最高的句子，
   拼成带引用的答案。它不是"生成"，但足以验证检索链路是否正常，
   也让项目在别人 clone 下来时不至于第一步就卡住。
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import List, Optional, Sequence

from .models import Hit
from .text import tokenize

SYSTEM_PROMPT = """你是一个企业内部知识库问答助手。规则：

1. 只依据【资料】回答，不要使用资料之外的知识，也不要推测。
2. 每个结论后面必须标注来源编号，例如 [1]、[2]，编号对应资料前的序号。
3. 如果资料不足以回答，直接说"现有资料未提及"，并说明缺什么信息，不要编造。
4. 涉及金额、时限、审批层级等硬性条件时，原文照抄关键数字，不要改写。
5. 用简体中文回答，控制在 200 字以内，先给结论再给依据。
"""

_SENT_SPLIT = re.compile(r"(?<=[。！？；\n])")


class LLMGenerator:
    """OpenAI 兼容的对话接口调用。"""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        temperature: float = 0.2,
    ) -> None:
        self.model = model or os.environ.get("LLM_MODEL", "deepseek-chat")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.base_url = (
            base_url or os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
        ).rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: Sequence[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "stream": False,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"生成接口返回 {exc.code}：{detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"生成接口不可达：{exc.reason}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("生成接口返回空结果")
        return (choices[0].get("message") or {}).get("content", "").strip()


def build_context(hits: Sequence[Hit]) -> str:
    """把命中片段拼成带编号的资料块。"""
    blocks: List[str] = []
    for i, h in enumerate(hits, start=1):
        blocks.append(f"【{i}】来源：{h.chunk.citation()}\n{h.chunk.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: Sequence[Hit]) -> List[dict]:
    user = f"【资料】\n{build_context(hits)}\n\n【问题】\n{question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _pick_sentences(text: str, q_terms: set, limit: int = 3) -> List[str]:
    """挑出与问题词重合最多的句子，保持原顺序。"""
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    scored = []
    for i, s in enumerate(sents):
        overlap = len(q_terms & set(tokenize(s)))
        if overlap:
            scored.append((overlap, i, s))
    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen = sorted(scored[:limit], key=lambda x: x[1])
    return [s for _, _, s in chosen]


def extractive_answer(question: str, hits: Sequence[Hit], max_sources: int = 3) -> str:
    """无 LLM 时的抽取式答案。"""
    if not hits:
        return "知识库中没有检索到相关内容。请换一种问法，或先确认文档已摄入。"
    q_terms = set(tokenize(question))
    parts: List[str] = []
    for i, h in enumerate(hits[:max_sources], start=1):
        sents = _pick_sentences(h.chunk.text, q_terms)
        body = "".join(sents) if sents else h.chunk.text[:160].replace("\n", " ")
        parts.append(f"{body} [{i}]")
    answer = "\n".join(parts)
    return (
        answer
        + "\n\n（未配置 LLM_API_KEY，以上为检索原文摘录，非模型生成。）"
    )
