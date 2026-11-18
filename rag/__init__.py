"""ai-rag-assistant 的核心包。

对外只需要三个名字：流水线、索引、答案结构。
"""

from .models import Answer, Chunk, Document, Hit
from .pipeline import IngestStats, RAGPipeline, build_pipeline_from_env

__all__ = [
    "Answer",
    "Chunk",
    "Document",
    "Hit",
    "IngestStats",
    "RAGPipeline",
    "build_pipeline_from_env",
]

__version__ = "0.1.0"
