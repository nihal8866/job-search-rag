"""Shared settings for job retrieval."""

from dataclasses import dataclass
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SOURCE_ROOT if (SOURCE_ROOT / "pyproject.toml").is_file() else Path.cwd()


@dataclass(frozen=True)
class Settings:
    embedding_model: str = "intfloat/multilingual-e5-small"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cuda"
    collection_name: str = "job_chunks_e5_small_v1"
    candidate_count: int = 20
    rrf_k: int = 60
    top_k: int = 5
    reranker_batch_size: int = 4
    reranker_max_length: int = 512
    chunk_size: int = 384
    chunk_overlap: int = 48
    embedding_batch_size: int = 16
    ingestion_batch_size: int = 500
    chunks_path: Path = PROJECT_ROOT / "data" / "processed" / "job_chunks.jsonl"
    database_path: Path = PROJECT_ROOT / "data" / "indexes" / "chroma"
