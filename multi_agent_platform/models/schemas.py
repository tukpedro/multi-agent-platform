from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    chroma_persist_dir: str = "./chroma_data"
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    retrieval_top_k: int = 10
    rerank_top_n: int = 3
    log_level: str = "INFO"


# ---------------------------------------------------------------------------
# Shared value objects
# ---------------------------------------------------------------------------

class ChunkMetadata(BaseModel):
    source: str
    chunk_index: int
    page_number: int | None = None


class Chunk(BaseModel):
    content: str
    metadata: ChunkMetadata


class Document(BaseModel):
    content: str
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent results
# ---------------------------------------------------------------------------

class ComplianceResult(BaseModel):
    compliant: bool
    reason: str
    category: str = ""
    layer: str = ""


class RouterResult(BaseModel):
    mode: Literal["rag", "direct"]
    reason: str


class SearchResult(BaseModel):
    content: str
    metadata: dict
    score: float


class RAGResult(BaseModel):
    response: str
    sources: list[str]
    chunks_retrieved: int
    chunks_used: int


# ---------------------------------------------------------------------------
# API request / response
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    mode: str
    compliance: ComplianceResult
    sources: list[str] | None = None
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    ingested: int
    chunks_created: int
    files: list[str]


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ComplianceViolation(Exception):
    def __init__(self, result: ComplianceResult) -> None:
        self.result = result
        super().__init__(result.reason)


class IngestionError(Exception):
    pass
