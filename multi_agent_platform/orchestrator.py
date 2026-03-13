from __future__ import annotations

import time
from pathlib import Path

from multi_agent_platform.agents.compliance_guard import ComplianceGuard
from multi_agent_platform.agents.direct_agent import DirectAgent
from multi_agent_platform.agents.rag_agent import RAGAgent
from multi_agent_platform.agents.response_synthesizer import ResponseSynthesizer
from multi_agent_platform.agents.router import Router
from multi_agent_platform.models.schemas import (
    ChatResponse,
    Chunk,
    ComplianceResult,
    ComplianceViolation,
    IngestResponse,
    Settings,
)
from multi_agent_platform.rag.chunker import chunk_text
from multi_agent_platform.rag.embedder import Embedder
from multi_agent_platform.rag.loader import load_file
from multi_agent_platform.rag.vector_store import VectorStore
from multi_agent_platform.utils.logger import get_logger

logger = get_logger(__name__)


class Orchestrator:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        embedder = Embedder(settings)
        self._vector_store = VectorStore(settings, embedder)

        self._compliance = ComplianceGuard(settings)
        self._router = Router(settings)
        self._rag_agent = RAGAgent(settings, self._vector_store)
        self._direct_agent = DirectAgent(settings)
        self._synthesizer = ResponseSynthesizer(settings)

    async def chat(self, message: str) -> ChatResponse:
        t0 = time.perf_counter()

        # Step 1: compliance check
        t_comp = time.perf_counter()
        compliance = await self._compliance.check(message)
        dt_comp = time.perf_counter() - t_comp
        logger.info(
            "Compliance check done",
            extra={"extra": {"compliant": compliance.compliant, "ms": round(dt_comp * 1000)}},
        )

        if not compliance.compliant:
            raise ComplianceViolation(compliance)

        # Step 2: route
        t_route = time.perf_counter()
        route = await self._router.route(message)
        dt_route = time.perf_counter() - t_route
        logger.info(
            "Routing done",
            extra={"extra": {"mode": route.mode, "ms": round(dt_route * 1000)}},
        )

        # Step 3: call appropriate agent
        sources: list[str] | None = None
        metadata: dict = {"mode": route.mode, "route_reason": route.reason}

        t_agent = time.perf_counter()
        if route.mode == "rag":
            rag_result = await self._rag_agent.answer(message)
            raw_response = rag_result.response
            sources = rag_result.sources
            metadata["chunks_retrieved"] = rag_result.chunks_retrieved
            metadata["chunks_used"] = rag_result.chunks_used
        else:
            raw_response = await self._direct_agent.answer(message)
        dt_agent = time.perf_counter() - t_agent
        logger.info(
            "Agent response done",
            extra={"extra": {"mode": route.mode, "ms": round(dt_agent * 1000)}},
        )

        # Step 4: synthesize final response
        t_synth = time.perf_counter()
        final_response = await self._synthesizer.synthesize(
            raw_response, route.mode, sources
        )
        dt_synth = time.perf_counter() - t_synth

        total_ms = round((time.perf_counter() - t0) * 1000)
        metadata["latency_ms"] = total_ms
        logger.info(
            "Pipeline complete",
            extra={"extra": {"total_ms": total_ms}},
        )

        return ChatResponse(
            response=final_response,
            mode=route.mode,
            compliance=compliance,
            sources=sources,
            metadata=metadata,
        )

    async def ingest(self, file_paths: list[Path]) -> IngestResponse:
        all_chunks: list[Chunk] = []
        ingested_files: list[str] = []

        for path in file_paths:
            doc = load_file(path)
            chunks = chunk_text(
                doc.content,
                source=doc.metadata.get("source", path.name),
                chunk_size=self._settings.chunk_size_tokens,
                chunk_overlap=self._settings.chunk_overlap_tokens,
            )
            all_chunks.extend(chunks)
            ingested_files.append(path.name)

        added = await self._vector_store.add_documents(all_chunks)

        logger.info(
            "Ingestion complete",
            extra={
                "extra": {
                    "files": len(ingested_files),
                    "chunks": added,
                }
            },
        )
        return IngestResponse(
            ingested=len(ingested_files),
            chunks_created=added,
            files=ingested_files,
        )
