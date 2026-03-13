from __future__ import annotations

import json
from pathlib import Path

from openai import AsyncOpenAI

from multi_agent_platform.models.schemas import RAGResult, SearchResult, Settings
from multi_agent_platform.rag.vector_store import VectorStore
from multi_agent_platform.utils.logger import get_logger

logger = get_logger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "rag_agent.txt"

RERANK_PROMPT = """You are a relevance scorer. Given a user query and a list of text chunks, score each chunk from 0 to 10 based on how relevant it is to answering the query.

User Query: {query}

Chunks:
{chunks}

Respond ONLY with a JSON array of objects, one per chunk, in order:
[{{"index": 0, "score": 7}}, {{"index": 1, "score": 2}}, ...]
"""


class RAGAgent:
    def __init__(
        self, settings: Settings, vector_store: VectorStore
    ) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        self._vector_store = vector_store
        self._top_n = settings.rerank_top_n
        self._prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

    async def _rerank(
        self, query: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        if not results:
            return []

        chunks_text = "\n\n".join(
            f"[Chunk {i}]: {r.content}" for i, r in enumerate(results)
        )
        prompt = RERANK_PROMPT.format(query=query, chunks=chunks_text)

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        raw = response.choices[0].message.content or "[]"
        try:
            scores = json.loads(raw)
            scored = []
            for item in scores:
                idx = item["index"]
                if 0 <= idx < len(results):
                    result = results[idx]
                    result.score = item["score"]
                    scored.append(result)
            scored.sort(key=lambda r: r.score, reverse=True)
            return scored[: self._top_n]
        except (json.JSONDecodeError, KeyError, IndexError):
            logger.warning("Rerank parse failed, using original order")
            return results[: self._top_n]

    async def answer(self, query: str) -> RAGResult:
        # Step 1: retrieve top-k from vector store
        results = await self._vector_store.search(query)
        chunks_retrieved = len(results)

        # Step 2: LLM batch rerank to top-n
        reranked = await self._rerank(query, results)
        chunks_used = len(reranked)

        if not reranked:
            return RAGResult(
                response="I couldn't find relevant information in the company documents to answer your question.",
                sources=[],
                chunks_retrieved=chunks_retrieved,
                chunks_used=0,
            )

        # Step 3: build context and generate answer
        context = "\n\n---\n\n".join(
            f"[Source: {r.metadata.get('source', 'unknown')}]\n{r.content}"
            for r in reranked
        )
        prompt = self._prompt_template.format(
            context=context, question=query
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        answer_text = response.choices[0].message.content or ""
        sources = list({r.metadata.get("source", "") for r in reranked})

        logger.info(
            "RAG answer generated",
            extra={
                "extra": {
                    "chunks_retrieved": chunks_retrieved,
                    "chunks_used": chunks_used,
                    "sources": sources,
                }
            },
        )
        return RAGResult(
            response=answer_text,
            sources=sources,
            chunks_retrieved=chunks_retrieved,
            chunks_used=chunks_used,
        )
