from __future__ import annotations

import chromadb

from multi_agent_platform.models.schemas import Chunk, SearchResult, Settings
from multi_agent_platform.rag.embedder import Embedder
from multi_agent_platform.utils.logger import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "documents"


class VectorStore:
    def __init__(self, settings: Settings, embedder: Embedder) -> None:
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = embedder
        self._top_k = settings.retrieval_top_k

    async def add_documents(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        embeddings = await self._embedder.embed_batch(texts)
        ids = [
            f"{c.metadata.source}_{c.metadata.chunk_index}" for c in chunks
        ]
        metadatas = [c.metadata.model_dump() for c in chunks]

        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(
            "Added documents to vector store",
            extra={"extra": {"count": len(chunks)}},
        )
        return len(chunks)

    async def search(
        self, query: str, top_k: int | None = None
    ) -> list[SearchResult]:
        k = top_k or self._top_k
        query_embedding = await self._embedder.embed_text(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],  # type: ignore[index]
                results["distances"][0],  # type: ignore[index]
            ):
                search_results.append(
                    SearchResult(
                        content=doc,  # type: ignore[arg-type]
                        metadata=meta,  # type: ignore[arg-type]
                        score=1 - dist,  # cosine distance → similarity
                    )
                )
        return search_results

    def clear(self) -> None:
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Cleared vector store")
