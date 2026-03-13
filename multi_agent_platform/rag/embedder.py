from __future__ import annotations

from openai import AsyncOpenAI

from multi_agent_platform.models.schemas import Settings
from multi_agent_platform.utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    def __init__(self, settings: Settings) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_embedding_model

    async def embed_text(self, text: str) -> list[float]:
        result = await self._client.embeddings.create(
            input=text, model=self._model
        )
        return result.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        result = await self._client.embeddings.create(
            input=texts, model=self._model
        )
        return [item.embedding for item in result.data]
