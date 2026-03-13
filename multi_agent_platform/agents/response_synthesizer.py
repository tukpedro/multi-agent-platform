from __future__ import annotations

from pathlib import Path

from openai import AsyncOpenAI

from multi_agent_platform.models.schemas import Settings
from multi_agent_platform.utils.logger import get_logger

logger = get_logger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "synthesizer.txt"


class ResponseSynthesizer:
    def __init__(self, settings: Settings) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        self._prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

    async def synthesize(
        self,
        agent_response: str,
        mode: str,
        sources: list[str] | None = None,
    ) -> str:
        prompt = self._prompt_template.format(
            mode=mode,
            sources=", ".join(sources) if sources else "N/A",
            agent_response=agent_response,
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        result = response.choices[0].message.content or agent_response
        logger.info("Response synthesized", extra={"extra": {"mode": mode}})
        return result
