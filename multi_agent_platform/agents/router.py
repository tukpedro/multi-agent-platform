from __future__ import annotations

import json
from pathlib import Path

from openai import AsyncOpenAI

from multi_agent_platform.models.schemas import RouterResult, Settings
from multi_agent_platform.utils.logger import get_logger

logger = get_logger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "router.txt"


class Router:
    def __init__(self, settings: Settings) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        self._system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    async def route(self, message: str) -> RouterResult:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": message},
            ],
            temperature=0,
        )

        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
            result = RouterResult(
                mode=data.get("mode", "direct"),
                reason=data.get("reason", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning(
                "Failed to parse router response, defaulting to direct",
                extra={"extra": {"raw": raw}},
            )
            result = RouterResult(
                mode="direct",
                reason="Router parse error — defaulting to direct",
            )

        logger.info(
            "Routed query",
            extra={"extra": {"mode": result.mode, "reason": result.reason}},
        )
        return result
