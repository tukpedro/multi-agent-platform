from __future__ import annotations

import json
import re
from pathlib import Path

from openai import AsyncOpenAI

from multi_agent_platform.models.schemas import ComplianceResult, Settings
from multi_agent_platform.utils.logger import get_logger

logger = get_logger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "compliance.txt"

# Bilingual keyword patterns (Portuguese & English)
BLOCKED_PATTERNS: list[re.Pattern[str]] = [
    # Politics
    re.compile(
        r"\b(elei[çc][ãa]o|elei[çc][õo]es|partido pol[íi]tico"
        r"|candidato\s+pol[íi]tico|candidata\s+pol[íi]tica"
        r"|votos?|votar|vota[çc][ãa]o"
        r"|elections?|political\s+party|political\s+candidate"
        r"|democrats?|republicans?"
        r"|ballots?|voting|poll|polls"
        r"|presidentes?|governors?|senators?|congressman|deputados?|vereadores?)\b",
        re.IGNORECASE,
    ),
    # Religion
    re.compile(
        r"\b(religi[ãa]o|igreja|deus|jesus|al[áa]|buda|buddha|bible|b[íi]blia"
        r"|cor[ãa]o|quran|torah|church|mosque|mesquita|temple|templo"
        r"|pray|orar|rezar|scripture|sermon|serm[ãa]o)\b",
        re.IGNORECASE,
    ),
    # Drugs
    re.compile(
        r"\b(drogas?|drugs?|entorpecentes?|narc[óo]ticos?|narcotics?"
        r"|maconha|marijuana|cannabis|coca[íi]na|cocaine|hero[íi]na|heroin"
        r"|metanfetamina|methamphetamine|crack|ecstasy|lsd|op[ió]ide|opioid"
        r"|overdose|traficantes?|drug\s+dealer|tr[áa]fico\s+de\s+drogas"
        r"|drug\s+trafficking|uso\s+de\s+drogas|drug\s+use)\b",
        re.IGNORECASE,
    ),
    # Illegal activities
    re.compile(
        r"\b(hackear|hack into|phishing|malware|ransomware"
        r"|launder|lavagem de dinheiro|money laundering"
        r"|contrabando|smuggling|fraud|fraude|counterfeit|falsificar)\b",
        re.IGNORECASE,
    ),
    # Violence
    re.compile(
        r"\b(matar|kill someone|assassinar|assassinate|bomb|bomba|terroris"
        r"|arma de fogo|firearm|weapon|explosiv|massacre|torture|tortura)\b",
        re.IGNORECASE,
    ),
    # Layoffs / dismissals
    re.compile(
        r"\b(layoffs?|demiss[ãa]o|demiss[õo]es|demitir|demitidos?"
        r"|fired|firing|downsizing|reestrutura[çc][ãa]o"
        r"|restructuring|redundancy|redundancies|cortes?\s+de\s+pessoal"
        r"|workforce\s+reduction|mass\s+layoff|desligamento)\b",
        re.IGNORECASE,
    ),
    # Jailbreak attempts
    re.compile(
        r"(ignore\s+(your|all|previous|the)\s+(instructions|rules|constraints|prompt)"
        r"|pretend\s+you\s+(are|have|can)|act\s+as\s+if"
        r"|ignore\s+.*?(restr|limit|guideline|filter)"
        r"|bypass\s+.*?(filter|safe|guard|compliance)"
        r"|DAN\s+mode|do\s+anything\s+now)",
        re.IGNORECASE,
    ),
]

PATTERN_CATEGORIES = [
    "politics",
    "religion",
    "drugs",
    "illegal",
    "violence",
    "layoffs",
    "jailbreak",
]


class ComplianceGuard:
    def __init__(self, settings: Settings) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        self._system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    def _check_keywords(self, message: str) -> ComplianceResult | None:
        for pattern, category in zip(BLOCKED_PATTERNS, PATTERN_CATEGORIES):
            if pattern.search(message):
                logger.info(
                    "Keyword compliance block",
                    extra={"extra": {"category": category}},
                )
                return ComplianceResult(
                    compliant=False,
                    reason=f"Message blocked by keyword filter: {category}",
                    category=category,
                    layer="keyword",
                )
        return None

    async def _check_semantic(self, message: str) -> ComplianceResult:
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
            result = ComplianceResult(
                compliant=data.get("compliant", True),
                reason=data.get("reason", ""),
                category=data.get("category", "none"),
                layer="semantic",
            )
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Failed to parse compliance LLM response, allowing message",
                extra={"extra": {"raw": raw}},
            )
            result = ComplianceResult(
                compliant=True,
                reason="Compliance check parse error — defaulting to allow",
                category="none",
                layer="semantic",
            )
        return result

    async def check(self, message: str) -> ComplianceResult:
        # Layer 1: fast keyword check
        keyword_result = self._check_keywords(message)
        if keyword_result is not None:
            return keyword_result

        # Layer 2: semantic LLM check
        return await self._check_semantic(message)
