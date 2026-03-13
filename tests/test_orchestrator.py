from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from multi_agent_platform.main import app, get_orchestrator, get_settings
from multi_agent_platform.models.schemas import (
    ChatResponse,
    ComplianceResult,
    IngestResponse,
    RAGResult,
    RouterResult,
    Settings,
)
from multi_agent_platform.orchestrator import Orchestrator


@pytest.fixture
def settings() -> Settings:
    return Settings(
        openai_api_key="test-key",
        chroma_persist_dir="./test_chroma_data",
    )


@pytest.fixture
def mock_orchestrator(settings: Settings) -> Orchestrator:
    return Orchestrator(settings)


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health(self) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "healthy"}


class TestChatEndpoint:
    @pytest.mark.asyncio
    async def test_blocked_message(self, settings: Settings) -> None:
        orch = Orchestrator(settings)

        # The keyword filter blocks "elections" before any LLM call
        from multi_agent_platform.models.schemas import ComplianceViolation

        with pytest.raises(ComplianceViolation) as exc_info:
            await orch.chat("Tell me about the upcoming elections")
        assert not exc_info.value.result.compliant
        assert exc_info.value.result.category == "politics"
        assert exc_info.value.result.layer == "keyword"

    @pytest.mark.asyncio
    async def test_blocked_via_api(self, settings: Settings) -> None:
        orch = Orchestrator(settings)

        def override_orchestrator() -> Orchestrator:
            return orch

        app.dependency_overrides[get_orchestrator] = override_orchestrator

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/chat",
                    json={"message": "Tell me about the upcoming elections"},
                )
                assert resp.status_code == 403
                data = resp.json()
                assert data["mode"] == "blocked"
                assert not data["compliance"]["compliant"]
        finally:
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_direct_flow(self, settings: Settings) -> None:
        orch = Orchestrator(settings)

        compliance_ok = ComplianceResult(
            compliant=True, reason="ok", category="none", layer="semantic"
        )
        route_direct = RouterResult(mode="direct", reason="general question")

        with (
            patch.object(
                orch._compliance,
                "check",
                new_callable=AsyncMock,
                return_value=compliance_ok,
            ),
            patch.object(
                orch._router,
                "route",
                new_callable=AsyncMock,
                return_value=route_direct,
            ),
            patch.object(
                orch._direct_agent,
                "answer",
                new_callable=AsyncMock,
                return_value="Python is a programming language.",
            ),
            patch.object(
                orch._synthesizer,
                "synthesize",
                new_callable=AsyncMock,
                return_value="Based on general knowledge:\n\nPython is a programming language.",
            ),
        ):
            result = await orch.chat("What is Python?")
            assert result.mode == "direct"
            assert result.compliance.compliant
            assert "Python" in result.response

    @pytest.mark.asyncio
    async def test_rag_flow(self, settings: Settings) -> None:
        orch = Orchestrator(settings)

        compliance_ok = ComplianceResult(
            compliant=True, reason="ok", category="none", layer="semantic"
        )
        route_rag = RouterResult(mode="rag", reason="company question")
        rag_result = RAGResult(
            response="The onboarding process starts on day one.",
            sources=["onboarding.md"],
            chunks_retrieved=10,
            chunks_used=3,
        )

        with (
            patch.object(
                orch._compliance,
                "check",
                new_callable=AsyncMock,
                return_value=compliance_ok,
            ),
            patch.object(
                orch._router,
                "route",
                new_callable=AsyncMock,
                return_value=route_rag,
            ),
            patch.object(
                orch._rag_agent,
                "answer",
                new_callable=AsyncMock,
                return_value=rag_result,
            ),
            patch.object(
                orch._synthesizer,
                "synthesize",
                new_callable=AsyncMock,
                return_value="Based on retrieved company documents:\n\nThe onboarding process starts on day one.\n\nSources: onboarding.md",
            ),
        ):
            result = await orch.chat("What is the onboarding process?")
            assert result.mode == "rag"
            assert result.sources == ["onboarding.md"]
            assert "onboarding" in result.response.lower()
