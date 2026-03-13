from __future__ import annotations

import pytest

from multi_agent_platform.models.schemas import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        openai_api_key="test-key",
        chroma_persist_dir="./test_chroma_data",
    )
