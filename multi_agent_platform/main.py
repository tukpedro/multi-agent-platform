from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse

from multi_agent_platform.models.schemas import (
    ChatRequest,
    ChatResponse,
    ComplianceResult,
    ComplianceViolation,
    IngestResponse,
    IngestionError,
    Settings,
)
from multi_agent_platform.orchestrator import Orchestrator
from multi_agent_platform.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Multi-Agent AI Platform",
    description="Orchestrator-based multi-agent system with compliance checking, intelligent routing, and RAG.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

_settings: Settings | None = None
_orchestrator: Orchestrator | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_orchestrator(settings: Settings = Depends(get_settings)) -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator(settings)
    return _orchestrator


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(ComplianceViolation)
async def compliance_violation_handler(
    request: Request, exc: ComplianceViolation
) -> JSONResponse:
    return JSONResponse(
        status_code=403,
        content=ChatResponse(
            response=f"I'm unable to process this request. Reason: {exc.result.reason}",
            mode="blocked",
            compliance=exc.result,
            sources=None,
            metadata={"blocked": True},
        ).model_dump(),
    )


@app.exception_handler(IngestionError)
async def ingestion_error_handler(
    request: Request, exc: IngestionError
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    logger.info("Chat request", extra={"extra": {"session_id": body.session_id}})
    return await orchestrator.chat(body.message)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    files: list[UploadFile],
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> IngestResponse:
    saved_paths: list[Path] = []

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            for upload in files:
                if not upload.filename:
                    continue
                dest = Path(tmpdir) / upload.filename
                content = await upload.read()
                dest.write_bytes(content)
                saved_paths.append(dest)

            if not saved_paths:
                raise IngestionError("No valid files provided")

            return await orchestrator.ingest(saved_paths)
    except IngestionError:
        raise
    except Exception as exc:
        logger.error("Ingestion failed", extra={"extra": {"error": str(exc)}})
        raise IngestionError(f"Failed to ingest files: {exc}") from exc
