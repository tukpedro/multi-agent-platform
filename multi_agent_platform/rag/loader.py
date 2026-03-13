from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from multi_agent_platform.models.schemas import Document
from multi_agent_platform.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_file(path: Path) -> Document:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    if ext == ".pdf":
        reader = PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        content = "\n\n".join(pages)
    else:
        content = path.read_text(encoding="utf-8")

    logger.info(
        "Loaded file",
        extra={"extra": {"file": path.name, "chars": len(content)}},
    )
    return Document(content=content, metadata={"source": path.name})
