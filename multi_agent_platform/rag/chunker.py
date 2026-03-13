from __future__ import annotations

import tiktoken

from multi_agent_platform.models.schemas import Chunk, ChunkMetadata
from multi_agent_platform.utils.logger import get_logger

logger = get_logger(__name__)


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    page_number: int | None = None,
) -> list[Chunk]:
    enc = tiktoken.get_encoding("cl100k_base")

    paragraphs = text.split("\n\n")
    chunks: list[Chunk] = []
    current_tokens: list[int] = []
    current_texts: list[str] = []

    def _flush() -> None:
        if not current_texts:
            return
        content = "\n\n".join(current_texts)
        chunks.append(
            Chunk(
                content=content,
                metadata=ChunkMetadata(
                    source=source,
                    chunk_index=len(chunks),
                    page_number=page_number,
                ),
            )
        )

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_tokens = enc.encode(para)

        if len(current_tokens) + len(para_tokens) > chunk_size:
            _flush()
            # keep overlap tokens from the end of the previous chunk
            if chunk_overlap > 0 and current_tokens:
                overlap_tokens = current_tokens[-chunk_overlap:]
                overlap_text = enc.decode(overlap_tokens)
                current_tokens = list(overlap_tokens)
                current_texts = [overlap_text]
            else:
                current_tokens = []
                current_texts = []

        # handle paragraphs larger than chunk_size
        if len(para_tokens) > chunk_size:
            for i in range(0, len(para_tokens), chunk_size - chunk_overlap):
                segment = para_tokens[i : i + chunk_size]
                segment_text = enc.decode(segment)
                chunks.append(
                    Chunk(
                        content=segment_text,
                        metadata=ChunkMetadata(
                            source=source,
                            chunk_index=len(chunks),
                            page_number=page_number,
                        ),
                    )
                )
            current_tokens = []
            current_texts = []
            continue

        current_tokens.extend(para_tokens)
        current_texts.append(para)

    _flush()

    logger.info(
        "Chunked text",
        extra={"extra": {"source": source, "chunks": len(chunks)}},
    )
    return chunks
