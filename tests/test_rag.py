from __future__ import annotations

from pathlib import Path

import pytest

from multi_agent_platform.rag.chunker import chunk_text
from multi_agent_platform.rag.loader import load_file


class TestLoader:
    def test_load_markdown(self, tmp_path: Path) -> None:
        md = tmp_path / "test.md"
        md.write_text("# Hello\n\nWorld")
        doc = load_file(md)
        assert "Hello" in doc.content
        assert doc.metadata["source"] == "test.md"

    def test_load_txt(self, tmp_path: Path) -> None:
        txt = tmp_path / "test.txt"
        txt.write_text("Some content here")
        doc = load_file(txt)
        assert doc.content == "Some content here"

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "test.csv"
        f.write_text("a,b,c")
        with pytest.raises(ValueError, match="Unsupported"):
            load_file(f)


class TestChunker:
    def test_basic_chunking(self) -> None:
        text = "\n\n".join(f"Paragraph {i} with some content." for i in range(20))
        chunks = chunk_text(text, source="test.md", chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 1
        assert all(c.metadata.source == "test.md" for c in chunks)

    def test_single_paragraph_no_split(self) -> None:
        text = "A short paragraph."
        chunks = chunk_text(text, source="test.md", chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0].content == "A short paragraph."

    def test_chunk_indices_sequential(self) -> None:
        text = "\n\n".join(f"Paragraph {i} " * 20 for i in range(5))
        chunks = chunk_text(text, source="test.md", chunk_size=100, chunk_overlap=10)
        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_text(self) -> None:
        chunks = chunk_text("", source="empty.md")
        assert len(chunks) == 0

    def test_preserves_metadata(self) -> None:
        text = "Some content for testing."
        chunks = chunk_text(text, source="doc.pdf", page_number=5)
        assert chunks[0].metadata.source == "doc.pdf"
        assert chunks[0].metadata.page_number == 5
