"""
Text Chunking Utilities
Split documents into chunks for embedding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

# Optional logger (don't fail if structlog not installed)
try:
    from src.logger import get_logger
    logger = get_logger("chunking")
except ImportError:
    import logging
    logger = logging.getLogger("chunking")


@dataclass
class Chunk:
    """A text chunk with metadata."""
    id: str
    text: str
    metadata: dict
    index: int = 0


class TextSplitter:
    """
    Split text into overlapping chunks.

    Supports multiple strategies:
    - Recursive: Split by hierarchical separators
    - Sentence: Split by sentence boundaries
    - Token: Split by token count (approximate)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
        length_fn: Callable[[str], int] = len,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_fn = length_fn

        # Default hierarchical separators
        self.separators = separators or [
            "\n\n\n",  # Section breaks
            "\n\n",    # Paragraph
            "\n",      # Line
            ". ",      # Sentence
            "; ",      # Clause
            ", ",      # Phrase
            " ",       # Word
            "",        # Character
        ]

    def split_text(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """
        Split text into chunks.

        Args:
            text: Input text
            metadata: Metadata to attach to each chunk

        Returns:
            List of Chunks
        """
        if not text.strip():
            return []

        metadata = metadata or {}
        chunks = []

        # Recursive splitting
        splits = self._split_by_separators(text)

        # Combine small splits into chunks
        current_chunk = ""
        current_size = 0

        for i, split in enumerate(splits):
            split_size = self.length_fn(split)

            # If single split exceeds chunk_size, recursively split
            if split_size > self.chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, i - 1, metadata))
                    current_chunk = ""
                    current_size = 0

                sub_chunks = self._split_long_text(split, metadata)
                chunks.extend(sub_chunks)
                continue

            # Check if adding this split would exceed chunk_size
            if current_size + split_size > self.chunk_size and current_chunk:
                chunks.append(self._create_chunk(current_chunk, i - 1, metadata))
                current_chunk = split
                current_size = split_size
            else:
                current_chunk += split
                current_size += split_size

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(self._create_chunk(current_chunk, len(splits) - 1, metadata))

        # Assign sequential IDs
        for idx, chunk in enumerate(chunks):
            chunk.index = idx

        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks

    def _split_by_separators(self, text: str) -> list[str]:
        """Split text by separators, keeping separators."""
        splits = [text]

        for sep in self.separators:
            if not sep:
                break

            new_splits = []
            for split in splits:
                parts = split.split(sep)
                new_splits.extend([p + sep for p in parts[:-1]])
                if parts[-1]:
                    new_splits.append(parts[-1])

            # If we got more splits, this separator works
            if len(new_splits) > len(splits):
                splits = [s for s in new_splits if s.strip()]

        return [s for s in splits if s.strip()]

    def _split_long_text(self, text: str, metadata: dict) -> list[Chunk]:
        """Recursively split long text."""
        chunks = []
        step = max(1, self.chunk_size // 4)

        for i in range(0, len(text), step):
            chunk_text = text[i:i + self.chunk_size + self.chunk_overlap]
            if chunk_text.strip():
                chunks.append(self._create_chunk(chunk_text, i, metadata))

        return chunks

    def _create_chunk(self, text: str, original_index: int, metadata: dict) -> Chunk:
        """Create a Chunk object."""
        return Chunk(
            id=f"chunk_{original_index}",
            text=text.strip(),
            metadata={
                **metadata,
                "char_count": len(text),
            },
        )


class MarkdownSplitter(TextSplitter):
    """Specialized splitter for Markdown documents."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        # Markdown-aware separators
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n## ",   # H2 headers
                "\n### ",  # H3 headers
                "\n#### ", # H4 headers
                "\n\n",    # Paragraph
                "\n",      # Line
                ". ",      # Sentence
                " ",
                "",
            ],
        )

    def split_text(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Split Markdown preserving headers."""
        metadata = metadata or {}

        # Split by headers
        sections = re.split(r"(?=\n#{1,6}\s)", text)
        chunks = []
        section_index = 0

        for section in sections:
            if not section.strip():
                continue

            # Check if section starts with header
            header_match = re.match(r"(#{1,6}\s[^\n]+\n?)", section)
            if header_match:
                header = header_match.group(1).strip()
                section_metadata = {**metadata, "header": header}
                section = section[header_match.end():]
            else:
                section_metadata = metadata

            # Split section into chunks
            for chunk in super().split_text(section, section_metadata):
                chunk.index = section_index
                chunk.id = f"chunk_{section_index}"
                section_index += 1
                chunks.append(chunk)

        return chunks
