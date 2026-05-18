from __future__ import annotations

from dataclasses import dataclass

from .loaders import Document

_SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    text: str
    source: str
    metadata: dict


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Рекурсивная сегментация по приоритетному списку разделителей."""
    text = text.strip()
    if not text:
        return []
    chunks = _recursive(text, _SEPARATORS, chunk_size)
    return _apply_overlap(chunks, overlap) if overlap > 0 else chunks


def _recursive(text: str, seps: list[str], size: int) -> list[str]:
    if len(text) <= size:
        return [text]
    if not seps:
        return [text[i : i + size] for i in range(0, len(text), size)]

    sep, *rest = seps
    parts = text.split(sep)
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{sep}{part}" if current else part
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(part) > size:
            chunks.extend(_recursive(part, rest, size))
            current = ""
        else:
            current = part
    if current:
        chunks.append(current)
    return chunks


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    if len(chunks) < 2:
        return chunks
    result = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = prev[-overlap:] if len(prev) > overlap else prev
        result.append(f"{tail} {cur}")
    return result


def chunk_documents(docs: list[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        for i, text in enumerate(split_text(doc.text, chunk_size, overlap)):
            out.append(
                Chunk(
                    text=text,
                    source=doc.source,
                    metadata={**doc.metadata, "chunk_index": i},
                )
            )
    return out
