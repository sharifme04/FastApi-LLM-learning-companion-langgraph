"""Token-aware semantic chunking with overlap.

We don't have a real tokenizer here (would be model-specific). Instead we
use a stable approximation: ~4 chars per token for English prose. Sentences
are kept intact; we accumulate sentences until the running token estimate
hits CHUNK_SIZE_TOKENS, then start a new chunk that overlaps the last
~CHUNK_OVERLAP_TOKENS worth of sentences.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.config import get_settings

logger = logging.getLogger("learning_companion")
settings = get_settings()

CHARS_PER_TOKEN = 4


@dataclass(slots=True)
class Chunk:
    text: str
    token_count: int
    page_start: int
    page_end: int
    chunk_index: int


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT.split(text)
    return [p for p in (s.strip() for s in parts) if p]


def _estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[Chunk]:
    """Chunk a list of (page_no, text) into overlapping semantic chunks.

    Args:
        pages: Output of pdf_parser.extract_pdf_pages.
        chunk_size_tokens: target chunk size; default from settings.
        overlap_tokens: overlap size; default from settings.

    Returns:
        List of Chunk objects with chunk_index assigned.
    """
    target = chunk_size_tokens or settings.chunk_size_tokens
    overlap = overlap_tokens or settings.chunk_overlap_tokens

    # Build a flat list of (page, sentence) pairs
    items: list[tuple[int, str]] = []
    for page_no, page_text in pages:
        for sent in _split_sentences(page_text):
            items.append((page_no, sent))

    chunks: list[Chunk] = []
    cur_sents: list[tuple[int, str]] = []
    cur_tokens = 0
    chunk_index = 0

    def _emit() -> None:
        nonlocal chunk_index
        if not cur_sents:
            return
        text = " ".join(s for _, s in cur_sents)
        page_start = cur_sents[0][0]
        page_end = cur_sents[-1][0]
        chunks.append(
            Chunk(
                text=text,
                token_count=_estimate_tokens(text),
                page_start=page_start,
                page_end=page_end,
                chunk_index=chunk_index,
            )
        )
        chunk_index += 1

    for page_no, sent in items:
        sent_tokens = _estimate_tokens(sent)
        if cur_tokens + sent_tokens > target and cur_sents:
            _emit()

            # Build overlap tail
            tail: list[tuple[int, str]] = []
            tail_tokens = 0
            for p, s in reversed(cur_sents):
                t = _estimate_tokens(s)
                if tail_tokens + t > overlap:
                    break
                tail.insert(0, (p, s))
                tail_tokens += t
            cur_sents = tail
            cur_tokens = tail_tokens

        cur_sents.append((page_no, sent))
        cur_tokens += sent_tokens

    _emit()

    logger.info(
        "Chunked document",
        extra={"chunks": len(chunks), "target": target, "overlap": overlap},
    )
    return chunks
