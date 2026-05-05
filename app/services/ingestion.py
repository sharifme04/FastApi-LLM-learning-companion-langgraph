"""Document ingestion pipeline: PDF → pages → chunks → embeddings → DB."""

from __future__ import annotations

import logging
import os
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.chunk import Chunk
from app.models.document import Document
from app.services.chunker import chunk_pages
from app.services.embedder import embed_texts
from app.services.pdf_parser import extract_pdf_pages
from app.utils.exceptions import IngestionError

logger = logging.getLogger("learning_companion")
settings = get_settings()


async def ingest_pdf(
    db: AsyncSession,
    filename: str,
    content: bytes,
    save_to_disk: bool = True,
) -> Document:
    """Parse, chunk, embed, and persist a PDF.

    Args:
        db: async DB session (caller commits).
        filename: original filename for display.
        content: raw PDF bytes.
        save_to_disk: if True, write the file to settings.upload_dir for retrieval.

    Returns:
        Persisted Document row with .total_chunks set.
    """
    if len(content) > settings.max_pdf_bytes:
        raise IngestionError(
            f"PDF too large: {len(content)} > {settings.max_pdf_bytes} bytes",
            detail={"size": len(content), "limit": settings.max_pdf_bytes},
        )

    pages = extract_pdf_pages(content)
    chunks = chunk_pages(pages)
    if not chunks:
        raise IngestionError("PDF parsed but no chunks were produced.")

    file_path: str | None = None
    if save_to_disk:
        try:
            os.makedirs(settings.upload_dir, exist_ok=True)
            unique = f"{uuid.uuid4().hex}_{filename}"
            file_path = os.path.join(settings.upload_dir, unique)
            with open(file_path, "wb") as f:
                f.write(content)
        except OSError as e:
            logger.warning("Failed to persist file to disk: %s", e)
            file_path = None

    text_length = sum(len(t) for _, t in pages)

    document = Document(
        filename=filename,
        title=filename,
        file_path=file_path,
        file_size_bytes=len(content),
        total_chunks=len(chunks),
        text_length=text_length,
    )
    db.add(document)
    await db.flush()  # need document.id

    embeddings = embed_texts([c.text for c in chunks])

    for c, vec in zip(chunks, embeddings):
        db.add(
            Chunk(
                document_id=document.id,
                chunk_index=c.chunk_index,
                text=c.text,
                embedding=vec,
                token_count=c.token_count,
                page_start=c.page_start,
                page_end=c.page_end,
            )
        )
    await db.flush()

    logger.info(
        "Document ingested",
        extra={
            "document_id": document.id,
            "chunks": len(chunks),
            "pages": len(pages),
            "size_bytes": len(content),
        },
    )
    return document
