"""Document endpoints: upload PDF, list, get, delete."""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.document import Document
from app.schemas.documents import DocumentDetail, DocumentSummary, UploadResponse
from app.services.ingestion import ingest_pdf

logger = logging.getLogger("learning_companion")
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> UploadResponse:
    """Upload a PDF for ingestion."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted")

    content = await file.read()
    document = await ingest_pdf(db, filename=file.filename, content=content)
    return UploadResponse(
        document_id=document.id,
        filename=document.filename,
        total_chunks=document.total_chunks,
        text_length=document.text_length,
    )


@router.get("", response_model=list[DocumentSummary])
async def list_documents(db: AsyncSession = Depends(get_db)) -> list[DocumentSummary]:
    rows = (
        await db.execute(select(Document).order_by(Document.uploaded_at.desc()))
    ).scalars().all()
    return [DocumentSummary.model_validate(d, from_attributes=True) for d in rows]


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
) -> DocumentDetail:
    doc = (
        await db.execute(select(Document).where(Document.id == document_id))
    ).scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    return DocumentDetail.model_validate(doc, from_attributes=True)


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    doc = (
        await db.execute(select(Document).where(Document.id == document_id))
    ).scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    await db.delete(doc)
