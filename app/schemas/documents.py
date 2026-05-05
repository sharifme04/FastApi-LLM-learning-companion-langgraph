"""Pydantic schemas for documents."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: int
    filename: str
    total_chunks: int
    text_length: int


class DocumentSummary(BaseModel):
    id: int
    filename: str
    title: Optional[str] = None
    total_chunks: int
    file_size_bytes: int
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class DocumentDetail(DocumentSummary):
    text_length: int = 0
    notes: Optional[str] = None
    source_url: Optional[str] = None
