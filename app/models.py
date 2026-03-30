from dataclasses import dataclass
from typing import Optional
from enum import Enum


class SupportedFileType(str, Enum):
    PDF  = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    CSV  = "csv"
    TXT  = "txt"
    MD   = "md"


@dataclass
class ChunkDoc:
    chunk_id:      str
    text:          str
    source:        str
    file_type:     SupportedFileType
    page_num:      Optional[int]
    section_title: Optional[str]
    char_start:    int
    char_end:      int

    def to_metadata_dict(self) -> dict:
        return {
            "chunk_id":      self.chunk_id,
            "source":        self.source,
            "file_type":     self.file_type.value,
            "page_num":      self.page_num if self.page_num is not None else -1,
            "section_title": self.section_title or "",
            "char_start":    self.char_start,
            "char_end":      self.char_end,
        }


@dataclass
class RetrievedChunk:
    chunk: ChunkDoc
    score: float
    rank:  int


@dataclass
class RAGResponse:
    answer:        str
    sources:       list[RetrievedChunk]
    refused:       bool
    refuse_reason: Optional[str] = None
