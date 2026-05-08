from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from datetime import date


class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CAD = "CAD"
    AUD = "AUD"
    JPY = "JPY"
    CHF = "CHF"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Evidence(BaseModel):
    """Canonical evidence record used across all pipeline stages.

    Required fields (source, chunk_id) are always populated:
    - RAG-retrieved evidence: populated by the evidence adapter from DocumentChunk metadata.
    - LLM-extracted evidence: source="llm_extracted", chunk_id="llm_extracted" as sentinels.

    Implementation deviation: `section` is retained as an optional field to carry
    LLM-extracted section identifiers (e.g. "3.1"). It is not part of the finalized
    design but kept for backward compatibility with LLM agents that return section labels.
    """
    text: str
    source: str                              # filename or "llm_extracted"
    chunk_id: str                            # RAG chunk ID or "llm_extracted"
    page: Optional[int] = None
    relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    section: Optional[str] = None           # deprecated: LLM-extracted section label only


class MoneyAmount(BaseModel):
    amount: float
    currency: Currency


class DateRange(BaseModel):
    start: date
    end: Optional[date] = None
