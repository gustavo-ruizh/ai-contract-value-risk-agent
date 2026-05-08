from enum import Enum
from typing import Optional, List
from datetime import date

from pydantic import BaseModel

from app.schemas.common import ConfidenceLevel, Evidence, DateRange


class ContractType(str, Enum):
    SERVICE_AGREEMENT = "service_agreement"
    PURCHASE_ORDER = "purchase_order"
    LEASE = "lease"
    LICENSE = "license"
    EMPLOYMENT = "employment"
    NDA = "nda"
    PARTNERSHIP = "partnership"
    OTHER = "other"
    UNKNOWN = "unknown"


class AnalyzabilityStatus(str, Enum):
    FULLY_ANALYZABLE = "fully_analyzable"
    PARTIALLY_ANALYZABLE = "partially_analyzable"
    NOT_ANALYZABLE = "not_analyzable"


class ContractSection(BaseModel):
    section_id: str
    title: Optional[str] = None
    content: str
    page: Optional[int] = None


class Party(BaseModel):
    name: str
    role: Optional[str] = None
    evidence: Optional[Evidence] = None


class ContractProfile(BaseModel):
    contract_type: ContractType
    parties: List[Party]
    effective_date: Optional[date] = None
    commencement_date: Optional[date] = None
    expiration_date: Optional[date] = None
    term: Optional[DateRange] = None
    governing_law: Optional[str] = None
    summary: Optional[str] = None
    confidence: ConfidenceLevel


class ContractIntakeInput(BaseModel):
    """Input for the contract intake agent.

    source_file is the path to the local contract file. The agent loads and
    reads this file to build the LLM prompt.
    retrieved_evidence carries RAG-retrieved passages for grounding; may be empty
    if RAG setup was skipped (no OpenAI key).
    optional_contract_type and analysis_goal are caller hints that sharpen the prompt.
    """
    contract_id: str
    source_file: str
    retrieved_evidence: List[Evidence] = []
    optional_contract_type: Optional[ContractType] = None
    analysis_goal: Optional[str] = None


class ContractIntakeOutput(BaseModel):
    contract_id: Optional[str] = None
    profile: ContractProfile
    sections: List[ContractSection]
    analyzability: AnalyzabilityStatus
    analyzability_notes: Optional[str] = None
    confidence: ConfidenceLevel
