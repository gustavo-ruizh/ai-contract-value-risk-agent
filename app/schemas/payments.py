from enum import Enum
from typing import Optional, List
from datetime import date

from pydantic import BaseModel

from app.schemas.common import ConfidenceLevel, Evidence, MoneyAmount, DateRange
from app.schemas.contract import ContractProfile


class PaymentType(str, Enum):
    FIXED = "fixed"
    VARIABLE = "variable"
    MILESTONE = "milestone"
    RECURRING = "recurring"
    ONE_TIME = "one_time"
    CONTINGENT = "contingent"
    PENALTY = "penalty"
    DEPOSIT = "deposit"
    OTHER = "other"


class PaymentFrequency(str, Enum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    SEMI_ANNUALLY = "semi_annually"
    WEEKLY = "weekly"
    DAILY = "daily"
    ONE_TIME = "one_time"
    ON_MILESTONE = "on_milestone"
    IRREGULAR = "irregular"


class ValuationTreatment(str, Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"
    CONDITIONAL = "conditional"


class PaymentObligation(BaseModel):
    obligation_id: str
    description: str
    payment_type: PaymentType
    frequency: PaymentFrequency
    amount: Optional[MoneyAmount] = None
    amount_min: Optional[MoneyAmount] = None
    amount_max: Optional[MoneyAmount] = None
    due_date: Optional[date] = None
    date_range: Optional[DateRange] = None
    payer: Optional[str] = None
    payee: Optional[str] = None
    conditions: Optional[str] = None
    evidence: Optional[Evidence] = None
    confidence: ConfidenceLevel
    is_included_in_base_valuation: bool
    valuation_treatment: ValuationTreatment


class PaymentExtractionIssue(BaseModel):
    issue_id: str
    description: str
    severity: str  # "warning" | "error"
    related_section: Optional[str] = None


class PaymentObligationInput(BaseModel):
    """Input for the payment obligation extraction agent.

    retrieved_evidence carries RAG-retrieved passages that ground the LLM extraction.
    When retrieved_evidence is empty (RAG skipped), the agent uses only the contract
    profile metadata — extraction quality will be lower.
    """
    contract_id: str
    contract_profile: ContractProfile
    retrieved_evidence: List[Evidence] = []


class PaymentObligationOutput(BaseModel):
    contract_id: Optional[str] = None
    obligations: List[PaymentObligation]
    extraction_issues: List[PaymentExtractionIssue] = []
    uncertainty_flags: List[str] = []
    confidence: ConfidenceLevel
