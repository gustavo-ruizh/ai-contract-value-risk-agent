from enum import Enum
from typing import Optional, List

from pydantic import BaseModel

from app.schemas.common import ConfidenceLevel, Evidence
from app.schemas.contract import ContractProfile
from app.schemas.payments import PaymentObligation
from app.schemas.valuation import ValuationAssumption


class RiskSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RiskCategory(str, Enum):
    FINANCIAL = "financial"
    LEGAL = "legal"
    OPERATIONAL = "operational"
    COMPLIANCE = "compliance"
    COUNTERPARTY = "counterparty"


class MinimalValuationContext(BaseModel):
    """Lightweight valuation summary passed to risk assessment.

    Contains only the assumptions and uncertainty flags produced during valuation —
    not the monetary totals. This avoids coupling the risk stage to final dollar
    amounts that may themselves be based on uncertain assumptions.

    The risk agent uses assumptions to understand *what was estimated* and
    uncertainty_flags to know *where confidence is low*.
    """
    assumptions: List[ValuationAssumption] = []
    uncertainty_flags: List[str] = []


class RiskItem(BaseModel):
    risk_id: str
    title: str
    description: str
    category: RiskCategory
    severity: RiskSeverity
    evidence: Optional[Evidence] = None
    mitigation: Optional[str] = None
    is_blocking: bool = False
    risk_priority_score: Optional[float] = None


class RiskAssessmentInput(BaseModel):
    contract_id: Optional[str] = None
    contract_profile: ContractProfile
    obligations: List[PaymentObligation]
    valuation_context: MinimalValuationContext
    retrieved_evidence: List[Evidence] = []


class RiskAssessmentOutput(BaseModel):
    contract_id: Optional[str] = None
    risks: List[RiskItem]
    overall_risk_level: RiskSeverity
    uncertainty_flags: List[str] = []
    confidence: ConfidenceLevel
