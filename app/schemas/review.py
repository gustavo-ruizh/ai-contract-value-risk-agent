from enum import Enum
from typing import Optional, List

from pydantic import BaseModel

from app.schemas.common import ConfidenceLevel
from app.schemas.contract import ContractProfile
from app.schemas.risks import RiskSeverity, RiskItem


class ReviewDecision(str, Enum):
    APPROVE = "approve"
    FINANCE_REVIEW = "finance_review"
    LEGAL_REVIEW = "legal_review"
    CLARIFICATION_REQUIRED = "clarification_required"


class FollowUpQuestion(BaseModel):
    question_id: str
    question: str
    target_audience: str  # "finance" | "legal" | "requester"
    related_risk_id: Optional[str] = None


class ReviewDecisionInput(BaseModel):
    contract_id: Optional[str] = None
    contract_profile: ContractProfile
    risks: List[RiskItem]
    overall_risk_level: RiskSeverity
    valuation_uncertainty_flags: List[str] = []
    risk_uncertainty_flags: List[str] = []


class ReviewDecisionOutput(BaseModel):
    contract_id: Optional[str] = None
    decision: ReviewDecision
    rationale: str
    follow_up_questions: List[FollowUpQuestion] = []
    decision_factors: List[str] = []
    decision_uncertainty_flags: List[str] = []
    confidence: ConfidenceLevel
