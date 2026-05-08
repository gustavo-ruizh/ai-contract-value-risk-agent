from typing import Optional
from datetime import date

from pydantic import BaseModel

from app.schemas.contract import ContractIntakeOutput, ContractType
from app.schemas.payments import PaymentObligationOutput
from app.schemas.valuation import ContractValueOutput
from app.schemas.risks import RiskAssessmentOutput
from app.schemas.review import ReviewDecisionOutput


class ContractAnalysisRequest(BaseModel):
    """Top-level request for the /analyze endpoint.

    source_file is a path to a local contract file (.txt or .pdf).
    The orchestrator loads, chunks, and embeds the file before running the pipeline.

    valuation_date is optional and not in the finalized design spec, but retained
    so callers can pin the PV calculation to a specific date (useful for tests and
    historical analysis). Defaults to today when omitted.
    """
    contract_id: str
    source_file: str
    discount_rate: Optional[float] = None
    optional_contract_type: Optional[ContractType] = None
    analysis_goal: Optional[str] = None
    valuation_date: Optional[date] = None


class ContractAnalysisResult(BaseModel):
    contract_id: Optional[str] = None
    intake: ContractIntakeOutput
    payment_obligations: PaymentObligationOutput
    contract_value: ContractValueOutput
    risk_assessment: RiskAssessmentOutput
    review_decision: ReviewDecisionOutput
