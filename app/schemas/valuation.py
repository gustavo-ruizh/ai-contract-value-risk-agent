from typing import Optional, List
from datetime import date

from pydantic import BaseModel

from app.schemas.common import ConfidenceLevel, MoneyAmount
from app.schemas.payments import PaymentObligation


class PaymentScheduleLine(BaseModel):
    obligation_id: str
    description: str
    payment_date: date
    nominal_amount: MoneyAmount
    present_value: MoneyAmount
    discount_rate_applied: float
    periods_discounted: int


class ValuationAssumption(BaseModel):
    assumption_id: str
    description: str
    value: str
    source_uncertainty_flag: bool = False


class CalculationWarning(BaseModel):
    warning_id: str
    description: str
    affected_obligation_ids: List[str] = []


class ContractValueInput(BaseModel):
    contract_id: Optional[str] = None
    obligations: List[PaymentObligation]
    valuation_date: date
    discount_rate: float
    uncertainty_flags: List[str] = []


class ContractValueOutput(BaseModel):
    contract_id: Optional[str] = None
    total_nominal_value: MoneyAmount
    total_present_value: MoneyAmount
    payment_schedule: List[PaymentScheduleLine]
    assumptions: List[ValuationAssumption]
    warnings: List[CalculationWarning] = []
    uncertainty_flags: List[str] = []
    confidence: ConfidenceLevel
