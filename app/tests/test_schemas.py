"""Verifies that all schema modules can be imported and basic instantiation works."""
import pytest
from datetime import date

from app.schemas.common import Currency, ConfidenceLevel, Evidence, MoneyAmount, DateRange
from app.schemas.contract import (
    ContractType,
    AnalyzabilityStatus,
    ContractSection,
    Party,
    ContractProfile,
    ContractIntakeInput,
    ContractIntakeOutput,
)
from app.schemas.payments import (
    PaymentType,
    PaymentFrequency,
    ValuationTreatment,
    PaymentObligation,
    PaymentObligationInput,
    PaymentObligationOutput,
)
from app.schemas.valuation import (
    ValuationAssumption,
    ContractValueInput,
    ContractValueOutput,
)
from app.schemas.risks import (
    RiskSeverity,
    RiskCategory,
    MinimalValuationContext,
    RiskItem,
    RiskAssessmentInput,
    RiskAssessmentOutput,
)
from app.schemas.review import (
    ReviewDecision,
    FollowUpQuestion,
    ReviewDecisionInput,
    ReviewDecisionOutput,
)
from app.schemas.analysis import ContractAnalysisRequest


# --- common ---

def test_money_amount():
    m = MoneyAmount(amount=50_000.0, currency=Currency.USD)
    assert m.amount == 50_000.0
    assert m.currency == Currency.USD


def test_date_range():
    dr = DateRange(start=date(2024, 1, 1), end=date(2026, 12, 31))
    assert dr.start < dr.end


def test_evidence_with_sentinels():
    ev = Evidence(
        text="Payment due on the first of each month.",
        source="llm_extracted",
        chunk_id="llm_extracted",
        section="3.1",
        page=4,
    )
    assert ev.page == 4
    assert ev.source == "llm_extracted"
    assert ev.chunk_id == "llm_extracted"
    assert ev.relevance_score is None


def test_evidence_with_rag_metadata():
    ev = Evidence(
        text="Lessee shall pay $5,000 per month.",
        source="lease_agreement.pdf",
        page=3,
        chunk_id="contract_001_p3_c2",
        relevance_score=0.92,
    )
    assert ev.source == "lease_agreement.pdf"
    assert ev.chunk_id == "contract_001_p3_c2"
    assert ev.relevance_score == 0.92


def test_evidence_relevance_score_bounds():
    with pytest.raises(Exception):
        Evidence(
            text="test",
            source="llm_extracted",
            chunk_id="llm_extracted",
            relevance_score=1.5,
        )
    with pytest.raises(Exception):
        Evidence(
            text="test",
            source="llm_extracted",
            chunk_id="llm_extracted",
            relevance_score=-0.1,
        )


# --- contract ---

def test_contract_intake_input_minimal():
    inp = ContractIntakeInput(
        contract_id="test-001",
        source_file="/tmp/test.txt",
    )
    assert inp.contract_id == "test-001"
    assert inp.source_file == "/tmp/test.txt"
    assert inp.retrieved_evidence == []


def test_contract_profile():
    profile = ContractProfile(
        contract_type=ContractType.SERVICE_AGREEMENT,
        parties=[Party(name="Acme Corp", role="buyer")],
        confidence=ConfidenceLevel.HIGH,
    )
    assert profile.contract_type == ContractType.SERVICE_AGREEMENT
    assert len(profile.parties) == 1


# --- payments ---

def test_payment_obligation_required_valuation_fields():
    fields = PaymentObligation.__fields__
    assert "is_included_in_base_valuation" in fields
    assert "valuation_treatment" in fields


def test_payment_obligation_output_uncertainty_flags():
    fields = PaymentObligationOutput.__fields__
    assert "uncertainty_flags" in fields


def test_payment_obligation_construction():
    obligation = PaymentObligation(
        obligation_id="p-001",
        description="Monthly SaaS fee",
        payment_type=PaymentType.RECURRING,
        frequency=PaymentFrequency.MONTHLY,
        amount=MoneyAmount(amount=5_000.0, currency=Currency.USD),
        confidence=ConfidenceLevel.HIGH,
        is_included_in_base_valuation=True,
        valuation_treatment=ValuationTreatment.INCLUDE,
    )
    assert obligation.is_included_in_base_valuation is True
    assert obligation.valuation_treatment == ValuationTreatment.INCLUDE


# --- valuation ---

def test_valuation_assumption_uncertainty_flag():
    fields = ValuationAssumption.__fields__
    assert "source_uncertainty_flag" in fields


def test_contract_value_output_uncertainty_flags():
    fields = ContractValueOutput.__fields__
    assert "uncertainty_flags" in fields


# --- risks ---

def test_minimal_valuation_context_holds_assumptions():
    ctx = MinimalValuationContext(
        assumptions=[
            ValuationAssumption(
                assumption_id="assume-001",
                description="Amount estimated as midpoint of 8000–12000 USD.",
                value="10000.0",
                source_uncertainty_flag=True,
            )
        ],
        uncertainty_flags=["amount_estimated:p-001"],
    )
    assert len(ctx.assumptions) == 1
    assert len(ctx.uncertainty_flags) == 1
    assert ctx.assumptions[0].assumption_id == "assume-001"


def test_minimal_valuation_context_empty_defaults():
    ctx = MinimalValuationContext()
    assert ctx.assumptions == []
    assert ctx.uncertainty_flags == []


def test_risk_item_required_fields():
    fields = RiskItem.__fields__
    assert "is_blocking" in fields
    assert "risk_priority_score" in fields


def test_risk_assessment_output_uncertainty_flags():
    fields = RiskAssessmentOutput.__fields__
    assert "uncertainty_flags" in fields


def test_risk_assessment_input_uses_minimal_valuation_context():
    # RiskAssessmentInput must accept MinimalValuationContext, not ContractValueOutput
    field = RiskAssessmentInput.__fields__["valuation_context"]
    # outer_type_ is Pydantic v1; annotation is Pydantic v2
    annotation = getattr(field, "annotation", None) or getattr(field, "outer_type_", None)
    assert annotation is MinimalValuationContext


def test_risk_assessment_input_has_retrieved_evidence():
    fields = RiskAssessmentInput.__fields__
    assert "retrieved_evidence" in fields


# --- review ---

def test_review_decision_enum_values():
    assert ReviewDecision.APPROVE.value == "approve"
    assert ReviewDecision.FINANCE_REVIEW.value == "finance_review"
    assert ReviewDecision.LEGAL_REVIEW.value == "legal_review"
    assert ReviewDecision.CLARIFICATION_REQUIRED.value == "clarification_required"


def test_review_decision_output_decision_factors():
    fields = ReviewDecisionOutput.__fields__
    assert "decision_factors" in fields
    assert "decision_uncertainty_flags" in fields


# --- analysis ---

def test_contract_analysis_request_uses_contract_id():
    req = ContractAnalysisRequest(
        contract_id="c-001",
        source_file="/tmp/sample_contract.txt",
    )
    assert req.contract_id == "c-001"
    assert req.discount_rate is None
    assert req.valuation_date is None
