"""Tests for ContractAnalysisOrchestrator wiring and happy-path execution."""
from datetime import date
from unittest.mock import MagicMock

import pytest

from app.agents.contract_intake_agent import ContractIntakeAgent
from app.agents.contract_value_agent import ContractValueAgent
from app.agents.payment_obligation_agent import PaymentObligationAgent
from app.agents.review_decision_agent import ReviewDecisionAgent
from app.agents.risk_assessment_agent import RiskAssessmentAgent
from app.orchestrator.contract_analysis_orchestrator import ContractAnalysisOrchestrator
from app.schemas.analysis import ContractAnalysisRequest, ContractAnalysisResult
from app.schemas.common import ConfidenceLevel, Currency, MoneyAmount
from app.schemas.contract import (
    AnalyzabilityStatus,
    ContractIntakeOutput,
    ContractProfile,
    ContractType,
)
from app.schemas.payments import PaymentObligationOutput
from app.schemas.review import ReviewDecision, ReviewDecisionOutput
from app.schemas.risks import RiskAssessmentOutput, RiskSeverity
from app.schemas.valuation import ContractValueOutput

_DEFAULT_SOURCE = "sample_data/sample_contract.txt"


def _make_intake_output():
    return ContractIntakeOutput(
        contract_id="c-001",
        profile=ContractProfile(
            contract_type=ContractType.SERVICE_AGREEMENT,
            parties=[],
            confidence=ConfidenceLevel.HIGH,
        ),
        sections=[],
        analyzability=AnalyzabilityStatus.FULLY_ANALYZABLE,
        confidence=ConfidenceLevel.HIGH,
    )


def _make_payment_output():
    return PaymentObligationOutput(
        contract_id="c-001",
        obligations=[],
        extraction_issues=[],
        uncertainty_flags=[],
        confidence=ConfidenceLevel.HIGH,
    )


def _make_value_output():
    return ContractValueOutput(
        contract_id="c-001",
        total_nominal_value=MoneyAmount(amount=0.0, currency=Currency.USD),
        total_present_value=MoneyAmount(amount=0.0, currency=Currency.USD),
        payment_schedule=[],
        assumptions=[],
        warnings=[],
        uncertainty_flags=[],
        confidence=ConfidenceLevel.HIGH,
    )


def _make_risk_output():
    return RiskAssessmentOutput(
        contract_id="c-001",
        risks=[],
        overall_risk_level=RiskSeverity.LOW,
        uncertainty_flags=[],
        confidence=ConfidenceLevel.HIGH,
    )


def _make_review_output():
    return ReviewDecisionOutput(
        contract_id="c-001",
        decision=ReviewDecision.APPROVE,
        rationale="No blocking or high-severity risks found.",
        follow_up_questions=[],
        decision_factors=["No blocking or high-severity risks identified"],
        decision_uncertainty_flags=[],
        confidence=ConfidenceLevel.HIGH,
    )


def _build_orchestrator_with_mocks():
    intake = MagicMock(spec=ContractIntakeAgent)
    intake.run.return_value = _make_intake_output()

    payment = MagicMock(spec=PaymentObligationAgent)
    payment.run.return_value = _make_payment_output()

    value = MagicMock(spec=ContractValueAgent)
    value.run.return_value = _make_value_output()

    risk = MagicMock(spec=RiskAssessmentAgent)
    risk.run.return_value = _make_risk_output()

    review = MagicMock(spec=ReviewDecisionAgent)
    review.run.return_value = _make_review_output()

    orch = ContractAnalysisOrchestrator(
        intake_agent=intake,
        payment_agent=payment,
        value_agent=value,
        risk_agent=risk,
        review_agent=review,
    )
    return orch, intake, payment, value, risk, review


def test_orchestrator_instantiates():
    orch, *_ = _build_orchestrator_with_mocks()
    assert orch is not None


def test_orchestrator_holds_all_five_agents():
    orch, *_ = _build_orchestrator_with_mocks()
    assert isinstance(orch.intake_agent, ContractIntakeAgent)
    assert isinstance(orch.payment_agent, PaymentObligationAgent)
    assert isinstance(orch.value_agent, ContractValueAgent)
    assert isinstance(orch.risk_agent, RiskAssessmentAgent)
    assert isinstance(orch.review_agent, ReviewDecisionAgent)


def test_orchestrator_run_returns_analysis_result():
    orch, *_ = _build_orchestrator_with_mocks()
    req = ContractAnalysisRequest(
        contract_id="c-001",
        source_file=_DEFAULT_SOURCE,
        valuation_date=date(2025, 1, 1),
    )
    result = orch.run(req)
    assert isinstance(result, ContractAnalysisResult)
    assert result.contract_id == "c-001"


def test_orchestrator_calls_each_agent_once():
    orch, intake, payment, value, risk, review = _build_orchestrator_with_mocks()
    req = ContractAnalysisRequest(contract_id="c-001", source_file=_DEFAULT_SOURCE)
    orch.run(req)
    intake.run.assert_called_once()
    payment.run.assert_called_once()
    value.run.assert_called_once()
    risk.run.assert_called_once()
    review.run.assert_called_once()


def test_orchestrator_approve_result_propagates():
    orch, *_ = _build_orchestrator_with_mocks()
    req = ContractAnalysisRequest(contract_id="c-001", source_file=_DEFAULT_SOURCE)
    result = orch.run(req)
    assert result.review_decision.decision == ReviewDecision.APPROVE


def test_orchestrator_skips_rag_without_retriever():
    """Pipeline completes with empty retrieved_evidence when RAG setup fails (e.g. file not found)."""
    orch, _, _, _, risk, _ = _build_orchestrator_with_mocks()
    req = ContractAnalysisRequest(contract_id="c-001", source_file="/nonexistent/contract.txt")
    orch.run(req)
    # _setup_rag fails gracefully on missing file — retrieved_evidence must be empty
    call_args = risk.run.call_args[0][0]
    assert call_args.retrieved_evidence == []


def test_orchestrator_passes_valuation_assumptions_to_risk_agent():
    """MinimalValuationContext should carry assumptions, not raw monetary totals."""
    from app.schemas.valuation import ValuationAssumption
    from app.schemas.risks import MinimalValuationContext

    orch, _, _, value, risk, _ = _build_orchestrator_with_mocks()

    assumption = ValuationAssumption(
        assumption_id="assume-001",
        description="Amount estimated as midpoint",
        value="10000.0",
        source_uncertainty_flag=True,
    )
    value_out = _make_value_output()
    value_out.assumptions = [assumption]
    value_out.uncertainty_flags = ["amount_estimated:p-001"]
    value.run.return_value = value_out

    req = ContractAnalysisRequest(contract_id="c-001", source_file=_DEFAULT_SOURCE)
    orch.run(req)

    risk_input = risk.run.call_args[0][0]
    assert isinstance(risk_input.valuation_context, MinimalValuationContext)
    assert len(risk_input.valuation_context.assumptions) == 1
    assert risk_input.valuation_context.assumptions[0].assumption_id == "assume-001"
