"""Tests for ReviewDecisionAgent decision logic and risk sorting."""
from app.agents.review_decision_agent import ReviewDecisionAgent, _sort_risks
from app.schemas.common import ConfidenceLevel
from app.schemas.contract import ContractProfile, ContractType
from app.schemas.review import ReviewDecision, ReviewDecisionInput
from app.schemas.risks import RiskCategory, RiskItem, RiskSeverity


def _make_profile():
    return ContractProfile(
        contract_type=ContractType.SERVICE_AGREEMENT,
        parties=[],
        confidence=ConfidenceLevel.HIGH,
    )


def _make_risk(risk_id="r-001", severity=RiskSeverity.MEDIUM,
               category=RiskCategory.FINANCIAL, is_blocking=False,
               priority_score=20.0):
    return RiskItem(
        risk_id=risk_id,
        title=f"Risk {risk_id}",
        description="Test risk",
        category=category,
        severity=severity,
        is_blocking=is_blocking,
        risk_priority_score=priority_score,
    )


def test_no_risks_returns_approve():
    agent = ReviewDecisionAgent()
    inp = ReviewDecisionInput(
        contract_profile=_make_profile(),
        risks=[],
        overall_risk_level=RiskSeverity.LOW,
    )
    out = agent.run(inp)
    assert out.decision == ReviewDecision.APPROVE


def test_blocking_legal_risk_returns_legal_review():
    agent = ReviewDecisionAgent()
    risk = _make_risk(
        severity=RiskSeverity.HIGH,
        category=RiskCategory.LEGAL,
        is_blocking=True,
        priority_score=130.0,
    )
    inp = ReviewDecisionInput(
        contract_profile=_make_profile(),
        risks=[risk],
        overall_risk_level=RiskSeverity.HIGH,
    )
    out = agent.run(inp)
    assert out.decision == ReviewDecision.LEGAL_REVIEW


def test_blocking_financial_risk_returns_clarification_required():
    agent = ReviewDecisionAgent()
    risk = _make_risk(
        severity=RiskSeverity.HIGH,
        category=RiskCategory.FINANCIAL,
        is_blocking=True,
        priority_score=130.0,
    )
    inp = ReviewDecisionInput(
        contract_profile=_make_profile(),
        risks=[risk],
        overall_risk_level=RiskSeverity.HIGH,
    )
    out = agent.run(inp)
    assert out.decision == ReviewDecision.CLARIFICATION_REQUIRED


def test_critical_risk_without_blocking_returns_finance_review():
    agent = ReviewDecisionAgent()
    risk = _make_risk(severity=RiskSeverity.CRITICAL, is_blocking=False, priority_score=40.0)
    inp = ReviewDecisionInput(
        contract_profile=_make_profile(),
        risks=[risk],
        overall_risk_level=RiskSeverity.CRITICAL,
    )
    out = agent.run(inp)
    assert out.decision == ReviewDecision.FINANCE_REVIEW


def test_high_legal_risk_returns_legal_review():
    agent = ReviewDecisionAgent()
    risk = _make_risk(
        severity=RiskSeverity.HIGH,
        category=RiskCategory.LEGAL,
        is_blocking=False,
        priority_score=30.0,
    )
    inp = ReviewDecisionInput(
        contract_profile=_make_profile(),
        risks=[risk],
        overall_risk_level=RiskSeverity.HIGH,
    )
    out = agent.run(inp)
    assert out.decision == ReviewDecision.LEGAL_REVIEW


def test_sort_risks_blocking_first():
    blocking = _make_risk("r-001", severity=RiskSeverity.LOW, is_blocking=True, priority_score=110.0)
    non_blocking = _make_risk("r-002", severity=RiskSeverity.CRITICAL, is_blocking=False, priority_score=40.0)
    sorted_risks = _sort_risks([non_blocking, blocking])
    assert sorted_risks[0].risk_id == "r-001"


def test_sort_risks_by_severity_within_same_blocking():
    high = _make_risk("r-001", severity=RiskSeverity.HIGH, is_blocking=False, priority_score=30.0)
    medium = _make_risk("r-002", severity=RiskSeverity.MEDIUM, is_blocking=False, priority_score=20.0)
    sorted_risks = _sort_risks([medium, high])
    assert sorted_risks[0].risk_id == "r-001"


def test_follow_up_questions_generated_for_blocking_risk():
    agent = ReviewDecisionAgent()
    risk = _make_risk(severity=RiskSeverity.HIGH, is_blocking=True, priority_score=130.0)
    inp = ReviewDecisionInput(
        contract_profile=_make_profile(),
        risks=[risk],
        overall_risk_level=RiskSeverity.HIGH,
    )
    out = agent.run(inp)
    assert len(out.follow_up_questions) >= 1
    assert out.follow_up_questions[0].related_risk_id == risk.risk_id


def test_approve_result_has_high_confidence_with_no_flags():
    agent = ReviewDecisionAgent()
    inp = ReviewDecisionInput(
        contract_profile=_make_profile(),
        risks=[],
        overall_risk_level=RiskSeverity.LOW,
    )
    out = agent.run(inp)
    assert out.confidence == ConfidenceLevel.HIGH
