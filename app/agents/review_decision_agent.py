import logging
from typing import List

from app.schemas.common import ConfidenceLevel
from app.schemas.review import (
    FollowUpQuestion,
    ReviewDecision,
    ReviewDecisionInput,
    ReviewDecisionOutput,
)
from app.schemas.risks import RiskItem, RiskSeverity

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {
    RiskSeverity.CRITICAL: 0,
    RiskSeverity.HIGH: 1,
    RiskSeverity.MEDIUM: 2,
    RiskSeverity.LOW: 3,
}


class ReviewDecisionAgent:
    """
    Evaluates risk severity, uncertainty flags, and blocking risks to decide
    whether the contract is approved or requires finance, legal, or clarification.
    Fully deterministic — no LLM calls.
    """

    def run(self, input_model: ReviewDecisionInput) -> ReviewDecisionOutput:
        logger.info("ReviewDecisionAgent: starting — contract_id=%s", input_model.contract_id)

        sorted_risks = _sort_risks(input_model.risks)
        decision, rationale, factors = _make_decision(
            sorted_risks,
            input_model.overall_risk_level,
            input_model.valuation_uncertainty_flags,
            input_model.risk_uncertainty_flags,
        )
        follow_ups = _build_follow_up_questions(sorted_risks)
        uncertainty_flags = _collect_decision_flags(
            sorted_risks,
            input_model.valuation_uncertainty_flags,
            input_model.risk_uncertainty_flags,
        )
        confidence = _derive_confidence(uncertainty_flags, sorted_risks)

        output = ReviewDecisionOutput(
            contract_id=input_model.contract_id,
            decision=decision,
            rationale=rationale,
            follow_up_questions=follow_ups,
            decision_factors=factors,
            decision_uncertainty_flags=uncertainty_flags,
            confidence=confidence,
        )
        logger.info(
            "ReviewDecisionAgent: complete — decision=%s confidence=%s",
            decision,
            confidence,
        )
        return output


# --- helpers ---

def _sort_risks(risks: List[RiskItem]) -> List[RiskItem]:
    """Blocking first, then by severity, then by priority score descending."""
    return sorted(
        risks,
        key=lambda r: (
            0 if r.is_blocking else 1,
            _SEVERITY_ORDER.get(r.severity, 3),
            -(r.risk_priority_score or 0),
        ),
    )


def _make_decision(
    risks: List[RiskItem],
    overall_risk_level: RiskSeverity,
    valuation_flags: List[str],
    risk_flags: List[str],
) -> tuple:
    factors: List[str] = []

    blocking_risks = [r for r in risks if r.is_blocking]
    critical_risks = [r for r in risks if r.severity == RiskSeverity.CRITICAL]
    high_risks = [r for r in risks if r.severity == RiskSeverity.HIGH]

    legal_categories = {"legal", "compliance"}
    has_legal_blocking = any(
        r.is_blocking and r.category.value in legal_categories for r in risks
    )
    has_financial_blocking = any(
        r.is_blocking and r.category.value == "financial" for r in risks
    )

    all_flags = list(dict.fromkeys(valuation_flags + risk_flags))

    if blocking_risks and has_legal_blocking:
        factors.append(f"{len(blocking_risks)} blocking risk(s) with legal/compliance exposure")
        rationale = (
            f"Contract has {len(blocking_risks)} blocking risk(s) including legal or compliance issues. "
            "Legal review is required before proceeding."
        )
        return ReviewDecision.LEGAL_REVIEW, rationale, factors

    if blocking_risks and has_financial_blocking:
        factors.append(f"{len(blocking_risks)} blocking financial risk(s)")
        rationale = (
            f"Contract has {len(blocking_risks)} blocking financial risk(s) that must be resolved. "
            "Clarification is required before approval."
        )
        return ReviewDecision.CLARIFICATION_REQUIRED, rationale, factors

    if blocking_risks:
        factors.append(f"{len(blocking_risks)} blocking risk(s) requiring clarification")
        rationale = (
            f"Contract has {len(blocking_risks)} blocking risk(s). "
            "Clarification required to resolve ambiguities before approval."
        )
        return ReviewDecision.CLARIFICATION_REQUIRED, rationale, factors

    if critical_risks:
        factors.append(f"{len(critical_risks)} critical risk(s) identified")
        has_critical_legal = any(
            r.severity == RiskSeverity.CRITICAL and r.category.value in legal_categories
            for r in risks
        )
        if has_critical_legal:
            rationale = (
                f"Contract has {len(critical_risks)} critical risk(s) including legal/compliance exposure. "
                "Legal review is required."
            )
            return ReviewDecision.LEGAL_REVIEW, rationale, factors
        rationale = (
            f"Contract has {len(critical_risks)} critical risk(s). "
            "Finance review required before approval."
        )
        return ReviewDecision.FINANCE_REVIEW, rationale, factors

    if high_risks:
        factors.append(f"{len(high_risks)} high-severity risk(s) identified")
        has_high_legal = any(
            r.severity == RiskSeverity.HIGH and r.category.value in legal_categories
            for r in risks
        )
        if has_high_legal:
            rationale = (
                f"Contract has {len(high_risks)} high-severity risk(s) including legal exposure. "
                "Legal review recommended."
            )
            return ReviewDecision.LEGAL_REVIEW, rationale, factors
        rationale = (
            f"Contract has {len(high_risks)} high-severity risk(s). "
            "Finance review recommended."
        )
        return ReviewDecision.FINANCE_REVIEW, rationale, factors

    if len(all_flags) > 3:
        factors.append(f"{len(all_flags)} uncertainty flags accumulated")
        rationale = (
            f"Contract has {len(all_flags)} uncertainty flags requiring clarification. "
            "Clarification recommended before final approval."
        )
        return ReviewDecision.CLARIFICATION_REQUIRED, rationale, factors

    factors.append("No blocking or high-severity risks identified")
    if risks:
        factors.append(f"{len(risks)} medium/low risk(s) within acceptable range")
    rationale = "No blocking or high-severity risks found. Contract is approved for standard processing."
    return ReviewDecision.APPROVE, rationale, factors


def _build_follow_up_questions(sorted_risks: List[RiskItem]) -> List[FollowUpQuestion]:
    questions = []
    top_risks = sorted_risks[:5]
    for i, risk in enumerate(top_risks):
        if risk.severity in (RiskSeverity.CRITICAL, RiskSeverity.HIGH) or risk.is_blocking:
            audience = _audience_for_risk(risk)
            questions.append(FollowUpQuestion(
                question_id=f"q-{i + 1:03d}",
                question=f"Regarding risk '{risk.title}': {_question_text(risk)}",
                target_audience=audience,
                related_risk_id=risk.risk_id,
            ))
    return questions


def _audience_for_risk(risk: RiskItem) -> str:
    legal_categories = {"legal", "compliance"}
    if risk.category.value in legal_categories:
        return "legal"
    if risk.category.value == "financial":
        return "finance"
    return "requester"


def _question_text(risk: RiskItem) -> str:
    if risk.is_blocking:
        return f"This is a blocking issue — {risk.description} How should this be resolved before approval?"
    if risk.severity == RiskSeverity.CRITICAL:
        return f"This is a critical risk — {risk.description} What mitigations are in place?"
    return f"{risk.description} Has this been reviewed and accepted?"


def _collect_decision_flags(
    risks: List[RiskItem],
    valuation_flags: List[str],
    risk_flags: List[str],
) -> List[str]:
    flags = list(dict.fromkeys(list(valuation_flags) + list(risk_flags)))
    if any(r.is_blocking for r in risks):
        flags.append("has_blocking_risks")
    if any(r.severity == RiskSeverity.CRITICAL for r in risks):
        flags.append("has_critical_risks")
    return flags


def _derive_confidence(flags: List[str], risks: List[RiskItem]) -> ConfidenceLevel:
    if "llm_risk_assessment_failed" in flags or "llm_extraction_failed" in flags:
        return ConfidenceLevel.LOW
    if len(flags) > 4 or any(r.severity == RiskSeverity.CRITICAL for r in risks):
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.HIGH
