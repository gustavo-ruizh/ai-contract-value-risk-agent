import json
import logging
from typing import List, Optional

from app.schemas.common import ConfidenceLevel, Evidence
from app.schemas.risks import (
    MinimalValuationContext,
    RiskAssessmentInput,
    RiskAssessmentOutput,
    RiskCategory,
    RiskItem,
    RiskSeverity,
)
from app.services.llm_client import LLMClient, LLMError

logger = logging.getLogger(__name__)

_SEVERITY_WEIGHT = {
    RiskSeverity.CRITICAL: 4,
    RiskSeverity.HIGH: 3,
    RiskSeverity.MEDIUM: 2,
    RiskSeverity.LOW: 1,
}

_MAX_OBLIGATIONS_IN_PROMPT = 20


class RiskAssessmentAgent:
    """
    Identifies financial and legal risks from the contract profile,
    obligations, and valuation context. Scores and prioritizes each risk item.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def run(self, input_model: RiskAssessmentInput) -> RiskAssessmentOutput:
        logger.info("RiskAssessmentAgent: starting — contract_id=%s", input_model.contract_id)
        try:
            prompt = self._build_prompt(input_model)
            raw = self.llm.generate_json(prompt)
            output = self._parse_response(raw, input_model.contract_id)
            logger.info(
                "RiskAssessmentAgent: complete — %d risks, overall=%s",
                len(output.risks),
                output.overall_risk_level,
            )
            return output
        except LLMError as exc:
            logger.error("RiskAssessmentAgent: LLM failed — %s", exc)
            return self._fallback_output(input_model)

    def _build_prompt(self, input_model: RiskAssessmentInput) -> str:
        profile = input_model.contract_profile
        parties_text = ", ".join(
            f"{p.name} ({p.role or 'party'})" for p in profile.parties
        ) or "unknown"

        obligations_subset = input_model.obligations[:_MAX_OBLIGATIONS_IN_PROMPT]
        obligations_json = json.dumps(
            [
                {
                    "obligation_id": ob.obligation_id,
                    "description": ob.description,
                    "payment_type": ob.payment_type.value,
                    "frequency": ob.frequency.value,
                    "amount": ob.amount.model_dump() if ob.amount else None,
                    "amount_min": ob.amount_min.model_dump() if ob.amount_min else None,
                    "amount_max": ob.amount_max.model_dump() if ob.amount_max else None,
                    "due_date": str(ob.due_date) if ob.due_date else None,
                    "date_range": (
                        {"start": str(ob.date_range.start), "end": str(ob.date_range.end)}
                        if ob.date_range else None
                    ),
                    "conditions": ob.conditions,
                    "confidence": ob.confidence.value,
                    "valuation_treatment": ob.valuation_treatment.value,
                }
                for ob in obligations_subset
            ],
            indent=2,
        )

        # Summarise what the valuation stage assumed — tells the risk agent
        # where valuation confidence is low without exposing final dollar amounts.
        vc = input_model.valuation_context
        if vc.assumptions:
            assumption_lines = "; ".join(a.description for a in vc.assumptions[:5])
            valuation_summary = f"Valuation assumptions: {assumption_lines}"
        else:
            valuation_summary = "No valuation assumptions recorded."
        if vc.uncertainty_flags:
            valuation_summary += f" Uncertainty flags: {vc.uncertainty_flags}"

        # RAG-retrieved evidence passages grounded in the original contract text
        evidence_section = ""
        if input_model.retrieved_evidence:
            lines = [
                f"- [p.{e.page or '?'} {e.source or 'contract'}] (score={e.relevance_score:.2f}): "
                f"{e.text[:200]}"
                for e in input_model.retrieved_evidence[:5]
                if e.relevance_score is not None
            ] or [
                f"- [p.{e.page or '?'} {e.source or 'contract'}]: {e.text[:200]}"
                for e in input_model.retrieved_evidence[:5]
            ]
            evidence_section = "\nRETRIEVED CONTRACT PASSAGES (for grounding):\n" + "\n".join(lines)

        return f"""You are a contract risk analyst.

Identify ALL material risks in the contract described below.

CONTRACT TYPE: {profile.contract_type.value}
PARTIES: {parties_text}
EFFECTIVE DATE: {profile.effective_date or 'unknown'}
EXPIRATION DATE: {profile.expiration_date or 'unknown'}
SUMMARY: {profile.summary or 'N/A'}
VALUATION CONTEXT: {valuation_summary}{evidence_section}

PAYMENT OBLIGATIONS:
{obligations_json}

Identify every material risk. Focus on:
- Missing critical payment amounts or dates (financial risk, potentially blocking)
- Ambiguous or conflicting payment clauses
- Termination penalties or automatic renewals with unclear financial impact
- Conditional escalation clauses (variable cost risk)
- High concentration in a single counterparty or obligation
- Compliance, legal jurisdiction, or regulatory exposure
- Counterparty credit or performance risk

For evidence: set page to the integer from the [p.X filename] label of the passage the quote comes from, or null if the observation is synthesized rather than from a retrieved passage. Set section to the clause or section name if identifiable, otherwise null.

Mark is_blocking=true when the risk materially prevents reliable valuation or has critical legal exposure.
Blocking risks include: missing critical payment amount, missing critical dates, conflicting payment clauses,
ambiguous renewal obligations that affect value.

Return JSON:
{{
  "risks": [
    {{
      "risk_id": "r-001",
      "title": "short title",
      "description": "detailed description of the risk",
      "category": "financial|legal|operational|compliance|counterparty",
      "severity": "critical|high|medium|low",
      "evidence": {{"text": "exact quote or observation", "page": integer_or_null, "section": "section_name or null"}} or null,
      "mitigation": "suggested mitigation or null",
      "is_blocking": false
    }}
  ],
  "overall_risk_level": "critical|high|medium|low",
  "uncertainty_flags": [],
  "confidence": "high|medium|low"
}}

Return only valid JSON. No markdown, no explanation."""

    def _parse_response(self, raw: dict, contract_id: Optional[str]) -> RiskAssessmentOutput:
        risks = []
        for i, r in enumerate(raw.get("risks", [])):
            try:
                risks.append(_parse_risk_item(r, i))
            except Exception as exc:
                logger.warning("Skipping unparseable risk %d: %s", i, exc)

        _assign_priority_scores(risks)

        return RiskAssessmentOutput(
            contract_id=contract_id,
            risks=risks,
            overall_risk_level=_safe_enum(
                RiskSeverity, raw.get("overall_risk_level"), _derive_overall_level(risks)
            ),
            uncertainty_flags=raw.get("uncertainty_flags", []),
            confidence=_safe_enum(ConfidenceLevel, raw.get("confidence"), ConfidenceLevel.MEDIUM),
        )

    def _fallback_output(self, input_model: RiskAssessmentInput) -> RiskAssessmentOutput:
        return RiskAssessmentOutput(
            contract_id=input_model.contract_id,
            risks=[],
            overall_risk_level=RiskSeverity.LOW,
            uncertainty_flags=["llm_risk_assessment_failed"],
            confidence=ConfidenceLevel.LOW,
        )


# --- helpers ---

def _parse_risk_item(r: dict, index: int) -> RiskItem:
    evidence = None
    if r.get("evidence") and isinstance(r["evidence"], dict):
        text = r["evidence"].get("text", "")
        if text:
            evidence = Evidence(
                    text=text,
                    source="llm_extracted",
                    chunk_id="llm_extracted",
                    page=r["evidence"].get("page"),
                    section=r["evidence"].get("section"),
                )

    item = RiskItem(
        risk_id=r.get("risk_id", f"r-{index + 1:03d}"),
        title=r.get("title", "Unnamed risk"),
        description=r.get("description", ""),
        category=_safe_enum(RiskCategory, r.get("category"), RiskCategory.FINANCIAL),
        severity=_safe_enum(RiskSeverity, r.get("severity"), RiskSeverity.MEDIUM),
        evidence=evidence,
        mitigation=r.get("mitigation"),
        is_blocking=bool(r.get("is_blocking", False)),
    )
    return item


def _assign_priority_scores(risks: List[RiskItem]) -> None:
    for risk in risks:
        blocking_bonus = 100 if risk.is_blocking else 0
        severity_score = _SEVERITY_WEIGHT.get(risk.severity, 1) * 10
        risk.risk_priority_score = float(blocking_bonus + severity_score)


def _derive_overall_level(risks: List[RiskItem]) -> RiskSeverity:
    if not risks:
        return RiskSeverity.LOW
    if any(r.severity == RiskSeverity.CRITICAL for r in risks):
        return RiskSeverity.CRITICAL
    if any(r.severity == RiskSeverity.HIGH for r in risks):
        return RiskSeverity.HIGH
    if any(r.severity == RiskSeverity.MEDIUM for r in risks):
        return RiskSeverity.MEDIUM
    return RiskSeverity.LOW


def _safe_enum(enum_cls, value, default):
    try:
        return enum_cls(value)
    except (ValueError, KeyError, TypeError):
        return default
