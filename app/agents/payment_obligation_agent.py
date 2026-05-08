import logging
from datetime import date
from typing import List, Optional

from app.schemas.common import ConfidenceLevel, Evidence, MoneyAmount
from app.schemas.payments import (
    PaymentExtractionIssue,
    PaymentFrequency,
    PaymentObligation,
    PaymentObligationInput,
    PaymentObligationOutput,
    PaymentType,
    ValuationTreatment,
)
from app.schemas.common import Currency, DateRange
from app.services.llm_client import LLMClient, LLMError

logger = logging.getLogger(__name__)


class PaymentObligationAgent:
    """
    Extracts every payment obligation from RAG-retrieved contract passages.
    Assigns valuation treatment and raises uncertainty flags for ambiguous clauses.
    When retrieved_evidence is empty, operates on profile metadata only — extraction
    quality will be lower and more uncertainty flags will be raised.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def run(self, input_model: PaymentObligationInput) -> PaymentObligationOutput:
        logger.info("PaymentObligationAgent: starting — contract_id=%s", input_model.contract_id)
        try:
            prompt = self._build_prompt(input_model)
            raw = self.llm.generate_json(prompt)
            output = self._parse_response(raw, input_model.contract_id)
            logger.info(
                "PaymentObligationAgent: complete — %d obligations, flags=%s",
                len(output.obligations),
                output.uncertainty_flags,
            )
            return output
        except LLMError as exc:
            logger.error("PaymentObligationAgent: LLM failed — %s", exc)
            return self._fallback_output(input_model)

    def _build_prompt(self, input_model: PaymentObligationInput) -> str:
        profile = input_model.contract_profile
        parties_text = ", ".join(
            f"{p.name} ({p.role or 'party'})" for p in profile.parties
        ) or "unknown"

        if input_model.retrieved_evidence:
            evidence_lines = [
                f"[p.{e.page or '?'} {e.source}]: {e.text}"
                for e in input_model.retrieved_evidence
            ]
            contract_text = "\n\n".join(evidence_lines)
        else:
            contract_text = (
                "(No contract passages available — extraction based on profile metadata only. "
                "Results will have high uncertainty.)"
            )

        return f"""You are a contract payment analyst.

Extract ALL payment obligations from the contract passages below.

CONTRACT TYPE: {profile.contract_type.value}
PARTIES: {parties_text}
EFFECTIVE DATE: {profile.effective_date or 'unknown'}
COMMENCEMENT DATE: {profile.commencement_date or 'unknown'}
EXPIRATION DATE: {profile.expiration_date or 'unknown'}
SUMMARY: {profile.summary or 'N/A'}

CONTRACT PASSAGES:
{contract_text}

Extract every payment obligation including recurring fees, deposits, taxes, late fees,
penalties, termination fees, purchase options, renewals, and conditional payments.

For evidence: set page to the integer from the [p.X filename] label of the passage the quote comes from, or null if not from a retrieved passage. Set section to the clause or section name if identifiable (e.g. "Section 3", "Late Charges"), otherwise null.

For each obligation set:
- is_included_in_base_valuation: true for core predictable cash flows
- valuation_treatment: "include" for core flows, "exclude" for optional/unlikely, "conditional" for scenario-based

For recurring obligations, set date_range.start to the date of the first payment:
- Use the COMMENCEMENT DATE from the profile header above if it is provided (not "unknown") — this is the most reliable source.
- If COMMENCEMENT DATE is unknown, use the commencement date if one is explicitly stated or clearly identifiable from the passages (e.g. "Commencement Date is the first of the month following installation").
- Only fall back to the contract effective date if no distinct commencement date is available from either the profile or the passages.
- Add "missing_commencement_date" to uncertainty_flags when the commencement date is absent and the effective date is used as a fallback.

Add to uncertainty_flags when applicable:
- "missing_payment_dates" if due dates are absent
- "ambiguous_amount" if the amount is unclear or uses ranges
- "conditional_escalation" if the amount can increase under conditions
- "ambiguous_renewal_terms" if renewal obligations are unclear
- "missing_commencement_date" if the start date is missing

Return JSON:
{{
  "obligations": [
    {{
      "obligation_id": "p-001",
      "description": "string",
      "payment_type": "fixed|variable|milestone|recurring|one_time|contingent|penalty|deposit|other",
      "frequency": "monthly|quarterly|annually|semi_annually|weekly|daily|one_time|on_milestone|irregular",
      "amount": {{"amount": 0.0, "currency": "USD"}} or null,
      "amount_min": null,
      "amount_max": null,
      "due_date": "YYYY-MM-DD or null",
      "date_range": {{"start": "YYYY-MM-DD — first payment date: use COMMENCEMENT DATE from profile if provided, else commencement date from passages if identifiable, else contract effective date", "end": "YYYY-MM-DD or null"}} or null,
      "payer": "party name or null",
      "payee": "party name or null",
      "conditions": "conditions string or null",
      "evidence": {{"text": "exact quote", "page": integer_or_null, "section": "section_name or null"}} or null,
      "confidence": "high|medium|low",
      "is_included_in_base_valuation": true,
      "valuation_treatment": "include|exclude|conditional"
    }}
  ],
  "extraction_issues": [
    {{"issue_id": "i-001", "description": "string", "severity": "warning|error", "related_section": null}}
  ],
  "uncertainty_flags": [],
  "confidence": "high|medium|low"
}}

Return only valid JSON. No markdown, no explanation."""

    def _parse_response(self, raw: dict, contract_id: str) -> PaymentObligationOutput:
        obligations = []
        for i, ob in enumerate(raw.get("obligations", [])):
            try:
                obligations.append(_parse_obligation(ob, i))
            except Exception as exc:
                logger.warning("Skipping unparseable obligation %d: %s", i, exc)

        issues = [
            PaymentExtractionIssue(
                issue_id=iss.get("issue_id", f"i-{i:03d}"),
                description=iss.get("description", ""),
                severity=iss.get("severity", "warning"),
                related_section=iss.get("related_section"),
            )
            for i, iss in enumerate(raw.get("extraction_issues", []))
        ]

        return PaymentObligationOutput(
            contract_id=contract_id,
            obligations=obligations,
            extraction_issues=issues,
            uncertainty_flags=raw.get("uncertainty_flags", []),
            confidence=_safe_enum(ConfidenceLevel, raw.get("confidence"), ConfidenceLevel.MEDIUM),
        )

    def _fallback_output(self, input_model: PaymentObligationInput) -> PaymentObligationOutput:
        return PaymentObligationOutput(
            contract_id=input_model.contract_id,
            obligations=[],
            extraction_issues=[
                PaymentExtractionIssue(
                    issue_id="i-fallback",
                    description="LLM extraction unavailable; no obligations extracted.",
                    severity="error",
                )
            ],
            uncertainty_flags=["llm_extraction_failed"],
            confidence=ConfidenceLevel.LOW,
        )


# --- helpers ---

def _parse_obligation(ob: dict, index: int) -> PaymentObligation:
    amount = _parse_money(ob.get("amount"))
    amount_min = _parse_money(ob.get("amount_min"))
    amount_max = _parse_money(ob.get("amount_max"))
    date_range = _parse_date_range(ob.get("date_range"))
    evidence = _parse_evidence(ob.get("evidence"))

    return PaymentObligation(
        obligation_id=ob.get("obligation_id", f"p-{index + 1:03d}"),
        description=ob.get("description", "Unnamed obligation"),
        payment_type=_safe_enum(PaymentType, ob.get("payment_type"), PaymentType.OTHER),
        frequency=_safe_enum(PaymentFrequency, ob.get("frequency"), PaymentFrequency.ONE_TIME),
        amount=amount,
        amount_min=amount_min,
        amount_max=amount_max,
        due_date=_parse_date(ob.get("due_date")),
        date_range=date_range,
        payer=ob.get("payer"),
        payee=ob.get("payee"),
        conditions=ob.get("conditions"),
        evidence=evidence,
        confidence=_safe_enum(ConfidenceLevel, ob.get("confidence"), ConfidenceLevel.MEDIUM),
        is_included_in_base_valuation=bool(ob.get("is_included_in_base_valuation", True)),
        valuation_treatment=_safe_enum(
            ValuationTreatment, ob.get("valuation_treatment"), ValuationTreatment.INCLUDE
        ),
    )


def _parse_money(value) -> Optional[MoneyAmount]:
    if not value or not isinstance(value, dict):
        return None
    try:
        return MoneyAmount(
            amount=float(value["amount"]),
            currency=_safe_enum(Currency, value.get("currency", "USD"), Currency.USD),
        )
    except (KeyError, ValueError, TypeError):
        return None


def _parse_date_range(value) -> Optional[DateRange]:
    if not value or not isinstance(value, dict):
        return None
    start = _parse_date(value.get("start"))
    if start is None:
        return None
    return DateRange(start=start, end=_parse_date(value.get("end")))


def _parse_evidence(value) -> Optional[Evidence]:
    """Parse LLM-extracted evidence. source and chunk_id are set to sentinels
    since the LLM quotes from text directly rather than from indexed RAG chunks."""
    if not value or not isinstance(value, dict):
        return None
    text = value.get("text", "")
    if not text:
        return None
    return Evidence(
        text=text,
        source="llm_extracted",
        chunk_id="llm_extracted",
        section=value.get("section"),
        page=value.get("page"),
    )


def _parse_date(value) -> Optional[date]:
    if not value or str(value).lower() in ("null", "none", ""):
        return None
    try:
        return date.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def _safe_enum(enum_cls, value, default):
    try:
        return enum_cls(value)
    except (ValueError, KeyError, TypeError):
        return default
