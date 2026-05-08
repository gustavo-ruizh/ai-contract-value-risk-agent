import logging
import re
from datetime import date
from typing import List, Optional

from app.schemas.common import ConfidenceLevel, Evidence
from app.schemas.contract import (
    AnalyzabilityStatus,
    ContractIntakeInput,
    ContractIntakeOutput,
    ContractProfile,
    ContractSection,
    ContractType,
    Party,
)
from app.services.llm_client import LLMClient, LLMError

logger = logging.getLogger(__name__)


class ContractIntakeAgent:
    """
    Parses a local contract file and produces a structured profile:
    contract type, parties, key dates, sections, and analyzability status.
    Falls back to a low-confidence stub if the LLM call or file load fails.

    Always reads the full document directly — independent of the RAG pipeline.
    The LLM's context window (200K tokens for Claude Sonnet) is the only
    effective length limit.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def run(self, input_model: ContractIntakeInput) -> ContractIntakeOutput:
        logger.info("ContractIntakeAgent: starting — contract_id=%s", input_model.contract_id)
        try:
            prompt = self._build_prompt(input_model)
            raw = self.llm.generate_json(prompt)
            output = self._parse_response(raw, input_model.contract_id)
            logger.info(
                "ContractIntakeAgent: complete — type=%s analyzability=%s",
                output.profile.contract_type,
                output.analyzability,
            )
            return output
        except LLMError as exc:
            logger.error("ContractIntakeAgent: LLM failed — %s", exc)
            return self._fallback_output(input_model)
        except Exception as exc:
            logger.error("ContractIntakeAgent: unexpected error — %s", exc)
            return self._fallback_output(input_model)

    def _build_prompt(self, input_model: ContractIntakeInput) -> str:
        # Always read the full document — intake needs a holistic view of the entire
        # contract and must not be constrained by the RAG retrieval query or top-k limit.
        contract_text = self._load_text(input_model.source_file)

        hint_lines = []
        if input_model.optional_contract_type:
            hint_lines.append(f"Expected contract type: {input_model.optional_contract_type.value}")
        if input_model.analysis_goal:
            hint_lines.append(f"Analysis goal: {input_model.analysis_goal}")
        hint_section = ("\nCALLER HINTS:\n" + "\n".join(hint_lines)) if hint_lines else ""

        return f"""Analyze the following contract text and extract its key attributes.
{hint_section}

CONTRACT TEXT:
{contract_text}

Return a JSON object with this exact structure (use null for missing values):
{{
  "contract_type": "service_agreement|purchase_order|lease|license|employment|nda|partnership|other|unknown",
  "parties": [{{"name": "string", "role": "buyer|seller|licensor|licensee|landlord|tenant|employer|employee|client|vendor|other"}}],
  "effective_date": "YYYY-MM-DD or null",
  "commencement_date": "YYYY-MM-DD — the date on which performance or occupancy begins (often distinct from effective_date); populate if explicitly stated or clearly derivable (e.g. 'first of the month following installation'); null if not identifiable",
  "expiration_date": "YYYY-MM-DD — if not explicitly stated but derivable from a stated term length plus commencement_date (preferred) or effective_date, calculate and populate it; otherwise null",
  "governing_law": "jurisdiction string or null",
  "summary": "one sentence plain-English summary",
  "confidence": "high|medium|low",
  "analyzability": "fully_analyzable|partially_analyzable|not_analyzable",
  "analyzability_notes": "reason if not fully analyzable, or note if expiration_date was derived rather than explicitly stated; else null",
  "sections": [
    {{
      "section_id": "s1",
      "title": "section heading or null",
      "content": "one sentence summary of this section",
      "page": integer_from_nearest_PAGE_marker_or_null
    }}
  ]
}}

Identify all major sections (payment terms, termination, liability, renewal, penalties, etc.).
For each section set page to the integer from the nearest preceding [PAGE N] marker in the contract text.
For each section write a one-sentence summary in the content field — do not reproduce the original text.
Never include [PAGE N] markers in any extracted text fields.
If expiration_date is not explicitly stated but a term length is mentioned (e.g. "24-month lease"), calculate expiration_date as commencement_date (preferred) or effective_date plus that term. Record this inference in analyzability_notes.
Return only valid JSON. No markdown, no explanation."""

    def _load_text(self, source_file: str) -> str:
        """Load full contract text with [PAGE N] boundary markers for page attribution."""
        from app.contract_rag.document_loader import DocumentLoader
        loader = DocumentLoader()
        pages = loader.load(source_file)
        return "\n\n".join(f"[PAGE {p.page_number}]\n{p.text}" for p in pages)

    def _parse_response(self, raw: dict, contract_id: str) -> ContractIntakeOutput:
        parties = [
            Party(name=p.get("name", "Unknown"), role=p.get("role"))
            for p in raw.get("parties", [])
        ]

        profile = ContractProfile(
            contract_type=_safe_enum(ContractType, raw.get("contract_type"), ContractType.UNKNOWN),
            parties=parties,
            effective_date=_parse_date(raw.get("effective_date")),
            commencement_date=_parse_date(raw.get("commencement_date")),
            expiration_date=_parse_date(raw.get("expiration_date")),
            governing_law=raw.get("governing_law"),
            summary=raw.get("summary"),
            confidence=_safe_enum(ConfidenceLevel, raw.get("confidence"), ConfidenceLevel.MEDIUM),
        )

        sections = [
            ContractSection(
                section_id=s.get("section_id", f"s{i + 1}"),
                title=_scrub_page_markers(s.get("title")),
                content=_scrub_page_markers(s.get("content", "")),
                page=s.get("page"),
            )
            for i, s in enumerate(raw.get("sections", []))
        ]

        return ContractIntakeOutput(
            contract_id=contract_id,
            profile=profile,
            sections=sections,
            analyzability=_safe_enum(
                AnalyzabilityStatus,
                raw.get("analyzability"),
                AnalyzabilityStatus.PARTIALLY_ANALYZABLE,
            ),
            analyzability_notes=raw.get("analyzability_notes"),
            confidence=_safe_enum(ConfidenceLevel, raw.get("confidence"), ConfidenceLevel.MEDIUM),
        )

    def _fallback_output(self, input_model: ContractIntakeInput) -> ContractIntakeOutput:
        """Return a minimal output when file load or LLM call fails."""
        profile = ContractProfile(
            contract_type=_safe_enum(
                ContractType,
                input_model.optional_contract_type.value if input_model.optional_contract_type else None,
                ContractType.UNKNOWN,
            ),
            parties=[],
            confidence=ConfidenceLevel.LOW,
        )
        return ContractIntakeOutput(
            contract_id=input_model.contract_id,
            profile=profile,
            sections=[],
            analyzability=AnalyzabilityStatus.NOT_ANALYZABLE,
            analyzability_notes="LLM analysis unavailable; manual review required.",
            confidence=ConfidenceLevel.LOW,
        )


# --- helpers ---

_PAGE_MARKER_RE = re.compile(r"\[PAGE \d+\]\s*", re.IGNORECASE)


def _scrub_page_markers(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    return _PAGE_MARKER_RE.sub("", value).strip() or None


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
