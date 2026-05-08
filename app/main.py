import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s: %(message)s",
)

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.config import settings
from app.orchestrator.contract_analysis_orchestrator import ContractAnalysisOrchestrator
from app.schemas.analysis import ContractAnalysisRequest, ContractAnalysisResult
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

class _ASCIIResponse(JSONResponse):
    """Serialize all responses with ensure_ascii=True so non-ASCII characters
    (em dashes, curly quotes, etc.) are output as \\uXXXX escape sequences.
    This eliminates encoding ambiguity for any client reading the JSON."""

    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=True, allow_nan=False).encode("utf-8")


app = FastAPI(
    title=settings.app_name,
    description="Local-first AI agent that analyzes contracts for value and risk.",
    version="0.1.0",
    default_response_class=_ASCIIResponse,
)

_orchestrator: ContractAnalysisOrchestrator | None = None


def _get_orchestrator() -> ContractAnalysisOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        llm = LLMClient(
            model=settings.default_llm_model,
            api_key=settings.anthropic_api_key,
        )
        _orchestrator = ContractAnalysisOrchestrator(llm_client=llm)
    return _orchestrator


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "app": settings.app_name}


@app.post("/analyze", response_model=ContractAnalysisResult)
def analyze(request: ContractAnalysisRequest) -> ContractAnalysisResult:
    try:
        orchestrator = _get_orchestrator()
        return orchestrator.run(request)
    except Exception as exc:
        logger.error("Analysis failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
