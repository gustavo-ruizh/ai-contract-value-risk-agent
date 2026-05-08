import logging
from datetime import date
from typing import List, Optional

from app.config import settings
from app.agents.contract_intake_agent import ContractIntakeAgent
from app.agents.contract_value_agent import ContractValueAgent
from app.agents.payment_obligation_agent import PaymentObligationAgent
from app.agents.review_decision_agent import ReviewDecisionAgent
from app.agents.risk_assessment_agent import RiskAssessmentAgent
from app.contract_rag.evidence_adapter import retrieval_results_to_evidence
from app.contract_rag.schemas import RetrievalQuery
from app.schemas.analysis import ContractAnalysisRequest, ContractAnalysisResult
from app.schemas.common import Evidence
from app.schemas.contract import ContractIntakeInput, ContractIntakeOutput
from app.schemas.payments import PaymentObligationInput
from app.schemas.review import ReviewDecisionInput
from app.schemas.risks import MinimalValuationContext, RiskAssessmentInput
from app.schemas.valuation import ContractValueInput
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ContractAnalysisOrchestrator:
    """
    Drives the five-agent linear pipeline:

        RAG evidence retrieval (optional, gracefully skipped)
        → ContractIntakeAgent
        → PaymentObligationAgent
        → ContractValueAgent
        → RiskAssessmentAgent
        → ReviewDecisionAgent

    Each agent receives a typed Pydantic input, returns a typed Pydantic output,
    and has no knowledge of the pipeline around it. The orchestrator is the only
    place that wires agent outputs to the next agent's inputs.

    All agent overrides accept None (defaults are constructed internally) to make
    the orchestrator testable without real LLM or OpenAI API keys.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        retriever=None,  # Optional[ContractRetriever] — typed loosely to avoid circular imports
        intake_agent: Optional[ContractIntakeAgent] = None,
        payment_agent: Optional[PaymentObligationAgent] = None,
        value_agent: Optional[ContractValueAgent] = None,
        risk_agent: Optional[RiskAssessmentAgent] = None,
        review_agent: Optional[ReviewDecisionAgent] = None,
    ) -> None:
        llm = llm_client or LLMClient()
        self._retriever = retriever
        self.intake_agent = intake_agent or ContractIntakeAgent(llm)
        self.payment_agent = payment_agent or PaymentObligationAgent(llm)
        self.value_agent = value_agent or ContractValueAgent()
        self.risk_agent = risk_agent or RiskAssessmentAgent(llm)
        self.review_agent = review_agent or ReviewDecisionAgent()

    def run(self, request: ContractAnalysisRequest) -> ContractAnalysisResult:
        logger.info("Orchestrator: starting — contract_id=%s", request.contract_id)

        # Step 1: Retrieve evidence from RAG (optional — skipped if no retriever or key)
        logger.info("Orchestrator: step 1 — RAG evidence retrieval")
        retriever = self._retriever or self._setup_rag(request.source_file, request.contract_id)
        retrieved_evidence: List[Evidence] = []
        if retriever is not None:
            try:
                query = RetrievalQuery(
                    query_text="payment obligations risks penalties termination fees",
                    contract_id=request.contract_id,
                    top_k=10,
                )
                results = retriever.retrieve(query)
                retrieved_evidence = retrieval_results_to_evidence(results)
                logger.info(
                    "Orchestrator: retrieved %d evidence items from RAG", len(retrieved_evidence)
                )
            except Exception as exc:
                logger.warning("Orchestrator: evidence retrieval failed — %s", exc)

        # Step 2: Contract Intake
        logger.info("Orchestrator: step 2 — ContractIntakeAgent")
        intake_input = ContractIntakeInput(
            contract_id=request.contract_id,
            source_file=request.source_file,
            retrieved_evidence=retrieved_evidence,
            optional_contract_type=request.optional_contract_type,
            analysis_goal=request.analysis_goal,
        )
        intake_output: ContractIntakeOutput = self.intake_agent.run(intake_input)

        # Step 3: Payment Obligation Extraction
        logger.info("Orchestrator: step 3 — PaymentObligationAgent")
        payment_input = PaymentObligationInput(
            contract_id=request.contract_id,
            contract_profile=intake_output.profile,
            retrieved_evidence=retrieved_evidence,
        )
        payment_output = self.payment_agent.run(payment_input)

        # Step 4: Contract Valuation (deterministic — no LLM)
        logger.info("Orchestrator: step 4 — ContractValueAgent")
        value_input = ContractValueInput(
            contract_id=request.contract_id,
            obligations=payment_output.obligations,
            valuation_date=(
                request.valuation_date
                or intake_output.profile.commencement_date
                or intake_output.profile.effective_date
                or date.today()
            ),
            discount_rate=(
                request.discount_rate
                if request.discount_rate is not None
                else settings.default_discount_rate
            ),
            uncertainty_flags=payment_output.uncertainty_flags,
        )
        value_output = self.value_agent.run(value_input)

        # Step 5: Build MinimalValuationContext from valuation assumptions + uncertainty flags.
        # Deliberately excludes monetary totals — the risk agent should reason about
        # *what was estimated*, not accept dollar amounts as ground truth.
        valuation_context = MinimalValuationContext(
            assumptions=value_output.assumptions,
            uncertainty_flags=value_output.uncertainty_flags,
        )

        # Step 6: Risk Assessment
        logger.info("Orchestrator: step 5 — RiskAssessmentAgent")
        risk_input = RiskAssessmentInput(
            contract_id=request.contract_id,
            contract_profile=intake_output.profile,
            obligations=payment_output.obligations,
            valuation_context=valuation_context,
            retrieved_evidence=retrieved_evidence,
        )
        risk_output = self.risk_agent.run(risk_input)

        # Step 7: Review Decision (deterministic — no LLM)
        logger.info("Orchestrator: step 6 — ReviewDecisionAgent")
        review_input = ReviewDecisionInput(
            contract_id=request.contract_id,
            contract_profile=intake_output.profile,
            risks=risk_output.risks,
            overall_risk_level=risk_output.overall_risk_level,
            valuation_uncertainty_flags=value_output.uncertainty_flags,
            risk_uncertainty_flags=risk_output.uncertainty_flags,
        )
        review_output = self.review_agent.run(review_input)

        result = ContractAnalysisResult(
            contract_id=request.contract_id,
            intake=intake_output,
            payment_obligations=payment_output,
            contract_value=value_output,
            risk_assessment=risk_output,
            review_decision=review_output,
        )
        logger.info(
            "Orchestrator: complete — decision=%s confidence=%s",
            review_output.decision,
            review_output.confidence,
        )
        return result

    def _setup_rag(self, source_file: str, contract_id: Optional[str]):
        """Load, chunk, embed, and index the contract file into an in-memory VectorStore.

        Returns a ContractRetriever or None if RAG setup is unavailable
        (no OpenAI key, openai package not installed, or any unexpected error).
        This is a graceful degradation — the pipeline continues without RAG evidence.
        """
        try:
            if not settings.openai_api_key:
                logger.info("Orchestrator: no OpenAI key — RAG evidence retrieval skipped")
                return None

            from app.contract_rag.chunker import ContractChunker
            from app.contract_rag.document_loader import DocumentLoader
            from app.contract_rag.embeddings import EmbeddingModel
            from app.contract_rag.retriever import ContractRetriever
            from app.contract_rag.vector_store import VectorStore

            pages = DocumentLoader().load(source_file)
            chunker = ContractChunker()
            chunks = chunker.chunk_pages(pages, contract_id or "unknown")

            embedding_model = EmbeddingModel(
                model=settings.default_embedding_model,
                api_key=settings.openai_api_key,
            )
            embedded_chunks = embedding_model.embed_batch(chunks)

            vector_store = VectorStore()
            vector_store.add(embedded_chunks)

            logger.info(
                "Orchestrator: RAG indexed %d chunks for contract_id=%s",
                len(chunks),
                contract_id,
            )
            return ContractRetriever(
                embedding_model=embedding_model,
                vector_store=vector_store,
            )

        except Exception as exc:
            logger.warning("Orchestrator: RAG setup failed — %s", exc)
            return None
