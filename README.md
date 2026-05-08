# AI Contract Value & Risk Intelligence Agent

Evidence-grounded AI contract analysis pipeline combining LLM extraction, deterministic valuation, uncertainty handling, and rule-based review routing.

---

# Project Overview

Enterprise contracts often contain recurring obligations, escalation clauses, penalties, renewal terms, and ambiguous financial conditions that are difficult to review manually.

This system analyzes a contract (plain text or PDF) and produces a structured decision-oriented output in a single API call:

- **Payment obligations** — recurring fees, deposits, penalties, conditional payments, termination costs
- **Contract valuation** — structured payment schedule with nominal and discounted present value
- **Risk assessment** — financial, legal, operational, compliance, and counterparty risks with blocking flags and priority scores
- **Review decision** — deterministic routing to:
  - `approve`
  - `finance_review`
  - `legal_review`
  - `clarification_required`

The system is designed to be explainable, strongly typed, deterministic where appropriate, and easy to reason about in a technical review. Every design choice intentionally favors inspectability over capability.

---

# Architecture

The system uses a **controlled linear agentic pipeline**.

There is intentionally:
- no orchestration framework
- no routing agent
- no planner
- no memory system
- no autonomous loop

```text
RAG Evidence Retrieval (optional, graceful skip)
    ↓
ContractIntakeAgent          (LLM)
    ↓
PaymentObligationAgent       (LLM)
    ↓
ContractValueAgent           (deterministic)
    ↓
MinimalValuationContext      (assumptions + uncertainty flags only)
    ↓
RiskAssessmentAgent          (LLM + retrieved evidence)
    ↓
ReviewDecisionAgent          (deterministic)
```

Each agent:
- accepts exactly one typed Pydantic input model
- returns exactly one typed Pydantic output model
- has no awareness of upstream/downstream stages
- is instantiated and wired together only in `ContractAnalysisOrchestrator`

The orchestrator contains no business logic. It connects outputs to inputs, logs each stage, and returns the assembled `ContractAnalysisResult`.

---

# Why This Architecture Is Intentionally Simple

This project intentionally avoids planners, routing agents, memory systems, and orchestration frameworks (LangChain, LlamaIndex, etc.) for three reasons:

## 1. Explainability

Financial and legal analysis must be auditable.

A linear pipeline with typed I/O makes it easy to identify:
- which agent produced an output
- which evidence supported it
- where uncertainty entered the system

## 2. Testability

Each agent can be tested independently.

The orchestrator can be tested with mocked agents.

There is no shared mutable state, hidden routing, or implicit behavior.

## 3. Reliability

LLM usage is isolated to:
- contract intake
- payment extraction
- risk assessment

Valuation and review routing are deterministic and produce identical outputs for identical inputs.

---

# Five-Agent Pipeline

| Agent | Input | Output | LLM? |
|---|---|---|---|
| `ContractIntakeAgent` | `ContractIntakeInput` | `ContractIntakeOutput` | Yes |
| `PaymentObligationAgent` | `PaymentObligationInput` | `PaymentObligationOutput` | Yes |
| `ContractValueAgent` | `ContractValueInput` | `ContractValueOutput` | No — deterministic |
| `RiskAssessmentAgent` | `RiskAssessmentInput` | `RiskAssessmentOutput` | Yes |
| `ReviewDecisionAgent` | `ReviewDecisionInput` | `ReviewDecisionOutput` | No — deterministic |

## LLM Agents (1, 2, 4)

Each LLM agent:
- builds a structured prompt
- calls `LLMClient.generate_json()`
- parses typed JSON outputs
- falls back to low-confidence stub outputs if the LLM call fails

This fallback behavior ensures the pipeline always completes.

Downstream agents receive:
- empty obligations
- empty risks
- degraded confidence

rather than an exception.

## Deterministic Agents (3, 5)

### ContractValueAgent

The valuation stage:
- iterates through extracted obligations
- resolves exact and estimated amounts
- builds payment schedule lines
- calculates nominal and present value using discrete discounting
- records all valuation assumptions explicitly

Discounting uses:

```text
PV = FV / (1 + r/n)^t
```

The discount rate defaults to 3% (configurable) and can be overridden per request.

### ReviewDecisionAgent

The review stage:
- sorts risks deterministically
- applies explicit routing rules
- produces one of four outcomes:
  - `approve`
  - `finance_review`
  - `legal_review`
  - `clarification_required`

It also generates follow-up questions for blocking or high-severity risks.

---

# RAG Evidence Grounding

The `contract_rag/` module provides:
- document loading
- chunking
- embeddings
- vector storage
- retrieval
- evidence grounding

RAG retrieval occurs before the agent pipeline begins.

Retrieved evidence is converted into canonical `Evidence` objects and passed into downstream stages.

## Document Access Strategy

| Agent | Document access |
|---|---|
| `ContractIntakeAgent` | Reads the full contract directly |
| `PaymentObligationAgent` | Uses RAG-retrieved payment-related chunks |
| `RiskAssessmentAgent` | Uses RAG-retrieved risk-related chunks |

When RAG is unavailable (for example, no OpenAI key configured), the pipeline degrades gracefully:
- payment and risk agents receive no retrieved evidence
- confidence decreases
- the pipeline still completes successfully

---

# My RAG Implementation

The RAG module was intentionally implemented from first principles without LangChain, LlamaIndex, or an external vector database.

| Component | Implementation |
|---|---|
| `DocumentLoader` | Loads `.txt` and `.pdf` files with OCR fallback using `ocrmypdf` |
| `ContractChunker` | Word-based sliding window chunking with overlap and metadata preservation |
| `EmbeddingModel` | OpenAI embeddings wrapper (`text-embedding-3-small`) |
| `VectorStore` | In-memory cosine similarity search using NumPy |
| `ContractRetriever` | Query embedding + top-k retrieval coordinator |
| `EvidenceAdapter` | Converts retrieval results into canonical `Evidence` objects |

Every retrieved chunk preserves:
- source filename
- page number
- chunk ID
- relevance score
- original text

The goal of this implementation was not to build the most advanced RAG system possible.

Instead, the goal was to build a:
- readable
- explainable
- metadata-preserving
- interview-friendly

retrieval pipeline that clearly demonstrates grounding mechanics.

---

# Uncertainty Handling

Uncertainty is surfaced explicitly throughout the pipeline rather than silently absorbed.

| Stage | Mechanism |
|---|---|
| Payment extraction | `uncertainty_flags` |
| Valuation | `ValuationAssumption` objects |
| Valuation | confidence degradation based on unresolved uncertainty |
| Risk assessment | `RiskItem.is_blocking` |
| Risk assessment | deterministic `risk_priority_score` |
| Review decision | `decision_uncertainty_flags` |
| Review decision | follow-up questions for blocking risks |

`MinimalValuationContext` contains only:
- assumptions
- uncertainty flags

and intentionally excludes monetary totals.

This prevents the risk agent from treating estimated valuations as facts.

---

# Design Trade-offs

| Decision | Rationale |
|---|---|
| Linear pipeline | Auditable data flow; easier to test and explain |
| Deterministic valuation | LLMs should not perform financial arithmetic |
| Deterministic review routing | Review decisions should remain inspectable |
| `MinimalValuationContext` | Decouples risk reasoning from uncertain valuations |
| In-memory vector store | Sufficient for single-contract analysis; no infrastructure required |
| Lazy imports | Allows local execution without optional heavy dependencies |
| Fallback outputs | Pipeline completes even when LLM calls fail |
| Simple chunking | Readable and explainable over highly optimized retrieval complexity |

---

# AI-Assisted Development Workflow

This project was developed using a structured AI-assisted workflow designed to preserve architectural clarity, implementation control, and production-oriented decision-making.

The development process was intentionally divided into three stages.

## 1. System Design & Architecture

A dedicated design workspace was used to:
- define the five-agent architecture
- establish strict Pydantic data contracts
- refine uncertainty propagation and evidence grounding
- enforce deterministic orchestration rules
- prevent unnecessary complexity and over-engineering

The focus of this stage was system design, explainability, and enterprise-oriented architecture.

## 2. Incremental Implementation

Implementation support was provided through Claude Code using tightly scoped prompts and explicit architectural guardrails.

Development was intentionally separated into:
- project scaffolding
- RAG pipeline implementation
- agent/system implementation
- iterative review and cleanup

This approach preserved architectural consistency while accelerating implementation.

## 3. Review & Packaging

A separate review process was used to:
- audit architecture fidelity
- identify over-engineering risks
- strengthen evidence grounding
- improve schema consistency
- refine portfolio presentation and documentation

## Personally Implemented Components

I personally implemented the `contract_rag/` module, including:
- document loading
- chunking
- metadata preservation
- embeddings integration
- vector retrieval
- evidence grounding

This portion of the project was intentionally developed hands-on to deepen practical understanding of:
- retrieval pipelines
- vector search workflows
- metadata-aware chunking
- evidence traceability
- grounding mechanisms for downstream agent reasoning

The final system reflects a hybrid workflow combining:
- human architectural judgment
- controlled AI-assisted implementation
- iterative review and refinement rather than autonomous code generation.

---

# Current Limitations

- Multi-currency contracts are not handled correctly.
- The vector store is in-memory only and rebuilt per request.
- Retrieval is limited to a single contract at a time.
- The API currently accepts only local filesystem paths.
- The intake agent is bounded by the LLM context window.
- The API is synchronous and non-streaming.
- The project is portfolio-oriented and not production-ready:
  - no authentication
  - no persistence
  - no rate limiting
  - no observability stack

---

# Project Structure

```text
ai-contract-value-risk-agent/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── agents/
│   ├── orchestrator/
│   ├── contract_rag/
│   ├── schemas/
│   ├── services/
│   └── tests/
├── sample_data/
└── requirements.txt
```

Key implementation areas:
- `agents/` — five-stage analysis pipeline
- `contract_rag/` — retrieval and evidence grounding
- `schemas/` — strict typed Pydantic contracts
- `services/` — valuation and LLM wrappers
- `tests/` — focused validation and orchestration tests

---

# How to Run Locally

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
# ANTHROPIC_API_KEY=...
# OPENAI_API_KEY=... (optional)
# DEFAULT_DISCOUNT_RATE=0.03

# Start API
uvicorn app.main:app --reload

# Run tests
pytest app/tests/
```

---

# Example Request

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "contract_id": "acme-techco-2025",
    "source_file": "sample_data/sample_contract.txt"
  }'
```

Optional request fields:
- `discount_rate`
- `valuation_date`
- `optional_contract_type`
- `analysis_goal`

---

# Real Example Response (abridged)

```json
{
  "contract_value": {
    "contract_id": "equipment-lease-016",
    "total_nominal_value": {
      "amount": 4225771.2,
      "currency": "USD"
    },
    "total_present_value": {
      "amount": 4106770.74,
      "currency": "USD"
    }
  },
  "risk_assessment": {
    "contract_id": "demo-016",
    "risks": [
      {
        "risk_id": "r-001",
        "title": "Missing Payment Due Dates for Base Monthly Charge",
        "description": "The base monthly lease charge of $176,073.80 (obligation p-001) lacks explicit due dates. While the date range is 2022-10-01 to 2024-09-30, the specific day-of-month payment is due is not stated, creating ambiguity in cash flow timing, late charge trigger points, and default determination.",
        "category": "financial",
        "severity": "high",
        "evidence": {
          "text": "to be paid by Lessee which non-payment continues for a period of ten (10) days from the date when due",
          "source": "llm_extracted",
          "chunk_id": "llm_extracted",
          "page": 5,
          "relevance_score": null,
          "section": "Event of Default"
        },
        "mitigation": "Confirm specific monthly due date from full lease schedule; amend contract or obtain written confirmation of payment date.",
        "is_blocking": true,
        "risk_priority_score": 130
      }
    ]
  },
  "review_decision": {
    "contract_id": "demo-016",
    "decision": "legal_review",
    "rationale": "Contract has 4 blocking risk(s) including legal or compliance issues. Legal review is required before proceeding.",
    "follow_up_questions": [
      {
        "question_id": "q-001",
        "question": "Regarding risk 'Casualty Loss Value Undefined – Unquantified Default Exposure': This is a blocking issue — Upon an Event of Default, the Casualty Loss Value of Equipment becomes immediately due and payable (obligation p-006), but this value is not defined or quantified in the retrieved passages. Given total equipment commitment up to $19,200,000, this could represent the single largest contingent liability, yet it cannot be reliably valued. How should this be resolved before approval?",
        "target_audience": "finance",
        "related_risk_id": "r-005"
      }
    ],
    "decision_factors": [
      "4 blocking risk(s) with legal/compliance exposure"
    ],
    "confidence": "medium"
  }
}
```

---

# Implementation Status

Implemented components include:

- Five-agent contract analysis pipeline
- Deterministic valuation and review routing
- RAG evidence retrieval and grounding
- Typed Pydantic schemas across all stages
- FastAPI API layer
- Anthropic-based LLM integration
- Focused test suite for orchestration, valuation, and evidence handling

Test suite currently passes locally.

---

# Running Tests

```bash
pytest app/tests/ -v
```

Tests do not require API keys.
LLM and embedding calls are mocked where appropriate.

