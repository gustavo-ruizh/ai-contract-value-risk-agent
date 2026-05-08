# Evaluation Results

**Contracts evaluated:** 2  
**Evaluation date:** 2026-05-08

---

## Summary Metrics

| Metric | Value |
|---|---|
| Routing Decision Accuracy | 50.0% |
| Payment Obligation Precision | 17.4% |
| Payment Obligation Recall | 36.4% |
| Payment Obligation F1 | 23.5% |
| Blocking Risk Recall | 50.0% |

---

## Per-Contract Results

| Contract | Routing | Expected | Actual | Oblig. Precision | Oblig. Recall | Oblig. F1 | Blocking Risk Recall | Extraction Issues |
|---|---|---|---|---|---|---|---|---|
| contract_01 | ❌ | legal_review | clarification_required | 15.4% | 25.0% | 19.0% | 50.0% | 5 |
| contract_02 | ✅ | legal_review | legal_review | 20.0% | 66.7% | 30.8% | 50.0% | 5 |

---

## Methodology

**Obligation matching** uses two strategies in priority order:
1. Amount-based match (1% tolerance) when the ground truth obligation has a known amount.
2. Description keyword overlap (Jaccard ≥ 0.25) for obligations without explicit amounts.

**Blocking risk matching** uses keyword overlap (Jaccard ≥ 0.25) between expected risk descriptions and extracted blocking risk titles. Recall is prioritised — a missed blocking risk is a false negative with direct business impact.

**Routing accuracy** is an exact match between the human-annotated expected decision and the system output (`approve`, `finance_review`, `legal_review`, `clarification_required`).

> Ground truth was manually annotated by reviewing each contract and pipeline output. This evaluation covers a small sample and results are indicative, not statistically conclusive.