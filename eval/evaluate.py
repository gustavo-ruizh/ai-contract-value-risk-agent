#!/usr/bin/env python3
"""
eval/evaluate.py

Evaluates the AI Contract Value & Risk Agent against human-annotated ground truth.

Metrics:
  - Payment obligation extraction: precision, recall, F1
  - Review routing decision: accuracy
  - Blocking risk detection: recall

Usage:
    python eval/evaluate.py
"""

import json
from datetime import date
from pathlib import Path
from dataclasses import dataclass


GROUND_TRUTH_DIR = Path("eval/ground_truth")
OUTPUTS_DIR = Path("eval/outputs")
RESULTS_PATH = Path("eval/results.md")

AMOUNT_TOLERANCE = 0.01       # 1% tolerance for numeric amount matching
KEYWORD_OVERLAP_THRESHOLD = 0.25  # Jaccard similarity threshold for text matching


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def jaccard(text_a: str, text_b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def amounts_match(a: float, b: float) -> bool:
    """True if two amounts are within AMOUNT_TOLERANCE of each other."""
    if a == 0 and b == 0:
        return True
    if a == 0 or b == 0:
        return False
    return abs(a - b) / max(abs(a), abs(b)) <= AMOUNT_TOLERANCE


def fmt(value, as_pct: bool = True) -> str:
    """Format a metric value for display."""
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%" if as_pct else f"{value:.2f}"


# ---------------------------------------------------------------------------
# Obligation matching
# ---------------------------------------------------------------------------

@dataclass
class ObligationMatchResult:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self):
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else None

    @property
    def recall(self):
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else None

    @property
    def f1(self):
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)


def match_obligations(gt_obligations: list, extracted_obligations: list) -> ObligationMatchResult:
    """
    Match ground truth obligations to extracted obligations.

    Matching strategy (in priority order):
      1. If the ground truth obligation has a known amount:
         match by amount within AMOUNT_TOLERANCE.
      2. If no known amount (or amount match fails):
         match by description Jaccard similarity >= KEYWORD_OVERLAP_THRESHOLD.

    Each extracted obligation can be matched at most once.
    """
    result = ObligationMatchResult()
    matched_ids = set()

    for gt in gt_obligations:
        gt_amount = gt.get("amount")
        gt_desc = gt.get("description", "")
        matched = False

        for ext in extracted_obligations:
            ext_id = ext.get("obligation_id")
            if ext_id in matched_ids:
                continue

            # --- Amount-based match ---
            if gt_amount is not None:
                ext_amount_obj = ext.get("amount")
                if ext_amount_obj and ext_amount_obj.get("amount") is not None:
                    if amounts_match(gt_amount, ext_amount_obj["amount"]):
                        matched = True
                        matched_ids.add(ext_id)
                        break

            # --- Description-based match ---
            if not matched:
                ext_desc = ext.get("description", "")
                if jaccard(gt_desc, ext_desc) >= KEYWORD_OVERLAP_THRESHOLD:
                    matched = True
                    matched_ids.add(ext_id)
                    break

        if matched:
            result.true_positives += 1
        else:
            result.false_negatives += 1

    result.false_positives = len(extracted_obligations) - len(matched_ids)
    return result


# ---------------------------------------------------------------------------
# Blocking risk matching
# ---------------------------------------------------------------------------

def match_blocking_risks(expected: list[str], extracted_risks: list) -> dict:
    """
    For each expected blocking risk description, check whether any extracted
    risk with is_blocking=True matches by Jaccard similarity.

    Returns recall — the primary metric of interest here.
    A missed blocking risk is a false negative with direct business impact.
    """
    extracted_blocking = [
        r.get("title", "") for r in extracted_risks if r.get("is_blocking", False)
    ]

    matched = 0
    for exp_desc in expected:
        for ext_title in extracted_blocking:
            if jaccard(exp_desc, ext_title) >= KEYWORD_OVERLAP_THRESHOLD:
                matched += 1
                break

    total = len(expected)
    return {
        "expected": total,
        "detected": matched,
        "recall": matched / total if total > 0 else None,
        "extracted_blocking_count": len(extracted_blocking),
    }


# ---------------------------------------------------------------------------
# Per-contract evaluation
# ---------------------------------------------------------------------------

@dataclass
class ContractEvalResult:
    contract_id: str
    routing_correct: bool
    expected_decision: str
    actual_decision: str
    obligation_match: ObligationMatchResult
    blocking_risk: dict
    extraction_issues_count: int
    uncertainty_flags: list


def evaluate_contract(gt_path: Path, output_path: Path) -> ContractEvalResult:
    gt = load_json(gt_path)
    output = load_json(output_path)

    # Routing decision
    expected_decision = gt["expected_routing_decision"]
    actual_decision = output.get("review_decision", {}).get("decision", "unknown")

    # Payment obligation extraction
    gt_obligations = gt.get("expected_payment_obligations", [])
    extracted_obligations = output.get("payment_obligations", {}).get("obligations", [])
    obligation_match = match_obligations(gt_obligations, extracted_obligations)

    # Blocking risk detection
    expected_blocking = gt.get("expected_blocking_risks", [])
    extracted_risks = output.get("risk_assessment", {}).get("risks", [])
    blocking_risk = match_blocking_risks(expected_blocking, extracted_risks)

    # Extraction health signals
    extraction_issues = output.get("payment_obligations", {}).get("extraction_issues", [])
    uncertainty_flags = output.get("payment_obligations", {}).get("uncertainty_flags", [])

    return ContractEvalResult(
        contract_id=gt["contract_id"],
        routing_correct=(expected_decision == actual_decision),
        expected_decision=expected_decision,
        actual_decision=actual_decision,
        obligation_match=obligation_match,
        blocking_risk=blocking_risk,
        extraction_issues_count=len(extraction_issues),
        uncertainty_flags=uncertainty_flags,
    )


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate(results: list[ContractEvalResult]) -> dict:
    routing_accuracy = sum(r.routing_correct for r in results) / len(results)

    # Micro-average across contracts for obligations
    tp = sum(r.obligation_match.true_positives for r in results)
    fp = sum(r.obligation_match.false_positives for r in results)
    fn = sum(r.obligation_match.false_negatives for r in results)
    micro_p = tp / (tp + fp) if (tp + fp) > 0 else None
    micro_r = tp / (tp + fn) if (tp + fn) > 0 else None
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r)
        if micro_p and micro_r else None
    )

    # Blocking risk recall across all contracts
    total_expected = sum(r.blocking_risk["expected"] for r in results)
    total_detected = sum(r.blocking_risk["detected"] for r in results)
    blocking_recall = total_detected / total_expected if total_expected > 0 else None

    return {
        "contracts_evaluated": len(results),
        "routing_accuracy": routing_accuracy,
        "obligation_precision": micro_p,
        "obligation_recall": micro_r,
        "obligation_f1": micro_f1,
        "blocking_risk_recall": blocking_recall,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def build_report(results: list[ContractEvalResult], agg: dict) -> str:
    lines = [
        "# Evaluation Results\n",
        f"**Contracts evaluated:** {agg['contracts_evaluated']}  ",
        f"**Evaluation date:** {date.today()}\n",
        "---\n",
        "## Summary Metrics\n",
        "| Metric | Value |",
        "|---|---|",
        f"| Routing Decision Accuracy | {fmt(agg['routing_accuracy'])} |",
        f"| Payment Obligation Precision | {fmt(agg['obligation_precision'])} |",
        f"| Payment Obligation Recall | {fmt(agg['obligation_recall'])} |",
        f"| Payment Obligation F1 | {fmt(agg['obligation_f1'])} |",
        f"| Blocking Risk Recall | {fmt(agg['blocking_risk_recall'])} |",
        "\n---\n",
        "## Per-Contract Results\n",
        "| Contract | Routing | Expected | Actual | Oblig. Precision | Oblig. Recall | Oblig. F1 | Blocking Risk Recall | Extraction Issues |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        routing_icon = "✅" if r.routing_correct else "❌"
        lines.append(
            f"| {r.contract_id} "
            f"| {routing_icon} "
            f"| {r.expected_decision} "
            f"| {r.actual_decision} "
            f"| {fmt(r.obligation_match.precision)} "
            f"| {fmt(r.obligation_match.recall)} "
            f"| {fmt(r.obligation_match.f1)} "
            f"| {fmt(r.blocking_risk['recall'])} "
            f"| {r.extraction_issues_count} |"
        )

    lines += [
        "\n---\n",
        "## Methodology\n",
        "**Obligation matching** uses two strategies in priority order:",
        "1. Amount-based match (1% tolerance) when the ground truth obligation has a known amount.",
        "2. Description keyword overlap (Jaccard ≥ 0.25) for obligations without explicit amounts.",
        "",
        "**Blocking risk matching** uses keyword overlap (Jaccard ≥ 0.25) between expected risk "
        "descriptions and extracted blocking risk titles. Recall is prioritised — a missed blocking "
        "risk is a false negative with direct business impact.",
        "",
        "**Routing accuracy** is an exact match between the human-annotated expected decision and "
        "the system output (`approve`, `finance_review`, `legal_review`, `clarification_required`).",
        "",
        "> Ground truth was manually annotated by reviewing each contract and pipeline output. "
        "This evaluation covers a small sample and results are indicative, not statistically conclusive.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    gt_files = sorted(GROUND_TRUTH_DIR.glob("*.json"))
    if not gt_files:
        print(f"No ground truth files found in {GROUND_TRUTH_DIR}/")
        return

    results = []
    for gt_path in gt_files:
        output_path = OUTPUTS_DIR / gt_path.name
        if not output_path.exists():
            print(f"⚠️  No pipeline output found for {gt_path.name} — skipping.")
            continue

        result = evaluate_contract(gt_path, output_path)
        results.append(result)
        print(f"✅  Evaluated {result.contract_id}")

    if not results:
        print("No contracts evaluated. Check that ground_truth/ and outputs/ filenames match.")
        return

    agg = aggregate(results)

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Contracts evaluated:       {agg['contracts_evaluated']}")
    print(f"Routing accuracy:          {fmt(agg['routing_accuracy'])}")
    print(f"Obligation precision:      {fmt(agg['obligation_precision'])}")
    print(f"Obligation recall:         {fmt(agg['obligation_recall'])}")
    print(f"Obligation F1:             {fmt(agg['obligation_f1'])}")
    print(f"Blocking risk recall:      {fmt(agg['blocking_risk_recall'])}")

    report = build_report(results, agg)
    RESULTS_PATH.write_text(report)
    print(f"\n📄 Report saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
