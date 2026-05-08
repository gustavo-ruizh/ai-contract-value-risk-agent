"""Tests for ContractValueAgent."""
from datetime import date

from app.agents.contract_value_agent import ContractValueAgent
from app.schemas.common import ConfidenceLevel, Currency, MoneyAmount
from app.schemas.payments import PaymentFrequency, PaymentObligation, PaymentType, ValuationTreatment
from app.schemas.valuation import ContractValueInput, ContractValueOutput


def _one_time_obligation(obligation_id="p-001", amount=10_000.0, due_date=None,
                          treatment=ValuationTreatment.INCLUDE):
    return PaymentObligation(
        obligation_id=obligation_id,
        description="Test obligation",
        payment_type=PaymentType.ONE_TIME,
        frequency=PaymentFrequency.ONE_TIME,
        amount=MoneyAmount(amount=amount, currency=Currency.USD),
        confidence=ConfidenceLevel.HIGH,
        is_included_in_base_valuation=True,
        valuation_treatment=treatment,
        due_date=due_date,
    )


def test_one_time_obligation_nominal_and_pv():
    agent = ContractValueAgent()
    ob = _one_time_obligation(amount=12_000.0, due_date=date(2026, 1, 1))
    inp = ContractValueInput(
        obligations=[ob],
        valuation_date=date(2025, 1, 1),
        discount_rate=0.10,
    )
    out: ContractValueOutput = agent.run(inp)
    assert out.total_nominal_value.amount == 12_000.0
    # PV must be less than nominal since payment is in the future
    assert out.total_present_value.amount < 12_000.0
    assert len(out.payment_schedule) == 1


def test_excluded_obligation_not_in_schedule():
    agent = ContractValueAgent()
    ob = _one_time_obligation(treatment=ValuationTreatment.EXCLUDE)
    inp = ContractValueInput(
        obligations=[ob],
        valuation_date=date(2025, 1, 1),
        discount_rate=0.10,
    )
    out = agent.run(inp)
    assert out.total_nominal_value.amount == 0.0
    assert len(out.payment_schedule) == 0


def test_missing_amount_produces_warning():
    agent = ContractValueAgent()
    ob = PaymentObligation(
        obligation_id="p-001",
        description="No amount obligation",
        payment_type=PaymentType.OTHER,
        frequency=PaymentFrequency.ONE_TIME,
        confidence=ConfidenceLevel.LOW,
        is_included_in_base_valuation=True,
        valuation_treatment=ValuationTreatment.INCLUDE,
    )
    inp = ContractValueInput(
        obligations=[ob],
        valuation_date=date(2025, 1, 1),
        discount_rate=0.10,
    )
    out = agent.run(inp)
    assert out.total_nominal_value.amount == 0.0
    assert len(out.warnings) == 1
    assert "p-001" in out.warnings[0].warning_id


def test_amount_range_uses_midpoint():
    agent = ContractValueAgent()
    ob = PaymentObligation(
        obligation_id="p-002",
        description="Range obligation",
        payment_type=PaymentType.VARIABLE,
        frequency=PaymentFrequency.ONE_TIME,
        amount_min=MoneyAmount(amount=8_000.0, currency=Currency.USD),
        amount_max=MoneyAmount(amount=12_000.0, currency=Currency.USD),
        confidence=ConfidenceLevel.MEDIUM,
        is_included_in_base_valuation=True,
        valuation_treatment=ValuationTreatment.INCLUDE,
    )
    inp = ContractValueInput(
        obligations=[ob],
        valuation_date=date(2025, 1, 1),
        discount_rate=0.0,
    )
    out = agent.run(inp)
    # Midpoint of 8000–12000 = 10000; rate=0 so nominal == PV
    assert out.total_nominal_value.amount == 10_000.0
    assert len(out.assumptions) >= 1


def test_conditional_obligation_adds_assumption_and_flag():
    agent = ContractValueAgent()
    ob = _one_time_obligation(treatment=ValuationTreatment.CONDITIONAL, amount=5_000.0)
    inp = ContractValueInput(
        obligations=[ob],
        valuation_date=date(2025, 1, 1),
        discount_rate=0.0,
    )
    out = agent.run(inp)
    assert out.total_nominal_value.amount == 5_000.0
    assert any("conditional_obligation" in f for f in out.uncertainty_flags)
    assert len(out.assumptions) >= 1


def test_confidence_degrades_with_uncertainty_flags():
    agent = ContractValueAgent()
    ob = _one_time_obligation()
    inp = ContractValueInput(
        obligations=[ob],
        valuation_date=date(2025, 1, 1),
        discount_rate=0.0,
        uncertainty_flags=["flag1", "flag2", "flag3", "flag4"],
    )
    out = agent.run(inp)
    assert out.confidence == ConfidenceLevel.LOW
