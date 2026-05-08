"""Tests for PVCalculator and advance_date."""
from datetime import date

import pytest

from app.services.pv_calculator import PVCalculator, advance_date


def test_present_value_zero_rate_returns_amount():
    calc = PVCalculator()
    assert calc.calculate_present_value(1000.0, 0.0, 12) == 1000.0


def test_present_value_zero_periods_returns_amount():
    calc = PVCalculator()
    assert calc.calculate_present_value(1000.0, 0.10, 0) == 1000.0


def test_present_value_negative_amount_returns_zero():
    calc = PVCalculator()
    assert calc.calculate_present_value(-500.0, 0.10, 6) == 0.0


def test_present_value_discounts_correctly():
    # PV = 1200 / (1 + 0.10/12)^12 ≈ 1086.25
    calc = PVCalculator()
    pv = calc.calculate_present_value(1200.0, 0.10, 12, 12)
    assert abs(pv - 1086.25) < 1.0


def test_periods_between_monthly():
    calc = PVCalculator()
    assert calc.periods_between(date(2024, 1, 1), date(2025, 1, 1), "monthly") == 12


def test_periods_between_annually():
    calc = PVCalculator()
    assert calc.periods_between(date(2024, 1, 1), date(2026, 1, 1), "annually") == 2


def test_periods_between_end_before_start_returns_zero():
    calc = PVCalculator()
    assert calc.periods_between(date(2025, 1, 1), date(2024, 1, 1), "monthly") == 0


def test_periods_per_year_monthly():
    calc = PVCalculator()
    assert calc.periods_per_year("monthly") == 12


def test_periods_per_year_quarterly():
    calc = PVCalculator()
    assert calc.periods_per_year("quarterly") == 4


def test_advance_date_monthly():
    assert advance_date(date(2024, 1, 31), 1, "monthly") == date(2024, 2, 29)


def test_advance_date_annually():
    assert advance_date(date(2024, 3, 15), 1, "annually") == date(2025, 3, 15)
