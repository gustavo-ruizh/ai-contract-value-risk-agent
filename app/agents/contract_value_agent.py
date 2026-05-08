import logging
from datetime import date
from typing import List, Optional, Tuple

from app.schemas.common import ConfidenceLevel, Currency, MoneyAmount
from app.schemas.payments import PaymentFrequency, PaymentObligation, ValuationTreatment
from app.schemas.valuation import (
    CalculationWarning,
    ContractValueInput,
    ContractValueOutput,
    PaymentScheduleLine,
    ValuationAssumption,
)
from app.services.pv_calculator import PVCalculator, advance_date

logger = logging.getLogger(__name__)

_MAX_SCHEDULE_LINES_PER_OBLIGATION = 360  # 30 years of monthly payments


class ContractValueAgent:
    """
    Converts payment obligations into a payment schedule and calculates
    nominal total value and present value. Fully deterministic — no LLM calls.
    """

    def __init__(self) -> None:
        self._pv = PVCalculator()

    def run(self, input_model: ContractValueInput) -> ContractValueOutput:
        logger.info("ContractValueAgent: starting — contract_id=%s", input_model.contract_id)

        schedule: List[PaymentScheduleLine] = []
        assumptions: List[ValuationAssumption] = []
        warnings: List[CalculationWarning] = []
        uncertainty_flags: List[str] = list(input_model.uncertainty_flags)

        for obligation in input_model.obligations:
            if obligation.valuation_treatment == ValuationTreatment.EXCLUDE:
                logger.debug("Skipping excluded obligation: %s", obligation.obligation_id)
                continue

            if obligation.valuation_treatment == ValuationTreatment.CONDITIONAL:
                assumptions.append(ValuationAssumption(
                    assumption_id=f"assume-conditional-{obligation.obligation_id}",
                    description=(
                        f"Conditional obligation '{obligation.description}' included at full stated value."
                    ),
                    value="included_at_full_value",
                    source_uncertainty_flag=True,
                ))
                uncertainty_flags.append(f"conditional_obligation:{obligation.obligation_id}")

            amount, currency = self._resolve_amount(obligation, assumptions, warnings, uncertainty_flags)
            if amount is None:
                continue

            lines = self._build_lines(
                obligation, amount, currency,
                input_model.valuation_date, input_model.discount_rate,
                assumptions, warnings, uncertainty_flags,
            )
            schedule.extend(lines)

        total_nominal, total_pv, base_currency = _sum_schedule(schedule)

        confidence = ConfidenceLevel.HIGH
        if uncertainty_flags:
            confidence = ConfidenceLevel.MEDIUM
        if len(uncertainty_flags) > 3:
            confidence = ConfidenceLevel.LOW

        output = ContractValueOutput(
            contract_id=input_model.contract_id,
            total_nominal_value=MoneyAmount(amount=round(total_nominal, 2), currency=base_currency),
            total_present_value=MoneyAmount(amount=round(total_pv, 2), currency=base_currency),
            payment_schedule=schedule,
            assumptions=assumptions,
            warnings=warnings,
            uncertainty_flags=uncertainty_flags,
            confidence=confidence,
        )
        logger.info(
            "ContractValueAgent: complete — nominal=%.2f pv=%.2f lines=%d flags=%s",
            total_nominal, total_pv, len(schedule), uncertainty_flags,
        )
        return output

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _resolve_amount(
        self,
        obligation: PaymentObligation,
        assumptions: List[ValuationAssumption],
        warnings: List[CalculationWarning],
        uncertainty_flags: List[str],
    ) -> Tuple[Optional[float], Currency]:
        if obligation.amount is not None:
            return obligation.amount.amount, obligation.amount.currency

        # Try midpoint of range
        if obligation.amount_min is not None and obligation.amount_max is not None:
            mid = (obligation.amount_min.amount + obligation.amount_max.amount) / 2
            currency = obligation.amount_min.currency
            assumptions.append(ValuationAssumption(
                assumption_id=f"assume-amount-{obligation.obligation_id}",
                description=(
                    f"Amount for '{obligation.description}' estimated as midpoint of "
                    f"{obligation.amount_min.amount}–{obligation.amount_max.amount} {currency.value}."
                ),
                value=str(round(mid, 2)),
                source_uncertainty_flag=True,
            ))
            uncertainty_flags.append(f"amount_estimated:{obligation.obligation_id}")
            return mid, currency

        # No amount available
        warnings.append(CalculationWarning(
            warning_id=f"warn-no-amount-{obligation.obligation_id}",
            description=(
                f"No amount for obligation '{obligation.description}'; excluded from valuation."
            ),
            affected_obligation_ids=[obligation.obligation_id],
        ))
        uncertainty_flags.append(f"missing_amount:{obligation.obligation_id}")
        return None, Currency.USD

    def _build_lines(
        self,
        obligation: PaymentObligation,
        amount: float,
        currency: Currency,
        valuation_date: date,
        discount_rate: float,
        assumptions: List[ValuationAssumption],
        warnings: List[CalculationWarning],
        uncertainty_flags: List[str],
    ) -> List[PaymentScheduleLine]:
        freq = obligation.frequency.value

        if obligation.frequency == PaymentFrequency.ONE_TIME:
            return self._one_time_lines(
                obligation, amount, currency, valuation_date, discount_rate, assumptions,
            )

        if obligation.date_range is not None:
            return self._recurring_lines(
                obligation, amount, currency, valuation_date, discount_rate, freq,
            )

        # Recurring but no date range — treat as immediate single payment
        assumptions.append(ValuationAssumption(
            assumption_id=f"assume-nodates-{obligation.obligation_id}",
            description=(
                f"No date range for recurring obligation '{obligation.description}'; "
                "treated as a single immediate payment."
            ),
            value=str(valuation_date),
            source_uncertainty_flag=True,
        ))
        uncertainty_flags.append(f"missing_date_range:{obligation.obligation_id}")
        pv = self._pv.calculate_present_value(amount, discount_rate, 0, 12)
        return [_make_line(obligation.obligation_id, obligation.description, valuation_date,
                           amount, pv, currency, discount_rate, 0)]

    def _one_time_lines(
        self,
        obligation: PaymentObligation,
        amount: float,
        currency: Currency,
        valuation_date: date,
        discount_rate: float,
        assumptions: List[ValuationAssumption],
    ) -> List[PaymentScheduleLine]:
        payment_date = (
            obligation.due_date
            or (obligation.date_range.start if obligation.date_range else None)
        )

        if payment_date is None:
            payment_date = valuation_date
            assumptions.append(ValuationAssumption(
                assumption_id=f"assume-date-{obligation.obligation_id}",
                description=(
                    f"No payment date for '{obligation.description}'; assumed immediate."
                ),
                value=str(valuation_date),
                source_uncertainty_flag=True,
            ))

        periods = self._pv.periods_between(valuation_date, payment_date, "monthly")
        pv = self._pv.calculate_present_value(amount, discount_rate, periods, 12)
        return [_make_line(
            obligation.obligation_id, obligation.description,
            payment_date, amount, pv, currency, discount_rate, periods,
        )]

    def _recurring_lines(
        self,
        obligation: PaymentObligation,
        amount: float,
        currency: Currency,
        valuation_date: date,
        discount_rate: float,
        freq: str,
    ) -> List[PaymentScheduleLine]:
        start = obligation.date_range.start
        end = obligation.date_range.end

        ppy = self._pv.periods_per_year(freq)

        if end is not None:
            n_periods = self._pv.periods_between(start, end, freq) + 1
        else:
            # Open-ended — fall back to one line from start
            n_periods = 1

        n_periods = min(n_periods, _MAX_SCHEDULE_LINES_PER_OBLIGATION)

        lines = []
        for i in range(n_periods):
            payment_date = advance_date(start, i, freq)
            periods_from_val = self._pv.periods_between(valuation_date, payment_date, freq)
            pv = self._pv.calculate_present_value(amount, discount_rate, periods_from_val, ppy)
            lines.append(_make_line(
                obligation.obligation_id,
                f"{obligation.description} (period {i + 1})",
                payment_date, amount, pv, currency, discount_rate, periods_from_val,
            ))
        return lines


# --- module-level helpers ---

def _make_line(
    obligation_id: str,
    description: str,
    payment_date: date,
    nominal: float,
    pv: float,
    currency: Currency,
    rate: float,
    periods: int,
) -> PaymentScheduleLine:
    return PaymentScheduleLine(
        obligation_id=obligation_id,
        description=description,
        payment_date=payment_date,
        nominal_amount=MoneyAmount(amount=round(nominal, 2), currency=currency),
        present_value=MoneyAmount(amount=round(pv, 2), currency=currency),
        discount_rate_applied=rate,
        periods_discounted=periods,
    )


def _sum_schedule(
    schedule: List[PaymentScheduleLine],
) -> Tuple[float, float, Currency]:
    if not schedule:
        return 0.0, 0.0, Currency.USD
    base_currency = schedule[0].nominal_amount.currency
    total_nominal = sum(ln.nominal_amount.amount for ln in schedule)
    total_pv = sum(ln.present_value.amount for ln in schedule)
    return total_nominal, total_pv, base_currency
