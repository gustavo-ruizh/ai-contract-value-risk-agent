import calendar
from datetime import date, timedelta


_PERIODS_PER_YEAR = {
    "monthly": 12,
    "quarterly": 4,
    "annually": 1,
    "semi_annually": 2,
    "weekly": 52,
    "daily": 365,
    "one_time": 1,
    "on_milestone": 1,
    "irregular": 12,  # treated as monthly for discounting
}


class PVCalculator:
    """
    Computes present value of future cash flows using discrete discounting.
    All discount rates are annual; periods are derived from payment frequency.
    """

    def calculate_present_value(
        self,
        amount: float,
        annual_discount_rate: float,
        periods: int,
        periods_per_year: int = 12,
    ) -> float:
        """
        PV = FV / (1 + r/n)^t
        where r = annual rate, n = periods/year, t = number of periods.

        Returns amount unchanged if rate is zero or periods is zero or less.
        """
        if amount <= 0:
            return max(0.0, amount)
        if annual_discount_rate <= 0 or periods <= 0:
            return amount
        period_rate = annual_discount_rate / periods_per_year
        return amount / (1 + period_rate) ** periods

    def periods_between(self, start: date, end: date, frequency: str) -> int:
        """Return the number of complete payment periods between start and end."""
        if end <= start:
            return 0

        ppy = self.periods_per_year(frequency)

        if ppy == 365:
            return max(0, (end - start).days)
        if ppy == 52:
            return max(0, (end - start).days // 7)

        # Month-based frequencies
        total_months = (end.year - start.year) * 12 + (end.month - start.month)
        if ppy == 12:
            return max(0, total_months)
        if ppy == 6:
            return max(0, total_months // 2)
        if ppy == 4:
            return max(0, total_months // 3)
        if ppy == 2:
            return max(0, total_months // 6)
        if ppy == 1:
            years = end.year - start.year
            if (end.month, end.day) < (start.month, start.day):
                years -= 1
            return max(0, years)

        return max(0, total_months)

    def periods_per_year(self, frequency: str) -> int:
        """Map a PaymentFrequency value to the number of periods in one year."""
        return _PERIODS_PER_YEAR.get(frequency, 12)


def advance_date(start: date, n: int, frequency: str) -> date:
    """Return the date after n payment periods of the given frequency."""
    if frequency == "daily":
        return start + timedelta(days=n)
    if frequency == "weekly":
        return start + timedelta(weeks=n)
    if frequency == "monthly" or frequency == "irregular":
        return _add_months(start, n)
    if frequency == "quarterly":
        return _add_months(start, n * 3)
    if frequency == "semi_annually":
        return _add_months(start, n * 6)
    if frequency == "annually":
        return _add_months(start, n * 12)
    return _add_months(start, n)


def _add_months(d: date, months: int) -> date:
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)
