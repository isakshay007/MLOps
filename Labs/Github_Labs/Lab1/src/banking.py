import csv
from pathlib import Path

DEFAULT_LOANS_FILE = Path(__file__).resolve().parent.parent / "data" / "loans.csv"


def _check_numbers(*values):
    """
    Raises ValueError if any value is not an int/float.
    bool is rejected explicitly because Python treats True/False as ints.
    """
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("All inputs must be numbers.")


def simple_interest(principal, rate, years):
    """
    Calculates simple interest.
    Args:
        principal (int/float): Amount borrowed or invested. Must be > 0.
        rate (int/float): Annual interest rate in percent (e.g. 5 for 5%). Must be >= 0.
        years (int/float): Time period in years. Must be > 0.
    Returns:
        float: Interest earned, rounded to 2 decimals.
        Raises:
        ValueError: If inputs are not numbers or are out of range.
    """
    _check_numbers(principal, rate, years)
    if principal <= 0 or rate < 0 or years <= 0:
        raise ValueError("Principal and years must be positive; rate cannot be negative.")
    return round(principal * rate * years / 100, 2)


def compound_interest(principal, rate, years, n=1):
    """
    Calculates compound interest.
    Args:
        principal (int/float): Amount borrowed or invested. Must be > 0.
        rate (int/float): Annual interest rate in percent. Must be >= 0.
        years (int/float): Time period in years. Must be > 0.
        n (int): Times interest is compounded per year (1 = yearly, 12 = monthly).
    Returns:
        float: Interest earned (final amount minus principal), rounded to 2 decimals.
        Raises:
        ValueError: If inputs are not numbers or are out of range.
    """
    _check_numbers(principal, rate, years, n)
    if principal <= 0 or rate < 0 or years <= 0 or n <= 0:
        raise ValueError("Principal, years and n must be positive; rate cannot be negative.")
    amount = principal * (1 + rate / (100 * n)) ** (n * years)
    return round(amount - principal, 2)


def emi(principal, annual_rate, months):
    """
    Calculates the EMI (Equated Monthly Installment) for a loan.
    Formula: P * r * (1 + r)^n / ((1 + r)^n - 1), where r is the monthly rate.
    Args:
        principal (int/float): Loan amount. Must be > 0.
        annual_rate (int/float): Annual interest rate in percent. Must be >= 0.
        months (int): Loan term in months. Must be > 0.
    Returns:
        float: Monthly payment, rounded to 2 decimals.
        Raises:
        ValueError: If inputs are not numbers or are out of range.
    """
    _check_numbers(principal, annual_rate, months)
    if principal <= 0 or annual_rate < 0 or months <= 0:
        raise ValueError("Principal and months must be positive; rate cannot be negative.")
    r = annual_rate / (12 * 100)
    if r == 0:
        # Interest-free loan: avoid dividing by zero
        return round(principal / months, 2)
    growth = (1 + r) ** months
    return round(principal * r * growth / (growth - 1), 2)


def loan_summary(principal, annual_rate, months):
    """
    Combines the EMI calculation into a full loan breakdown.
    Args:
        principal (int/float): Loan amount.
        annual_rate (int/float): Annual interest rate in percent.
        months (int): Loan term in months.
    Returns:
        dict: monthly_emi, total_payment and total_interest.
    """
    monthly = emi(principal, annual_rate, months)
    total_payment = round(monthly * months, 2)
    return {
        "monthly_emi": monthly,
        "total_payment": total_payment,
        "total_interest": round(total_payment - principal, 2),
    }


def summarize_loans(path=DEFAULT_LOANS_FILE):
    """
    Reads loans from a CSV file and computes a loan_summary for each one.
    Args:
        path (str/Path): CSV with columns loan_id, principal, annual_rate, months.
    Returns:
        list[dict]: One summary per loan, including its loan_id.
        Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Loans file not found: {path}")

    results = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            summary = loan_summary(
                float(row["principal"]),
                float(row["annual_rate"]),
                int(row["months"]),
            )
            results.append({"loan_id": row["loan_id"], **summary})
    return results


if __name__ == "__main__":
    for loan in summarize_loans():
        print(loan)
