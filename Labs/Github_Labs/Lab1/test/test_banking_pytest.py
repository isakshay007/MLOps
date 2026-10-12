import pytest
from src import banking


# Expected values are calculated by hand from the formulas in banking.py

@pytest.mark.parametrize("principal, rate, years, expected", [
    (10000, 5, 2, 1000.0),
    (5000, 10, 1, 500.0),
    (2500, 4, 3, 300.0),
    (1000, 0, 5, 0.0),
])
def test_simple_interest(principal, rate, years, expected):
    assert banking.simple_interest(principal, rate, years) == expected


@pytest.mark.parametrize("principal, rate, years, n, expected", [
    (10000, 10, 2, 1, 2100.0),      # 10000 * 1.1^2 - 10000
    (1000, 5, 1, 1, 50.0),          # one year yearly = simple interest
    (1000, 12, 1, 12, 126.83),      # monthly compounding
    (5000, 0, 3, 1, 0.0),
])
def test_compound_interest(principal, rate, years, n, expected):
    assert banking.compound_interest(principal, rate, years, n) == expected


def test_compound_beats_simple_over_time():
    assert banking.compound_interest(10000, 8, 10) > banking.simple_interest(10000, 8, 10)


@pytest.mark.parametrize("principal, annual_rate, months, expected", [
    (100000, 12, 12, 8884.88),
    (10000, 12, 24, 470.73),
    (12000, 0, 12, 1000.0),         # interest-free loan
])
def test_emi(principal, annual_rate, months, expected):
    assert banking.emi(principal, annual_rate, months) == expected


def test_loan_summary():
    summary = banking.loan_summary(100000, 12, 12)
    assert summary == {
        "monthly_emi": 8884.88,
        "total_payment": 106618.56,
        "total_interest": 6618.56,
    }


def test_loan_summary_zero_rate_has_no_interest():
    assert banking.loan_summary(18000, 0, 36)["total_interest"] == 0.0


@pytest.mark.parametrize("func, args", [
    (banking.simple_interest, ("1000", 5, 2)),
    (banking.compound_interest, (1000, None, 2)),
    (banking.emi, (1000, 5, True)),
])
def test_rejects_non_numbers(func, args):
    with pytest.raises(ValueError):
        func(*args)


@pytest.mark.parametrize("func, args", [
    (banking.simple_interest, (-1000, 5, 2)),     # negative principal
    (banking.simple_interest, (1000, -5, 2)),     # negative rate
    (banking.compound_interest, (1000, 5, 0)),    # zero years
    (banking.compound_interest, (1000, 5, 2, 0)), # zero compounding periods
    (banking.emi, (1000, 5, 0)),                  # zero-month term
    (banking.emi, (0, 5, 12)),                    # zero principal
])
def test_rejects_out_of_range(func, args):
    with pytest.raises(ValueError):
        func(*args)


def test_summarize_loans_reads_dataset():
    results = banking.summarize_loans()
    assert len(results) == 8
    assert results[0]["loan_id"] == "L001"
    assert all(r["total_payment"] >= r["monthly_emi"] for r in results)


def test_summarize_loans_custom_file(tmp_path):
    csv_file = tmp_path / "loans.csv"
    csv_file.write_text("loan_id,principal,annual_rate,months\nX1,100000,12,12\n")
    assert banking.summarize_loans(csv_file) == [
        {"loan_id": "X1", "monthly_emi": 8884.88, "total_payment": 106618.56, "total_interest": 6618.56}
    ]


def test_summarize_loans_missing_file():
    with pytest.raises(FileNotFoundError):
        banking.summarize_loans("does_not_exist.csv")
