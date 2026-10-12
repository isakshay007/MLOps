import sys
import os
import unittest

# Get the path to the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src import banking


class TestBanking(unittest.TestCase):

    def test_simple_interest(self):
        self.assertEqual(banking.simple_interest(10000, 5, 2), 1000.0)
        self.assertEqual(banking.simple_interest(5000, 10, 1), 500.0)
        self.assertEqual(banking.simple_interest(1000, 0, 5), 0.0)

    def test_compound_interest(self):
        self.assertEqual(banking.compound_interest(10000, 10, 2), 2100.0)
        self.assertAlmostEqual(banking.compound_interest(1000, 12, 1, 12), 126.83, places=2)
        self.assertGreater(banking.compound_interest(10000, 8, 10), banking.simple_interest(10000, 8, 10))

    def test_emi(self):
        self.assertEqual(banking.emi(100000, 12, 12), 8884.88)
        self.assertEqual(banking.emi(10000, 12, 24), 470.73)
        self.assertEqual(banking.emi(12000, 0, 12), 1000.0)

    def test_loan_summary(self):
        summary = banking.loan_summary(100000, 12, 12)
        self.assertEqual(summary["monthly_emi"], 8884.88)
        self.assertEqual(summary["total_payment"], 106618.56)
        self.assertEqual(summary["total_interest"], 6618.56)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            banking.simple_interest("1000", 5, 2)
        with self.assertRaises(ValueError):
            banking.simple_interest(-1000, 5, 2)
        with self.assertRaises(ValueError):
            banking.compound_interest(1000, -5, 2)
        with self.assertRaises(ValueError):
            banking.emi(1000, 5, 0)

    def test_summarize_loans(self):
        results = banking.summarize_loans()
        self.assertEqual(len(results), 8)
        self.assertEqual(results[0]["loan_id"], "L001")
        self.assertTrue(all("total_interest" in r for r in results))

    def test_summarize_loans_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            banking.summarize_loans("does_not_exist.csv")


if __name__ == '__main__':
    unittest.main()
