from __future__ import annotations

import math

from src.data import parse_listish, parse_salary, to_float


def test_to_float_handles_percent_and_missing() -> None:
    assert to_float("12.5%") == 12.5
    assert to_float("1,234") == 1234.0
    assert math.isnan(to_float(""))
    assert math.isnan(to_float(None))


def test_parse_listish_multiple_formats() -> None:
    assert parse_listish(None) == []
    assert parse_listish("[") == ["["]
    assert parse_listish("[\"a\", \"b\"]") == ["a", "b"]
    assert parse_listish("['a', 'b']") == ["a", "b"]
    assert parse_listish("a;b") == ["a", "b"]
    assert parse_listish([" x ", "", "y"]) == ["x", "y"]


def test_parse_salary_k_and_range() -> None:
    mn, mx, med = parse_salary("$90k-$110k")
    assert mn == 90000.0
    assert mx == 110000.0
    assert med == 100000.0

    mn2, mx2, med2 = parse_salary("120k")
    assert mn2 == mx2 == med2 == 120000.0

    mn3, mx3, med3 = parse_salary(None)
    assert math.isnan(mn3) and math.isnan(mx3) and math.isnan(med3)
