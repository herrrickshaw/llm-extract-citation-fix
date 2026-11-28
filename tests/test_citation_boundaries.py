"""A citation must not match part of a longer word or number."""
import pytest

from llm_extract import _verify_citation

SOURCE = (
    "Cash Rs 15,230.75; price band Rs. 210 to Rs. 222; face value Rs 10. "
    "Listed on NSE, BSE. FY2025."
)


@pytest.mark.parametrize("span", [
    "Rs 1", "Rs 15", "Rs 15,230", "230.75", "75", "Rs. 21", "SE", "2025.1", "FY202",
])
def test_partial_tokens_are_rejected(span):
    assert not _verify_citation(span, SOURCE)


@pytest.mark.parametrize("span", [
    "Rs 15,230.75", "15,230.75", "Rs. 210 to Rs. 222", "222", "Rs 10", "NSE, BSE", "NSE", "FY2025.",
])
def test_whole_tokens_are_accepted(span):
    assert _verify_citation(span, SOURCE)


def test_later_whole_occurrence_counts_after_an_earlier_partial_one():
    assert _verify_citation("Rs 1", "Rs 15 now, was Rs 1 before")
