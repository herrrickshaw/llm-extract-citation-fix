"""Calculated fields can be recomputed from grounded inputs."""
import pytest

from llm_extract import check_derived, extract_structured

MID = {"mid": (["low", "high"], lambda lo, hi: (lo + hi) / 2)}


def result(mid, low=210, high="222", low_grounded=True):
    grounding = {k: {"grounded": True, "span": "x"} for k in ("low", "high", "mid")}
    grounding["low"]["grounded"] = low_grounded
    return {"low": low, "high": high, "mid": mid, "_grounding": grounding}


@pytest.mark.parametrize("mid,ok", [(216, True), ("216.00", True), (217, False), (None, False)])
def test_recomputed_value_is_compared(mid, ok):
    out = check_derived(result(mid), MID)
    assert out["_grounding"]["mid"]["derived_ok"] is ok
    assert out["mid"] == mid


@pytest.mark.parametrize("kwargs", [{"low_grounded": False}, {"low": None}, {"low": "n/a"}])
def test_unusable_inputs_mean_unverified(kwargs):
    assert check_derived(result(216, **kwargs), MID)["_grounding"]["mid"]["derived_ok"] is None


def test_division_by_zero_means_unverified():
    rule = {"mid": (["low", "high"], lambda lo, hi: lo / 0)}
    assert check_derived(result(1), rule)["_grounding"]["mid"]["derived_ok"] is None


def test_unknown_field_and_missing_citations_are_errors():
    with pytest.raises(ValueError, match="not in the schema"):
        check_derived(result(1), {"nope": ([], lambda: 1)})
    with pytest.raises(ValueError, match="require_citations"):
        extract_structured("t", {"mid": "n"}, require_citations=False, derived=MID)
