from pathlib import Path

from pyrecest._backend.capabilities import API_BACKEND_CAPABILITIES
from pyrecest.api_registry import (
    PUBLIC_API_CATEGORIES,
    PUBLIC_API_REGISTRY,
    get_public_api_registry_entry,
    iter_public_api_registry,
)
from scripts.check_public_api_registry import render_markdown, validate_registry


def test_public_api_registry_rows_are_valid():
    assert not validate_registry()
    assert PUBLIC_API_REGISTRY
    for api_name, row in iter_public_api_registry():
        assert api_name
        assert row["category"] in PUBLIC_API_CATEGORIES
        assert row["module"].startswith("pyrecest")
        assert row["notes"]


def test_backend_capability_rows_have_public_api_registry_entries():
    for api_name in API_BACKEND_CAPABILITIES:
        assert api_name in PUBLIC_API_REGISTRY


def test_get_public_api_registry_entry_returns_copy():
    row = get_public_api_registry_entry("KalmanFilter")
    row["category"] = "mutated"
    assert PUBLIC_API_REGISTRY["KalmanFilter"]["category"] == "stable"


def test_get_public_api_registry_entry_returns_empty_for_non_string_lookup():
    for value in (123, ("KalmanFilter",), ["KalmanFilter"], {"api": "KalmanFilter"}):
        assert get_public_api_registry_entry(value) == {}


def test_iter_public_api_registry_returns_row_copies():
    rows = dict(iter_public_api_registry())
    rows["KalmanFilter"]["category"] = "mutated"
    assert PUBLIC_API_REGISTRY["KalmanFilter"]["category"] == "stable"


def test_public_api_registry_document_contains_generated_table():
    document = Path("docs/public-api-registry.md").read_text(encoding="utf-8")
    assert render_markdown() in document
