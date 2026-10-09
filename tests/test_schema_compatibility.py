from typing import Any, Iterator

import pytest
from google.genai import types

from app.services.gemini import (
    BOX_DETECTION_SCHEMA,
    ISSUES_SCHEMA,
    ORIENTATION_SCHEMA,
    PAIR_CONFIRMATION_SCHEMA,
)

# Gemini's response_schema is an OpenAPI 3.0 subset, not full JSON Schema.
_SUPPORTED_TYPES = {"object", "array", "string", "number", "integer", "boolean"}


def _iter_schema_nodes(schema: Any) -> Iterator[dict]:
    if isinstance(schema, dict):
        yield schema
        for value in schema.values():
            yield from _iter_schema_nodes(value)
    elif isinstance(schema, list):
        for item in schema:
            yield from _iter_schema_nodes(item)


def _assert_gemini_structured_output_compatible(schema: dict, schema_name: str) -> None:
    # The SDK forwards a plain dict straight to the API without validating it, so a schema the API
    # dislikes only surfaces as a 400 at request time. These checks front-run that:
    # additionalProperties is legal JSON Schema, and required by some other providers, but is not
    # part of the subset the Gemini Developer API accepts.
    for node in _iter_schema_nodes(schema):
        node_type = node.get("type")
        if node_type is None:
            continue
        assert node_type in _SUPPORTED_TYPES, f"{schema_name}: unsupported type {node_type!r} in {node}"
        assert "additionalProperties" not in node, (
            f"{schema_name}: additionalProperties is not supported by Gemini structured output: {node}"
        )
        if node_type == "object":
            properties = node.get("properties")
            assert properties, f"{schema_name}: object node must declare properties: {node}"
            unknown = set(node.get("required", [])) - set(properties)
            assert not unknown, f"{schema_name}: required names absent from properties: {sorted(unknown)}"
        if node_type == "array":
            assert "items" in node, f"{schema_name}: array node must declare items: {node}"

    try:
        types.Schema.model_validate(schema)
    except Exception as exc:  # pragma: no cover - the assertion message is the point
        pytest.fail(f"{schema_name} is not a valid Gemini schema: {exc}")


def test_box_detection_schema_is_gemini_structured_output_compatible() -> None:
    _assert_gemini_structured_output_compatible(BOX_DETECTION_SCHEMA, "BOX_DETECTION_SCHEMA")


def test_orientation_schema_is_gemini_structured_output_compatible() -> None:
    _assert_gemini_structured_output_compatible(ORIENTATION_SCHEMA, "ORIENTATION_SCHEMA")

    # top_edge stays an enum on the Gemini side too: the quarter turn is looked up in code, and an
    # unconstrained string would let the model answer with an angle again.
    top_edge = ORIENTATION_SCHEMA["properties"]["documents"]["items"]["properties"]["top_edge"]
    assert top_edge["enum"] == ["top", "right", "bottom", "left"]


def test_issues_schema_is_gemini_structured_output_compatible() -> None:
    _assert_gemini_structured_output_compatible(ISSUES_SCHEMA, "ISSUES_SCHEMA")


def test_pair_confirmation_schema_is_gemini_structured_output_compatible() -> None:
    _assert_gemini_structured_output_compatible(PAIR_CONFIRMATION_SCHEMA, "PAIR_CONFIRMATION_SCHEMA")
