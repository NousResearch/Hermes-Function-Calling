from validator import validate_function_call_schema


def _build_signature(name, properties, required):
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
    ]


def test_validator_accepts_zero_for_integer():
    signatures = _build_signature(
        "set_threshold",
        {"threshold": {"type": "integer"}},
        ["threshold"],
    )

    valid, message = validate_function_call_schema(
        {"name": "set_threshold", "arguments": {"threshold": 0}},
        signatures,
    )

    assert valid is True
    assert message is None


def test_validator_rejects_bool_for_integer():
    signatures = _build_signature(
        "set_threshold",
        {"threshold": {"type": "integer"}},
        ["threshold"],
    )

    valid, message = validate_function_call_schema(
        {"name": "set_threshold", "arguments": {"threshold": False}},
        signatures,
    )

    assert valid is False
    assert "Type mismatch for parameter threshold" in message


def test_validator_rejects_none_for_required_string():
    signatures = _build_signature(
        "lookup_symbol",
        {"symbol": {"type": "string"}},
        ["symbol"],
    )

    valid, message = validate_function_call_schema(
        {"name": "lookup_symbol", "arguments": {"symbol": None}},
        signatures,
    )

    assert valid is False
    assert "Type mismatch for parameter symbol" in message


def test_validator_rejects_invalid_enum_for_falsey_string():
    signatures = _build_signature(
        "submit_order",
        {"side": {"type": "string", "enum": ["buy", "sell"]}},
        ["side"],
    )

    valid, message = validate_function_call_schema(
        {"name": "submit_order", "arguments": {"side": ""}},
        signatures,
    )

    assert valid is False
    assert "Invalid value '' for parameter side" in message