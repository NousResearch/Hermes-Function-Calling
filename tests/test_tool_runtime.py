import json

import pytest

from tool_runtime import format_tool_result, invoke_tool


def test_invoke_tool_passes_kwargs_independent_of_dict_order():
    def combine(origin, destination):
        return f"{origin}->{destination}"

    result = invoke_tool(
        combine,
        "combine",
        {"destination": "LAX", "origin": "JFK"},
    )

    assert result == "JFK->LAX"


def test_invoke_tool_prefers_invoke_when_available():
    class FakeTool:
        def __init__(self):
            self.received = None

        def invoke(self, args):
            self.received = args
            return {"ok": True}

    tool = FakeTool()
    result = invoke_tool(tool, "fake", {"x": 1})

    assert result == {"ok": True}
    assert tool.received == {"x": 1}


def test_invoke_tool_rejects_unknown_function():
    with pytest.raises(ValueError, match="not defined"):
        invoke_tool(None, "missing_tool", {})


def test_format_tool_result_emits_valid_json():
    payload = format_tool_result("echo", "hello")
    assert json.loads(payload) == {"name": "echo", "content": "hello"}


def test_format_tool_result_serializes_non_json_objects():
    class Unserializable:
        def __str__(self):
            return "unserializable-value"

    payload = format_tool_result("echo", Unserializable())
    assert json.loads(payload) == {"name": "echo", "content": "unserializable-value"}