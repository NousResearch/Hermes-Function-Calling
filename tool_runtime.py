import json
from collections.abc import Mapping
from typing import Any


def invoke_tool(function_to_call: Any, function_name: str, function_args: Mapping[str, Any] | None) -> Any:
    """Invoke tool implementations with stable argument handling.

    Supports both LangChain-style tool objects exposing ``invoke`` and plain
    Python callables.
    """
    if function_to_call is None:
        raise ValueError(f"Function '{function_name}' is not defined in functions.py")

    if function_args is None:
        function_args = {}
    if not isinstance(function_args, Mapping):
        raise TypeError(
            f"Invalid arguments payload for function '{function_name}'. "
            f"Expected a JSON object/dict, got {type(function_args)}."
        )

    normalized_args = dict(function_args)
    tool_invoke = getattr(function_to_call, "invoke", None)

    if callable(tool_invoke):
        return tool_invoke(normalized_args)
    if callable(function_to_call):
        return function_to_call(**normalized_args)

    raise TypeError(
        f"Function '{function_name}' is not callable and does not implement an invoke() method."
    )


def _json_default(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass
    return str(value)


def format_tool_result(function_name: str, function_response: Any) -> str:
    payload = {"name": function_name, "content": function_response}
    return json.dumps(payload, default=_json_default)