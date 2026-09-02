#!/usr/bin/env python3
"""Filmed CLI for the Hermes function-calling toolkit.

Each subcommand exercises real repository code — tool schema conversion,
<xml> tool-call parsing, pydantic/jsonschema validation, and live tool
execution. Nothing here is staged or replayed from a fixture.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.getLogger("function-calling-inference").setLevel(logging.CRITICAL)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def banner(title: str) -> None:
    line = "=" * 72
    print(line)
    print(f"  Hermes Function Calling  ·  {title}")
    print(line)
    print()


def cmd_tools(_args: argparse.Namespace) -> int:
    from functions import get_openai_tools

    tools = get_openai_tools()
    banner("OpenAI tool schemas from functions.py")
    print(f"{len(tools)} tools registered:\n")
    for i, tool in enumerate(tools, 1):
        fn = tool["function"]
        print(f"  {i:2}. {fn['name']}")
    fund = next(t for t in tools if t["function"]["name"] == "get_stock_fundamentals")
    print("\nConverted schema for get_stock_fundamentals:\n")
    print(json.dumps(fund["function"]["parameters"], indent=2))
    return 0


def cmd_parse(_args: argparse.Namespace) -> int:
    from utils import validate_and_extract_tool_calls

    banner("Parse a Hermes <tool_call> from model output")
    raw = (
        "<tool_call>\n"
        '{"name": "code_interpreter",'
        ' "arguments": {"code_markdown": "```python\\nprint(6 * 7)\\n```"}}\n'
        "</tool_call>"
    )
    print("Raw assistant message:\n")
    print(raw)
    print()
    ok, calls, err = validate_and_extract_tool_calls(raw)
    print(f"extracted_ok = {ok}")
    if err:
        print(f"error        = {err}")
    print("tool_calls   =")
    print(json.dumps(calls, indent=2))
    return 0 if ok else 1


def cmd_validate(_args: argparse.Namespace) -> int:
    from functions import get_openai_tools
    from validator import validate_function_call_schema

    banner("Validate calls against the live tool schemas")
    tools = get_openai_tools()
    cases = [
        (
            "valid call",
            {
                "name": "get_stock_fundamentals",
                "arguments": {"symbol": "TSLA"},
            },
        ),
        (
            "missing required argument",
            {"name": "get_stock_fundamentals", "arguments": {}},
        ),
        (
            "unknown function",
            {"name": "launch_the_missiles", "arguments": {"symbol": "TSLA"}},
        ),
    ]
    for label, call in cases:
        ok, err = validate_function_call_schema(call, tools)
        verdict = "ACCEPT" if ok else "REJECT"
        print(f"{verdict:6}  {label}")
        print(f"        call = {json.dumps(call)}")
        if err:
            print(f"        why  = {err}")
        print()
    return 0


def cmd_execute(_args: argparse.Namespace) -> int:
    from functions import code_interpreter

    banner("Execute a real tool: code_interpreter")
    code_markdown = (
        "```python\n"
        "squares = [n * n for n in range(1, 8)]\n"
        "total = sum(squares)\n"
        "print('squares', squares)\n"
        "print('total', total)\n"
        "```"
    )
    print("Invoking functions.code_interpreter with this Python:\n")
    print(code_markdown)
    print("Live result (exec namespace, not a fixture):\n")
    result = code_interpreter.invoke({"code_markdown": code_markdown})
    print(json.dumps(result, indent=2))
    expected = {"squares": [1, 4, 9, 16, 25, 36, 49], "total": 140}
    if result != expected:
        print(f"\nUNEXPECTED: {result!r} != {expected!r}", file=sys.stderr)
        return 1
    print("\nOK  squares 1..7 sum to 140")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tools", help="list converted OpenAI tool schemas")
    sub.add_parser("parse", help="extract JSON from a <tool_call> block")
    sub.add_parser("validate", help="accept a good call, reject bad ones")
    sub.add_parser("execute", help="run code_interpreter for real")
    args = parser.parse_args()
    return {
        "tools": cmd_tools,
        "parse": cmd_parse,
        "validate": cmd_validate,
        "execute": cmd_execute,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
