# Copilot instructions for Hermes-Function-Calling

## Project overview
This repository implements Hermes-style function calling and JSON-schema generation for local LLM inference. The project is built around Python scripts that load a Hugging Face model, build a ChatML prompt, validate function calls, and execute tools in a recursive loop.

Key files:
- `functioncall.py`: main recursive function-calling inference loop.
- `jsonmode.py`: JSON mode inference and schema-bound output generation.
- `functions.py`: tool registry and available callable functions. New tools should be declared with `@tool` and included in `get_openai_tools()`.
- `schema.py`: Pydantic definitions for function calls and signatures.
- `prompter.py`: builds the system prompt from `prompt_assets/sys_prompt.yml`.
- `validator.py`: validates generated function-call payloads.
- `utils.py`: shared parsing and formatting helpers.

## Working conventions
- Keep changes surgical and Pythonic. Prefer small, targeted edits over broad refactors.
- Preserve the repository's existing behavior and CLI conventions documented in `README.md`.
- When changing prompt handling, retain compatibility with Hermes ChatML-style messages and XML tags such as `<tool_call>` and `<tool_response>`.
- When changing schema or validation logic, keep the `FunctionCall` contract in `schema.py` and the validation flow in `validator.py` aligned.
- When adding or modifying tool functions in `functions.py`, keep the function signature, docstring, and OpenAI tool conversion consistent with the existing pattern.
- Model behavior is highly prompt-sensitive; do not “simplify” or rewrite prompt logic unless the change specifically addresses the task.

## Tool registry guidance
- Add new callable tools in `functions.py` using the existing `@tool` decorator pattern.
- Expose the tool through `get_openai_tools()` so the model can see it.
- Return values should remain JSON-serializable or pandas-friendly when used by the model.
- Keep function descriptions specific and accurate; the model depends on them for argument selection.

## Validation workflow
- There is no dedicated repo test suite in this project; prefer lightweight smoke checks and import validation.
- Use the safest targeted validation command for the change, such as:
  - `python -m py_compile functioncall.py jsonmode.py functions.py prompter.py schema.py validator.py utils.py`
  - `python functioncall.py --query "I need the current stock price of Tesla (TSLA)"`
  - `python jsonmode.py --query "Please return a json object to represent Goku from the anime Dragon Ball Z?"`
- If a local model is not available or runtime validation is not possible, state that explicitly instead of claiming a full end-to-end verification.

## Reliability and safety
- Preserve the existing defaults in the CLI arguments and the documented setup flow in `README.md`.
- Avoid introducing unnecessary dependencies or broad changes to the prompt format.
- Be careful with the `code_interpreter` tool in `functions.py`; it executes Python code and should remain explicit and constrained.
- Do not silently broaden tool behavior or loosen validation checks in ways that could produce invalid or unsafe function calls.

## Documentation expectations
- Update `README.md` when behavior, usage, or tool registration changes materially.
- Keep examples consistent with the workspace's actual scripts and defaults.
