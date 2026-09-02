# Proof movie — Hermes Function Calling

A rebuildable movie that proves the function-calling **toolkit** works on
this machine. It does **not** load Hermes-2-Pro (that needs a GPU and
several gigabytes of weights). Yahoo Finance also 429'd from this network,
so the live tool shown is `code_interpreter` rather than a faked stock quote.

Pipeline is the proving-it-works skill (`prime-radiant-inc/proving-it-works`):
terminal capture via ttyd + tmux + Chrome, local narration, burned-in
subtitles, then `check-movie`.

## What the movie shows

| Scene | Real code path |
|---|---|
| tools | `functions.get_openai_tools()` |
| parse | `utils.validate_and_extract_tool_calls()` |
| validate | `validator.validate_function_call_schema()` |
| execute | `functions.code_interpreter.invoke(...)` |

## Rebuild

```bash
# once: lightweight venv (no torch / flash-attn)
uv venv .venv
uv pip install --python .venv/bin/python \
  langchain==0.1.9 pydantic==2.6.2 jsonschema==4.21.1 \
  yfinance==0.2.36 pandas==2.2.0 beautifulsoup4 requests art pyyaml \
  pillow websockets

# ttyd + uv on PATH, Google Chrome installed
proof/make-movie.sh
```

`proof/scripts/` are the proving-it-works helpers (MIT), vendored so the
movie can be re-cut without a plugin install. Title/end cards are PNG
stills from `render_cards.py` (PIL) so assemble does not depend on a
browser.

Frames, narration wavs, and the encoded movie stay out of git — they are
scratch. `scenes.yaml`, `show.py`, `film.py`, `render_cards.py`,
`cards/*.png`, and `make-movie.sh` are the source of truth.
