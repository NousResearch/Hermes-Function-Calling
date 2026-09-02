#!/usr/bin/env bash
# Rebuild the Hermes Function Calling proof movie.
# Requires: ffmpeg, ffprobe, uv, ttyd, Google Chrome, the repo .venv
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROOF="$ROOT/proof"
SKILL="$PROOF/scripts"
SESSION=hermes-demo
TTYD_PORT=7681
export PATH="$HOME/.local/bin:$ROOT/.venv/bin:$PATH"

cd "$ROOT"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "missing $ROOT/.venv — create it and pip-install langchain, pydantic, jsonschema, yfinance, pandas, art, pyyaml, pillow, websockets" >&2
  exit 1
fi

# --- terminal session served over HTTP ---------------------------------
tmux has-session -t "$SESSION" 2>/dev/null || \
  tmux new-session -d -s "$SESSION" -x 125 -y 34 -- bash -l

# Point the pane at the repo with a short prompt.
tmux send-keys -t "$SESSION" "cd $ROOT" Enter
tmux send-keys -t "$SESSION" "export PATH=$ROOT/.venv/bin:\$HOME/.local/bin:\$PATH" Enter
tmux send-keys -t "$SESSION" "export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore" Enter
tmux send-keys -t "$SESSION" "export PS1='$ '" Enter
sleep 0.4

if ! curl -fsS "http://127.0.0.1:${TTYD_PORT}/" >/dev/null 2>&1; then
  ttyd -p "$TTYD_PORT" \
    -t fontSize=17 \
    -t 'fontFamily=DejaVu Sans Mono,monospace' \
    -t 'theme={"background":"#101014","foreground":"#e8e6e1"}' \
    tmux attach -t "$SESSION" >/tmp/ttyd-hermes.log 2>&1 &
  # wait until it answers
  for _ in $(seq 1 30); do
    curl -fsS "http://127.0.0.1:${TTYD_PORT}/" >/dev/null 2>&1 && break
    sleep 0.3
  done
fi

python "$PROOF/render_cards.py"
python "$PROOF/film.py" tools    "http://127.0.0.1:${TTYD_PORT}/"
python "$PROOF/film.py" parse    "http://127.0.0.1:${TTYD_PORT}/"
python "$PROOF/film.py" validate "http://127.0.0.1:${TTYD_PORT}/"
python "$PROOF/film.py" execute  "http://127.0.0.1:${TTYD_PORT}/"

"$SKILL/narrate"        "$PROOF/scenes.yaml" "$PROOF/narration/"
"$SKILL/assemble"       "$PROOF/scenes.yaml" "$PROOF/silent-cut.mp4"
"$SKILL/make-subtitles" "$PROOF/narration/manifest.json" "$PROOF/movie.srt" \
                        --offsets-json "$PROOF/segments/offsets.json"
"$SKILL/burn-subtitles" "$PROOF/silent-cut.mp4" "$PROOF/movie.srt" "$PROOF/movie.mp4"
"$SKILL/check-movie"    "$PROOF/movie.mp4"
