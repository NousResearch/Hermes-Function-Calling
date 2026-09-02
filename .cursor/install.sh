#!/usr/bin/env bash
#
# Idempotent install script for the Hermes-Function-Calling Cloud Agent dev
# environment. Creates a project virtualenv at .venv and installs a CPU-only
# dependency set (see .cursor/requirements-cpu.txt for why this differs from the
# GPU-oriented root requirements.txt).
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
VENV_DIR=".venv"

# Ensure the venv module is available (Debian splits it into python3-venv).
if ! "$PYTHON" -c "import ensurepip" >/dev/null 2>&1; then
  echo "==> Installing python3-venv"
  sudo apt-get update -qq
  sudo apt-get install -y "$("$PYTHON" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}-venv")')" python3-pip
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "==> Creating virtualenv at $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip tooling"
python -m pip install --upgrade pip wheel setuptools

echo "==> Installing CPU-only PyTorch"
pip install --index-url https://download.pytorch.org/whl/cpu torch

echo "==> Installing project dependencies (CPU set)"
pip install -r .cursor/requirements-cpu.txt

echo "==> Verifying imports"
python - <<'PY'
import functions, prompter, validator, utils, schema
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
tools = functions.get_openai_tools()
print(f"OK: imported core modules; {len(tools)} tools available")
PY

echo "==> Install complete. Activate with: source .venv/bin/activate"
