#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_BACKEND="${TORCH_BACKEND:-cu128}"  # Set cpu or another uv-supported CUDA backend if needed.
# Re-running preserves the existing environment; use a new directory to change Python.
[[ -x "$VENV_DIR/bin/python" ]] || uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
uv pip install --python "$VENV_DIR/bin/python" --torch-backend "$TORCH_BACKEND" -r requirements.txt -e .
printf 'Activate with: source %q/bin/activate\n' "$VENV_DIR"
