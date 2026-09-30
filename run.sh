#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"  # Activated environment, or an explicit Python executable.
"$PYTHON" -c 'import sys; print(f"Python: {sys.executable}\nEnvironment: {sys.prefix}", file=sys.stderr)'
exec "$PYTHON" -m fusioneval launch --config "$@"
