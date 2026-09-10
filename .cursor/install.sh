#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the bone-marrow project.
# Recreates the Python 3.10 uv environment defined by pyproject.toml + uv.lock.
set -euo pipefail

# System libraries needed to build Python deps from source.
# scikit-sparse (via phlowerpy) needs SuiteSparse headers (cholmod.h).
if ! dpkg -s libsuitesparse-dev >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends build-essential libsuitesparse-dev
fi

# Install uv (the pinned package manager) if it is not already available.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Build .venv (Python 3.10, pinned by .python-version) exactly from the lockfile.
uv sync --frozen

echo "install.sh: environment ready. Activate with: source .venv/bin/activate"
