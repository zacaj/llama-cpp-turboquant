#!/bin/bash
# Convenience wrapper for bench-filter.py that runs it through docker if needed.
# Can be run from the repo root or from inside the container.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Check if we're in the llama-cpp docker container by looking for /build (build directory)
if [[ -f /build/bin/llama-server ]]; then
  # Inside container, run directly
  exec python3 "${SCRIPT_DIR}/scripts/bench-filter.py" "$@"
else
  # Outside container, run via docker compose
  cd "${SCRIPT_DIR}"
  exec docker compose exec llama-cpp python3 /workspace/scripts/bench-filter.py "$@"
fi
