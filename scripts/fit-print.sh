#!/usr/bin/env bash
# Preview the memory footprint of the llama-cpp compose service's current settings
# without loading real model weights. Reads flags straight from docker-compose.yml
# (via `docker compose config`) so this never goes stale, and drops the service's own
# verbosity so INFO/TRACE startup logging doesn't bury the summary.
#
# Any arguments given here are appended last, so they can override anything above,
# e.g. `./scripts/fit-print.sh --verbosity 4` to bring the detailed tables back.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SERVICE=llama-cpp

mapfile -t compose_args < <(docker compose config --format json | jq -r --arg svc "$SERVICE" '.services[$svc].command[1:] | .[]')

docker compose run --rm --no-deps --entrypoint /app/llama-fit-params "$SERVICE" \
    "${compose_args[@]}" --verbosity 2 --fit-print on "$@"
