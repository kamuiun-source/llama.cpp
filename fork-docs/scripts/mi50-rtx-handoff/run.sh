#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
exec timeout --signal=TERM --kill-after=5s "${HANDOFF_TIMEOUT:-90s}" stdbuf -oL -eL ./handoff "$@"
