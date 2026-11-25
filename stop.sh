#!/usr/bin/env bash
# Stop every rank (exact container name).
set -euo pipefail
source "$(dirname "$0")/config.sh"
docker rm -f $CONTAINER >/dev/null 2>&1 && echo "stopped here" || true
for w in $WORKERS; do ssh -n -o BatchMode=yes "$w" "docker rm -f $CONTAINER >/dev/null 2>&1 && echo stopped on $w || true"; done
