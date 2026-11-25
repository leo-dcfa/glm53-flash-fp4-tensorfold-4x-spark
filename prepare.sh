#!/usr/bin/env bash
# Fetch and verify the TensorFold release on this node, then copy this recipe (with the unpacked release) to every worker.
set -euo pipefail
source "$(dirname "$0")/config.sh"; cd "$HERE"; mkdir -p dist
[ -f dist/$TF_ARCHIVE ] || curl -fsSL -o dist/$TF_ARCHIVE "$TF_URL"
echo "$TF_SHA256  dist/$TF_ARCHIVE" | sha256sum -c -
[ -x "$TF_DIR/bin/tensorfold-native" ] || tar xzf dist/$TF_ARCHIVE -C dist
(cd "$TF_DIR" && sha256sum -c --quiet SHA256SUMS)
for w in $WORKERS; do
  rsync -a --delete --exclude .git -e "ssh -o BatchMode=yes" "$HERE/" "$w:$HERE/"
  ssh -n -o BatchMode=yes "$w" "cd '$TF_DIR' && sha256sum -c --quiet SHA256SUMS && docker image inspect $BASE_IMAGE >/dev/null && test -d '$MODEL_DIR'" && echo "ready: $w"
done
