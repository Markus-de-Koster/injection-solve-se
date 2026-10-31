#!/bin/bash
# Fetch the published code of Mack et al. (2024) at its pinned commit and install the
# environment it runs in (TensorFlow 2.13 with CUDA 11.8 / cuDNN 8.6, the versions the
# authors pin). Nothing of their code is copied into this repository.
#
# usage: bash mack/setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_URL=https://github.com/th-koeln-iet/pqse_concept_pann
COMMIT=6d0b1e96bbb5b1231dd872b002780acf587482db
DEST="$HERE/external/pqse_concept_pann"
if [ ! -d "$DEST/.git" ]; then
  git clone --quiet "$REPO_URL" "$DEST"
fi
git -C "$DEST" -c advice.detachedHead=false checkout --quiet "$COMMIT"
test "$(git -C "$DEST" rev-parse HEAD)" = "$COMMIT"
echo "Mack et al. code at $COMMIT in $DEST"
pixi install --manifest-path "$HERE/pixi.toml"
