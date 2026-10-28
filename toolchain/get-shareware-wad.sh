#!/usr/bin/env bash
# Fetch the shareware doom1.wad (v1.9) from Debian's non-free archive (package doom-wad-shareware) into wad-src/.
# The shareware licence allows redistribution only unmodified: keep wad-src/ and anything generated from it out of git.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); mkdir -p "$ROOT/wad-src"
docker run --rm -v "$ROOT/wad-src:/o" debian:bookworm-slim bash -c "
  sed -i 's/Components: main/Components: main contrib non-free/' /etc/apt/sources.list.d/debian.sources
  apt-get update -qq && cd /tmp && apt-get download -qq doom-wad-shareware && dpkg-deb -x doom-wad-shareware_*.deb x
  cp x/usr/share/games/doom/doom1.wad /o/ && chown $(id -u):$(id -g) /o/doom1.wad"
echo "$(md5sum "$ROOT/wad-src/doom1.wad")   (shareware v1.9 is f0cefca49926d00903cf57551d901abe)"
