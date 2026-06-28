#!/usr/bin/env bash
#
# Build prebuilt PROPOSAL wheels for Colab (manylinux x86_64, CPython 3.11 + 3.12).
#
# WHY: PyPI ships PROPOSAL *source-only* (no wheels), and its classifiers stop at
# Python 3.11 while Colab now runs 3.12 -- so `pip install proposal` compiles from
# source (cmake + C++14 + pybind11; minutes, and can fail). Notebook 2's Part 1 only
# needs `import proposal`, so we install a prebuilt wheel from this directory instead
# (mirrors the committed siren-*.whl approach).
#
# MUST run on Linux x86_64 with Docker (cibuildwheel uses a manylinux container).
# It will NOT run on macOS/Apple Silicon -- run it on a Linux box or in CI, then
# commit the resulting wheels next to this script.
#
# Usage:
#   bash build_proposal_wheels.sh            # builds into ./  (this wheels/ dir)
#
# After it finishes you should have, e.g.:
#   proposal-7.6.2-cp311-cp311-manylinux_2_28_x86_64.whl
#   proposal-7.6.2-cp312-cp312-manylinux_2_28_x86_64.whl
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROPOSAL_VERSION="${PROPOSAL_VERSION:-7.6.2}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

python -m pip install --upgrade pip cibuildwheel >/dev/null

# Grab the PROPOSAL source distribution and unpack it.
python -m pip download "proposal==${PROPOSAL_VERSION}" --no-binary :all: --no-deps -d "$WORK"
tar -xzf "$WORK"/proposal-*.tar.gz -C "$WORK"
SRC="$(echo "$WORK"/proposal-*/)"

# Build manylinux wheels for the two CPython versions Colab may run.
CIBW_BUILD="cp311-manylinux_x86_64 cp312-manylinux_x86_64" \
CIBW_ARCHS_LINUX="x86_64" \
  python -m cibuildwheel --platform linux --output-dir "$HERE" "$SRC"

echo
echo "Built wheels in: $HERE"
ls -1 "$HERE"/proposal-*.whl
