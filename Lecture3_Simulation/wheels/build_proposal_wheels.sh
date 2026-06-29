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
#   proposal-7.6.2-cp311-cp311-manylinux_2_34_x86_64.whl
#   proposal-7.6.2-cp312-cp312-manylinux_2_34_x86_64.whl
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

# PROPOSAL's binding CMake over-requests Python's full `Development` component, which
# expands to Development.Embed -> needs a shared libpython that manylinux Pythons do
# NOT ship, so cmake reports "Could NOT find Python". A pybind11 extension only needs
# Development.Module (headers, no libpython to link). Patch the find_package() calls
# (src/pyPROPOSAL/CMakeLists.txt and any others) before cibuildwheel mounts the source.
grep -rl 'Interpreter Development REQUIRED' "$SRC" \
  | xargs -r sed -i 's/Interpreter Development REQUIRED/Interpreter Development.Module REQUIRED/'

# Build manylinux wheels for the two CPython versions Colab may run.
# cibuildwheel (linux) bind-mounts its WORKING DIRECTORY into the manylinux
# container, so the package source must live INSIDE the cwd. We therefore run it
# from within the unpacked sdist (package_dir defaults to ".") and send the
# finished wheels back to this wheels/ dir via --output-dir.
#
# Image: PROPOSAL pulls boost via conan, and conan's prebuilt `b2` binary needs
# GLIBC up to 2.34 -- too new for the default manylinux2014 (glibc 2.17) and even
# manylinux_2_28. So build in manylinux_2_34 (glibc 2.34); the wheel still installs
# on Colab (glibc 2.35). We pass the FULLY-QUALIFIED image (the short
# "manylinux_2_34" alias isn't resolved by older cibuildwheel and podman then tries
# to pull docker.io/library/manylinux_2_34, which 404s). Override to change it.
( cd "$SRC" && \
  CIBW_BUILD="cp311-manylinux_x86_64 cp312-manylinux_x86_64" \
  CIBW_ARCHS_LINUX="x86_64" \
  CIBW_MANYLINUX_X86_64_IMAGE="${CIBW_MANYLINUX_X86_64_IMAGE:-quay.io/pypa/manylinux_2_34_x86_64}" \
  python -m cibuildwheel --platform linux --output-dir "$HERE" )

echo
echo "Built wheels in: $HERE"
ls -1 "$HERE"/proposal-*.whl
