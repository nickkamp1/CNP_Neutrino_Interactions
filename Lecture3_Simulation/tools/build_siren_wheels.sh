#!/usr/bin/env bash
# =============================================================================
# Build self-contained, Colab-ready manylinux wheels for a LOCAL SIREN checkout.
# =============================================================================
# Why this exists:
#   `pip install siren` pulls the released wheel from PyPI. Until the new version
#   is published you can build an equivalent wheel from a local source tree and
#   `pip install` THAT on Colab. A wheel built directly in a spack env (lienv) is
#   NOT Colab-portable: it is tagged for that env's Python ABI and dynamically
#   linked against spack-built libgsl/libcfitsio/libphotospline/libhdf5 at paths
#   that don't exist on Colab. This script instead replicates SIREN's own
#   cibuildwheel recipe INSIDE the manylinux_2_28 container, so:
#     * it builds with a modern gcc/cmake but targets old glibc (2.28) -> installs
#       on Colab (glibc 2.35);
#     * `auditwheel repair` vendors GSL + cfitsio INTO the wheel (photospline is
#       statically linked, hence --exclude); the result is self-contained.
#   The host only needs a working `podman` (or docker); the ancient host Python
#   and missing host cmake are irrelevant -- everything happens in the container.
#
# Usage:
#   bash build_siren_wheels.sh <SIREN_SRC> <OUTPUT_DIR> ["cp311 cp312"]
#
# Parallelism:
#   The build uses all cores of the host node by default (JOBS=nproc inside the
#   container). To pin it, set JOBS, e.g.:  JOBS=16 bash build_siren_wheels.sh ...
#   First make sure you ARE on a multi-core node -- on SLURM grab cores with e.g.
#   `salloc -p test -c 16 --mem 32G -t 0-02:00` (a podman container uses all host
#   cores unless you pass `--cpus N`).
#
# Example (the lienv SIREN source):
#   bash build_siren_wheels.sh \
#       /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/nkamp/LIV2/sources/SIREN \
#       ./wheels  "cp311 cp312"
#
# The ABI tag(s) MUST match Colab's Python -- check with `!python --version` in a
# Colab cell (then pip-install the matching wheel). Building extra ABIs just costs
# time. The manylinux_2_28 image ships cp38..cp313 under /opt/python.
#
# On Colab:
#   from google.colab import files; files.upload()         # upload the .whl
#   !pip install ./siren-0.0.3-cp311-cp311-manylinux_2_28_x86_64.whl
#   import siren                                            # pure-Py deps auto-resolve
# (Or stash the wheel in Google Drive and pip-install it from there each session.)
# =============================================================================
set -euo pipefail

SRC="${1:?path to SIREN source}"
OUT="${2:?output dir for wheels}"
PYTAGS="${3:-cp311 cp312}"                 # CPython ABIs to build (match Colab)
IMAGE="${IMAGE:-quay.io/pypa/manylinux_2_28_x86_64}"
ENGINE="${ENGINE:-podman}"                 # set ENGINE=docker to use docker

SRC="$(cd "$SRC" && pwd)"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"

# ---- inner script: runs INSIDE the manylinux container ----------------------
cat > "$OUT/_inner.sh" <<'INNER'
set -xeuo pipefail
PYTAGS="$1"

# 1) System dep that auditwheel will vendor into the wheel.
yum install -y gsl-devel

# 2) Work on a writable copy (source is mounted read-only; your repo is untouched).
cp -a /src /work
cd /work

# 3) Env expected by SIREN's before_all + pyproject (CMAKE_PREFIX_PATH default).
# CIBUILDWHEEL=1 is the critical flag the real cibuildwheel sets automatically and
# that SIREN's CMake reads from the environment (set(CIBUILDWHEEL "$ENV{CIBUILDWHEEL}")).
# It (a) skips the C++ unit-test targets, and (b) makes CMake use Python's
# Development.Module component so libSIREN.so is NOT linked against -lpython3.X --
# essential on manylinux, which ships no libpythonX.Y.so. Without it the build either
# compiles 250 test targets or fails at link with "cannot find -lpython3.11".
export CIBUILDWHEEL=1
export RUNNER_OS=Linux
export CI_INSTALL_PREFIX=/tmp/downloads/local
export CMAKE_PREFIX_PATH=$CI_INSTALL_PREFIX
export CFITSIOROOT=$CI_INSTALL_PREFIX
export LD_LIBRARY_PATH=$CI_INSTALL_PREFIX/lib:$CI_INSTALL_PREFIX/lib64:${LD_LIBRARY_PATH:-}

# Parallelism: CMAKE_BUILD_PARALLEL_LEVEL drives every `cmake --build` (cfitsio AND
# the SIREN build via scikit-build-core, any generator); MAKEFLAGS covers raw make.
# Ninja (SIREN's default) already uses all cores, but we set both to be explicit and
# to parallelise the cfitsio step too. JOBS is forwarded from the host (default nproc).
JOBS="${JOBS:-$(nproc)}"
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
export MAKEFLAGS="-j${JOBS}"
echo ">>> building with JOBS=${JOBS} (container sees $(nproc) cores)"

# 4) Build the external C deps once (cfitsio from source; GSL via yum above).
#    before_all calls `python`/`pip`; point them at a container CPython.
export PATH=/opt/python/cp311-cp311/bin:$PATH
python -m pip install -U pip build auditwheel
bash /work/tools/wheels/cibw_before_all.sh /work

# 5) Build + repair one wheel per requested CPython ABI.
for tag in $PYTAGS; do
    PYDIR=$(ls -d /opt/python/${tag}-${tag}* | head -1)
    echo "=== building SIREN for ${tag} (${PYDIR}) ==="
    rm -rf /tmp/raw; mkdir -p /tmp/raw
    "$PYDIR/bin/python" -m build --wheel --outdir /tmp/raw /work
    # Vendor GSL/cfitsio; exclude photospline (static), matching
    # pyproject [tool.cibuildwheel.linux].repair-wheel-command.
    auditwheel repair --exclude photospline -w /output /tmp/raw/*.whl
done

echo "=== finished manylinux wheels ==="
ls -la /output/*.whl
INNER

# ---- launch the container ----------------------------------------------------
echo ">>> building SIREN manylinux wheels via ${ENGINE} (${IMAGE}) for: ${PYTAGS}"
echo ">>> source : ${SRC}"
echo ">>> output : ${OUT}"
"$ENGINE" run --rm \
    -e JOBS="${JOBS:-}" \
    -v "$SRC":/src:ro \
    -v "$OUT":/output:z \
    "$IMAGE" \
    bash /output/_inner.sh "$PYTAGS"

rm -f "$OUT/_inner.sh"
echo ">>> DONE. Colab-ready wheels:"
ls -la "$OUT"/*.whl
