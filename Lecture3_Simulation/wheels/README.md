# Prebuilt wheels for Colab

The Lecture-3 notebooks install heavy native packages from prebuilt wheels here so
Colab doesn't have to compile them (which is slow and version-fragile).

| Package | Used by | Why a wheel |
|---|---|---|
| `siren-*.whl` | Notebook 1 (flux → injection → weights) | SIREN has no PyPI release |
| `proposal-*.whl` | Notebook 2, Part 1 (live τ/μ propagation) | PyPI PROPOSAL is **source-only**; its classifiers stop at Py 3.11 while Colab runs 3.12, so `pip install proposal` builds from source (minutes, can fail) |

Wheels are tagged by CPython version (`cp311`, `cp312`); the Setup cell installs the
one matching the running kernel. If no matching `proposal` wheel is present the Setup
cell **falls back** to a source `pip install proposal`, so the notebook still works.

## Build scripts

Both build scripts live here; each needs a Linux x86_64 host with a container engine
(a manylinux image — neither runs on macOS).

| Script | Builds | Notes |
|---|---|---|
| `build_proposal_wheels.sh` | `proposal-*.whl` (cp311, cp312) | `bash build_proposal_wheels.sh` — needs Docker; downloads the PyPI sdist and builds it via cibuildwheel |
| `build_siren_wheels.sh` | `siren-*.whl` (cp311, cp312) | `bash build_siren_wheels.sh <SIREN_SRC> <OUTPUT_DIR> ["cp311 cp312"]` — needs podman/docker; builds a local SIREN checkout and `auditwheel repair`s it |

Commit the resulting wheels next to the scripts.
