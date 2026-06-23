# Lecture 3 — Simulating a neutrino event

Hands-on companion to *Neutrino Interactions, Simulation and Event Generation* (N. Kamp).
Students walk the full telescope pipeline and generate / display $\nu_\tau$ events:

```
FLUX → INTERACTION → PROPAGATION → LIGHT → DETECTOR → WEIGHTS
```

**[➡ Open `neutrino_sim_lecture3.ipynb` in Colab]** — replace `USER/REPO` below with your repo,
then use this badge link:
`https://colab.research.google.com/github/USER/REPO/blob/main/Lecture3_Simulation/neutrino_sim_lecture3.ipynb`

## Design philosophy: it must not break live

Most telescope tools are compiled C++ with Python bindings — the fragile part in Colab. So **every
pipeline stage falls back to a cached data file**: if a live install fails, the plots and the final
event display still work. Stage-by-stage reality:

| Stage | Live in Colab? | Fallback |
|---|---|---|
| Flux (atmo + astro) | ✅ numpy only — reads SIREN's flux table | `data/atmo_flux_siren.npz` |
| Interaction | ⚠️ `siren` if pinned/tested | `data/siren_nutau_injection.parquet` → synthetic |
| Propagation | ✅ `proposal` (analytic demo) | analytic curve |
| Light + detector | ⚠️ `prometheus`+`fennel` (CPU parametric) | `data/prometheus_nutau_example.parquet` → synthetic double-bang |
| Weights | ✅ pure numpy | reuses the at-detector flux table |

**Flux:** SIREN ships atmospheric tables (surface **and** at-detector, the latter with nuSQuIDS
oscillation + Earth absorption already folded in) on Zenodo record `20129082`. `make_cache.py` extracts
a small `.npz` from that 1.2 GB archive once; the notebook only ever downloads the small file and reads
numpy — so **`daemonflux`/`nuflux`/`nuSQuIDS` are gone entirely**. Plotting surface vs at-detector (§1b)
shows what Earth propagation does without running anything.

`GENIE`, `GEANT4`, `Tauola`, `pythia8` are **discuss-and-show**, never installed live.

## Running on Google Colab

**The runtime:** Ubuntu + Python 3.11/3.12 (check `!python --version`; Colab bumps it), with
`numpy/scipy/matplotlib/pandas` pre-installed, `pip`/`apt` (sudo) available, a free T4 GPU you don't
need here, ~12 GB RAM, and an **ephemeral disk wiped every session** — so the setup cell must re-run
each time. Sessions also time out (~90 min idle).

**The catch — these are compiled C++ tools:**

| Package | PyPI install | Implication |
|---|---|---|
| `siren` | wheel exists **but lags the git repo** | need `pip install git+…SIREN` → **builds from source** (`cmake` + compiler, ~minutes) |
| `proposal` | source-only (no wheel) | `pip install proposal` **compiles** (~5–10 min, can fail) |
| `prometheus` | heavy; depends on PROPOSAL | don't build live |

**So the lecture never builds these live.** Every stage reads a small cached file and the notebook
just analyses/displays it. Flux needs no package at all (numpy on a SIREN-derived `.npz`).

**To offer SIREN live (optional, for §2):** build a wheel **once** and host it, then the setup cell
`pip install`s the URL in seconds — far better UX than compiling in front of students:

```bash
# in a throwaway Colab session, matching the lecture-day runtime:
!pip wheel "git+https://github.com/Harvard-Neutrino/SIREN.git@<tag>" -w /content/wheels
# download the resulting siren-*.whl, attach it to your GitHub release, then in the
# notebook setup cell set:  SIREN_WHEEL_URL = "https://github.com/USER/REPO/releases/.../siren-*.whl"
```

Pin the git `<tag>`/commit so the wheel is reproducible, and rebuild it if Colab's Python changes.

## Files

- `neutrino_sim_lecture3.ipynb` — the lecture notebook (22 cells, pipeline-ordered, with 🔧 exercises).
- `src/helpers.py` — cache loading, the 3-D event display, and the synthetic double-bang placeholder.
- `make_cache.py` — **run once** in a full-toolchain environment to build the real cached data files.
- `requirements.txt` — pin these to versions you tested the week of the lecture.

## Before the lecture (checklist)

1. **Build the cache.** On a machine with the real toolchain: `python make_cache.py`. Fill in the
   `TODO`/`NotImplementedError` bodies with your standard nuSQuIDS / SIREN / Prometheus configs.
2. **Host the cache.** Upload `data/*.{npz,parquet}` to a GitHub release (or bucket/Zenodo) and set
   `CACHE_BASE_URL` in `src/helpers.py` (or the `NUSIM_CACHE_URL` env var).
3. **Pin versions.** Do a clean Colab install, `pip freeze`, and paste exact versions into `requirements.txt`.
4. **Dry run the morning of.** Colab base images drift; run the whole notebook once, top to bottom.

Until step 1 is done the notebook still runs end-to-end on **synthetic placeholder data** (clearly
labelled), so you can develop and demo the flow immediately.

## Local run

```bash
pip install -r requirements.txt
jupyter lab neutrino_sim_lecture3.ipynb   # helpers auto-found via ./src
```
