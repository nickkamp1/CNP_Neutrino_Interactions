# Lecture 3 — Simulating a neutrino event

Hands-on companion to *Neutrino Interactions, Simulation and Event Generation* (N. Kamp).
Students walk the full telescope pipeline and generate / display $\nu_\tau$ events:

```
FLUX → INTERACTION → PROPAGATION → LIGHT → DETECTOR → WEIGHTS
```

It's split into **three notebooks**, one per tool, runnable independently:

| # | Notebook | Stages | Open in Colab |
|---|---|---|---|
| a | `1_SIREN_flux_injection_weights.ipynb` | flux · injection · weights | `colab.research.google.com/github/USER/REPO/blob/main/Lecture3_Simulation/1_SIREN_flux_injection_weights.ipynb` |
| b | `2_PROPOSAL_tau_propagation.ipynb` | tau propagation / decay | `…/2_PROPOSAL_tau_propagation.ipynb` |
| c | `3_Prometheus_light_and_event_display.ipynb` | light · detector · displays | `…/3_Prometheus_light_and_event_display.ipynb` |

Replace `USER/REPO` with your repository in both the badge links **and** the `REPO_URL` in each
notebook's Setup cell (see next section).

## Colab can't see `src/` — the clone fix

When you open a notebook through Colab's **GitHub browser**, Colab downloads *only that one `.ipynb`*
into a fresh VM — it does **not** clone the repo, so `src/helpers.py` and `data/` are absent and
`import helpers` fails. Each notebook's first **Setup** cell fixes this: on Colab it `git clone`s the
repo into the runtime and adds `<repo>/Lecture3_Simulation/src` to `sys.path`. You must:

1. Set `REPO_URL` in the Setup cell to your repository (and make the repo public, or clone with a token).
2. **Commit the small cache files** (`data/*.npz`, `data/*.parquet`) so the clone brings them. They're
   small; the 1.2 GB SIREN flux archive is *not* committed (`make_cache.py` extracts a small `.npz`).
   Alternatively, host them and set `CACHE_BASE_URL` in `helpers.py` — `helpers.cached_path` downloads
   on demand. If neither is present, the notebooks fall back to labelled **synthetic** data.

Running locally needs none of this — the Setup cell finds `./src` automatically.

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

- `1_SIREN_…`, `2_PROPOSAL_…`, `3_Prometheus_…` `.ipynb` — the three lecture notebooks (each starts
  with a self-contained Setup cell and 🔧 exercises; runnable independently).
- `src/helpers.py` — flux/tau/cache helpers, the 3-D event display, and synthetic fallbacks.
- `make_cache.py` — **run once** in a full-toolchain environment to build the real cached data files.
- `requirements.txt` — pin these to versions you tested the week of the lecture.

## Before the lecture (checklist)

1. **Build the cache.** On a machine with the real toolchain: `python make_cache.py`. Fill in the
   `TODO`/`NotImplementedError` bodies with your standard nuSQuIDS / SIREN / Prometheus configs.
2. **Ship the cache.** Either commit `data/*.{npz,parquet}` so the Colab clone brings them, or upload
   them to a GitHub release (or bucket/Zenodo) and set `CACHE_BASE_URL` in `src/helpers.py`
   (or the `NUSIM_CACHE_URL` env var).
3. **Set `REPO_URL`** in each notebook's Setup cell to your repo.
4. **Pin versions.** Do a clean Colab install, `pip freeze`, and paste exact versions into `requirements.txt`.
5. **Dry run the morning of.** Colab base images drift; run each notebook once, top to bottom.

Until step 1 is done the notebooks still run end-to-end on **synthetic placeholder data** (clearly
labelled), so you can develop and demo the flow immediately.

## Local run

```bash
pip install -r requirements.txt
jupyter lab 1_SIREN_flux_injection_weights.ipynb   # helpers auto-found via ./src
```
