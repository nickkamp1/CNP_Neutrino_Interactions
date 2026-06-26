"""
Helper functions for the Lecture 3 neutrino-simulation notebooks.

Design: the notebooks use **real software** — either installed and run live, or
read from **cached files produced by running that software** (see make_cache.py).
There are NO synthetic / analytic stand-ins for data: if a required cache file is
missing and the tool isn't available, the loaders raise a clear error telling you
how to generate it. This keeps every plot physically faithful.

Author: scaffold for N. Kamp summer-school lecture 3.
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np

# ---------------------------------------------------------------------------
# Cache handling
# ---------------------------------------------------------------------------
# Point CACHE_BASE_URL at wherever you host the pre-generated files (a GitHub
# release, a public bucket, a Zenodo record, ...). If the files are committed to
# data/ (so a Colab `git clone` brings them) this is never used.
CACHE_BASE_URL = os.environ.get(
    "NUSIM_CACHE_URL",
    "https://github.com/USER/REPO/releases/download/lecture3-data/",
)

# Resolve a local data directory whether we're in the repo or on Colab.
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("NUSIM_DATA_DIR", os.path.join(_HERE, "..", "data"))
DATA_DIR = os.path.abspath(DATA_DIR)


def cached_path(filename: str, download: bool = True, hint: str = "") -> str:
    """Return a local path to ``filename`` in the data dir.

    Looks in data/ first; if missing, tries to download from CACHE_BASE_URL.
    Raises FileNotFoundError with an actionable message (``hint``) if neither
    works — we never substitute fake data.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    local = os.path.join(DATA_DIR, filename)
    if os.path.exists(local):
        return local

    tip = f"\n  -> {hint}" if hint else ""
    if not download:
        raise FileNotFoundError(f"Missing cached file: {local}{tip}")

    url = CACHE_BASE_URL + filename
    print(f"[cache] {filename} not in data/; trying {url}")
    try:
        urllib.request.urlretrieve(url, local)
        print(f"[cache] downloaded to {local}")
    except Exception as exc:  # noqa: BLE001
        # Clean up any partial file so a retry is clean.
        if os.path.exists(local):
            os.remove(local)
        raise FileNotFoundError(
            f"Could not find or download '{filename}'.\n"
            f"  Commit it to {DATA_DIR}, set CACHE_BASE_URL, or generate it.{tip}"
        ) from exc
    return local


def have(module_name: str) -> bool:
    """True if a module is importable. Used to decide live-vs-cached per stage."""
    import importlib
    import importlib.util

    importlib.invalidate_caches()  # so it sees packages pip-installed this session
    return importlib.util.find_spec(module_name) is not None


# ---------------------------------------------------------------------------
# Flux  (section 1)  -- SIREN's atmospheric tables
# ---------------------------------------------------------------------------
def load_atmo_flux(filename="atmo_flux_siren.npz"):
    """Load SIREN's atmospheric flux tables (extracted by make_cache.build_flux_subset).

    Returns a dict with energy_gev, coszen, flux_surface, flux_detector
    (shape [coszen, energy, flavor, nu/nubar]) and flavors. The *detector* table
    already includes nuSQuIDS oscillation + Earth absorption, so plotting surface
    vs detector shows what propagation through the Earth does — without running
    nuSQuIDS. Raises if the file is missing (run `python make_cache.py --flux`).
    """
    path = cached_path(filename, hint="generate with: python make_cache.py --flux")
    d = np.load(path)
    return {k: d[k] for k in d.files}


def astro_powerlaw(energy_gev, phi0=1.8e-18, gamma=2.5, e0=1.0e5):
    """IceCube-style single power-law astrophysical flux (per-flavor, per-nu+nubar).

    dPhi/dE = phi0 * (E / E0)^(-gamma),  units GeV^-1 cm^-2 s^-1 sr^-1.
    This is a *flux model* (not a data stand-in): the diffuse astro flux is
    well-described by a single power law. Defaults ~ IceCube diffuse fit.
    """
    energy_gev = np.asarray(energy_gev, dtype=float)
    return phi0 * (energy_gev / e0) ** (-gamma)


# ---------------------------------------------------------------------------
# Interaction  (section 2)  -- SIREN injection
# ---------------------------------------------------------------------------
def load_injection(filename="siren_nutau_injection.parquet"):
    """Load a cached SIREN nu_tau injection (make_cache.build_siren_events).

    Expected columns: energy_gev, bjorken_y, cos_zen, gen_weight (others ok).
    Raises if missing (run `python make_cache.py --siren`)."""
    import pandas as pd  # noqa: PLC0415

    path = cached_path(filename, hint="generate with: python make_cache.py --siren")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Propagation  (section 3)  -- PROPOSAL tau decays
# ---------------------------------------------------------------------------
TAU_MASS_GEV = 1.77686      # PDG tau mass
TAU_CTAU_M = 8.703e-5       # c * lifetime, metres (~87 microns)


def tau_mean_decay_length_m(e_tau_gev):
    """Lab-frame mean tau decay length L = (E/m) * c*tau, in metres (~50 m/PeV).

    Exact ONLY in the no-energy-loss limit; shown in section 3 as a labelled
    reference curve against the real PROPOSAL distribution (which includes the
    stochastic losses that bend it at the highest energies)."""
    e_tau_gev = np.asarray(e_tau_gev, float)
    return (e_tau_gev / TAU_MASS_GEV) * TAU_CTAU_M


def load_tau_decays(filename="tau_decays_proposal.parquet"):
    """Load cached PROPOSAL tau-propagation results (make_cache.build_tau_decays).

    Expected columns: energy_gev (initial tau energy) and decay_length_m (the
    distance at which PROPOSAL decayed the tau, after stochastic losses).
    Raises if missing (run `python make_cache.py --tau`)."""
    import pandas as pd  # noqa: PLC0415

    path = cached_path(filename, hint="generate with: python make_cache.py --tau")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Light + detector  (section 5 / notebook 3c)  -- Prometheus event displays
# ---------------------------------------------------------------------------
def plot_event_display(hits, ax=None, title=None, geo=None, max_dots=4000):
    """3-D event display of photon hits on detector modules.

    Parameters
    ----------
    hits : object with fields x, y, z (module positions, metres), t (hit time,
           ns) and optionally npe (charge). A dict of arrays works.
    geo  : optional (N,3) array of all module positions, drawn as a faint outline.
    """
    import matplotlib.pyplot as plt

    x = np.asarray(hits["x"], float)
    y = np.asarray(hits["y"], float)
    z = np.asarray(hits["z"], float)
    t = np.asarray(hits["t"], float)
    q = np.asarray(hits.get("npe", np.ones_like(x)), float)

    if x.size > max_dots:  # keep the display responsive for big events
        idx = np.random.default_rng(0).choice(x.size, max_dots, replace=False)
        x, y, z, t, q = x[idx], y[idx], z[idx], t[idx], q[idx]

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

    if geo is not None:
        geo = np.asarray(geo, float)
        ax.scatter(geo[:, 0], geo[:, 1], geo[:, 2], s=1, c="0.85", alpha=0.3)

    size = 10 + 60 * (q / (q.max() + 1e-9))
    sc = ax.scatter(x, y, z, c=t, s=size, cmap="rainbow_r", alpha=0.8)
    cb = ax.get_figure().colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
    cb.set_label("hit time [ns]  (early = red)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    if title:
        ax.set_title(title)
    return ax


def load_prometheus_event(filename="prometheus_nutau_example.parquet", event=0):
    """Load one event from a cached Prometheus output file into the hit dict
    format used by plot_event_display(). Adjust the column names to match your
    Prometheus version's photon output schema. Raises if the file is missing
    (run `python make_cache.py --events`)."""
    import pandas as pd  # noqa: PLC0415

    path = cached_path(filename, hint="generate with: python make_cache.py --events")
    df = pd.read_parquet(path)
    ev = df[df["event_id"] == sorted(df["event_id"].unique())[event]]
    hits = {
        "x": ev["sensor_x"].to_numpy(),
        "y": ev["sensor_y"].to_numpy(),
        "z": ev["sensor_z"].to_numpy(),
        "t": ev["t"].to_numpy(),
        "npe": ev.get("npe", np.ones(len(ev))),
    }
    geo = df[["sensor_x", "sensor_y", "sensor_z"]].drop_duplicates().to_numpy()
    return hits, geo
