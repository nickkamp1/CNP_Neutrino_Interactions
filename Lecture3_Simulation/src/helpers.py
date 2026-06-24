"""
Helper functions for the Lecture 3 neutrino-simulation notebook.

The notebook is meant to run in Google Colab where several of the heavy,
compiled tools (Prometheus, nuSQuIDS, ...) may or may not install cleanly.
Every stage of the pipeline therefore falls back to a *cached* data file so
the lecture can continue even if a live install breaks. These helpers hide
that try-live / else-load-cache logic so the notebook cells stay short.

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
# release, a public bucket, a Zenodo record, ...). The notebook downloads from
# here on Colab; locally it just reads ./data.
CACHE_BASE_URL = os.environ.get(
    "NUSIM_CACHE_URL",
    "https://github.com/USER/REPO/releases/download/lecture3-data/",
)

# Resolve a local data directory whether we're in the repo or on Colab.
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("NUSIM_DATA_DIR", os.path.join(_HERE, "..", "data"))
DATA_DIR = os.path.abspath(DATA_DIR)


def cached_path(filename: str, download: bool = True) -> str:
    """Return a local path to ``filename`` in the data dir, downloading it from
    CACHE_BASE_URL if it is missing and ``download`` is True."""
    os.makedirs(DATA_DIR, exist_ok=True)
    local = os.path.join(DATA_DIR, filename)
    if os.path.exists(local):
        return local
    if not download:
        raise FileNotFoundError(local)
    url = CACHE_BASE_URL + filename
    print(f"[cache] downloading {filename} ...")
    try:
        urllib.request.urlretrieve(url, local)
        print(f"[cache] saved to {local}")
    except Exception as exc:  # noqa: BLE001 - lecture robustness over purity
        raise FileNotFoundError(
            f"Could not fetch {filename} from {url}. "
            f"Drop the file into {DATA_DIR} manually, or run make_cache.py."
        ) from exc
    return local


def have(module_name: str) -> bool:
    """True if a module is importable. Used to decide live-vs-cached per stage."""
    import importlib.util

    return importlib.util.find_spec(module_name) is not None


# ---------------------------------------------------------------------------
# Flux helpers
# ---------------------------------------------------------------------------
def load_atmo_flux(filename="atmo_flux_siren.npz"):
    """Load SIREN's atmospheric flux tables (extracted by make_cache.build_flux_subset).

    Returns a dict with energy_gev, coszen, flux_surface, flux_detector
    (shape [coszen, energy, flavor, nu/nubar]) and flavors. The *detector*
    table already includes nuSQuIDS oscillation + Earth absorption, so plotting
    surface vs detector shows exactly what propagation through the Earth does --
    no nuSQuIDS install required.

    If the cached file is missing, returns a clearly-labelled SYNTHETIC table
    (meta['synthetic'] == True) so the flux/weighting cells still render during
    development. Swap in the real SIREN table for the lecture.
    """
    try:
        d = np.load(cached_path(filename))
        out = {k: d[k] for k in d.files}
        out.setdefault("synthetic", np.array(False))
        return out
    except FileNotFoundError:
        return _synthetic_atmo_table()


def _synthetic_atmo_table():
    """Teaching placeholder atmospheric flux with the same schema as the real
    SIREN tables. Captures the two qualitative Earth effects so section 1b/6
    look right: nu_tau appearance at low E (oscillation) and up-going absorption
    at high E. NOT a real flux."""
    E = np.logspace(2, 7, 100)                     # 100 GeV - 10 PeV
    cz = np.linspace(-1, 0.2, 40)                  # up-going .. just above horizon
    flavors = np.array(["nue", "numu", "nutau"])
    EE = E[None, :]

    # --- surface flux (conventional, falling ~E^-3.7), zenith-flat-ish ---
    numu = 1.0 * (E / 1e3) ** -3.7
    surf = np.zeros((cz.size, E.size, 3, 2))
    surf[:, :, 1, :] = (numu[None, :] * 0.5)[..., None]      # numu, split nu/nubar
    surf[:, :, 0, :] = surf[:, :, 1, :] * 0.05               # nue ~ 5% of numu
    surf[:, :, 2, :] = surf[:, :, 1, :] * 1e-4               # nutau ~ prompt-only

    # --- at detector: + oscillation appearance of nutau, + Earth absorption ---
    det = surf.copy()
    # nu_tau appearance: peaks ~25 GeV, only matters for up-going (long baseline)
    appear = np.exp(-0.5 * ((np.log10(EE) - np.log10(25.0)) / 0.5) ** 2)
    appear = appear * np.clip(-cz[:, None], 0, 1)           # up-going only
    det[:, :, 2, :] += (appear * numu[None, :] * 0.5)[..., None]
    # Earth absorption: up-going, high energy (sigma ~ E), depends on path length
    pathlen = np.clip(-cz[:, None], 0, 1)                   # 0 horizon .. 1 straight up
    absorb = np.exp(-(EE / 1e6) * 2.0 * pathlen)
    det *= absorb[..., None, None]

    return {
        "energy_gev": E, "coszen": cz, "flavors": flavors,
        "flux_surface": surf, "flux_detector": det,
        "synthetic": np.array(True),
    }


def astro_powerlaw(energy_gev, phi0=1.8e-18, gamma=2.5, e0=1.0e5):
    """IceCube-style single power-law astrophysical flux (per-flavor, per-nu+nubar).

    dPhi/dE = phi0 * (E / E0)^(-gamma),  units GeV^-1 cm^-2 s^-1 sr^-1.
    Defaults roughly match the IceCube diffuse muon-neutrino fit (E0 = 100 TeV).
    """
    energy_gev = np.asarray(energy_gev, dtype=float)
    return phi0 * (energy_gev / e0) ** (-gamma)


# ---------------------------------------------------------------------------
# Tau propagation / decay  (section 3)
# ---------------------------------------------------------------------------
TAU_MASS_GEV = 1.77686      # PDG tau mass
TAU_CTAU_M = 8.703e-5       # c * lifetime, metres (~87 microns)


def tau_mean_decay_length_m(e_tau_gev):
    """Lab-frame mean tau decay length L = (E/m) * c*tau, in metres
    (~50 m per PeV)."""
    e_tau_gev = np.asarray(e_tau_gev, float)
    return (e_tau_gev / TAU_MASS_GEV) * TAU_CTAU_M


def sample_tau_decay_length_m(e_tau_gev, size=None, rng=None):
    """Sample lab-frame tau decay lengths (exponential, mean = gamma*c*tau).

    Exact decay kinematics. PROPOSAL additionally folds in stochastic energy
    loss *before* the decay, which erodes this simple scaling at the highest
    energies (the tau loses energy, so it doesn't fly as far as gamma*c*tau).
    """
    rng = rng or np.random.default_rng()
    return rng.exponential(tau_mean_decay_length_m(e_tau_gev), size=size)


def double_bang_efficiency(e_tau_gev, d_min=10.0, d_max=1000.0):
    """Fraction of taus whose decay length lands in a resolvable window
    [d_min, d_max] m, from the exponential CDF: e^{-d_min/L} - e^{-d_max/L}."""
    lam = tau_mean_decay_length_m(e_tau_gev)
    return np.exp(-d_min / lam) - np.exp(-d_max / lam)


# ---------------------------------------------------------------------------
# Event displays  (the payoff at the end of the lecture)
# ---------------------------------------------------------------------------
def plot_event_display(hits, ax=None, title=None, geo=None, max_dots=4000):
    """3-D event display of photon hits on detector modules.

    Parameters
    ----------
    hits : structured object with fields x, y, z (module positions, metres),
           t (hit time, ns) and optionally npe (charge). A dict of arrays or a
           pandas/awkward-style object both work as long as these keys exist.
    geo  : optional (N,3) array of all module positions to draw as a faint
           grey detector outline.
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


# ---------------------------------------------------------------------------
# Placeholder event generator
# ---------------------------------------------------------------------------
# Until real Prometheus output is dropped into ./data, this produces a
# physically-motivated *fake* nu_tau "double-bang": two energy depositions
# (the CC vertex hadronic cascade and the tau-decay cascade) separated along
# the tau direction, each lighting up nearby modules with a time gradient.
# Clearly labelled as synthetic so nobody mistakes it for a real simulation.
def synthetic_double_bang(e_nu_gev=3.0e6, inelasticity=0.25, seed=0):
    """Return (hits_dict, geo, meta) approximating a nu_tau CC double-bang.

    This is a teaching placeholder, NOT a real simulation. Swap in a cached
    Prometheus event with load_prometheus_event() once you have one.
    """
    rng = np.random.default_rng(seed)

    # Build a small IceCube-like hexagonal string geometry.
    strings, dom_z = [], np.linspace(-450, 450, 40)
    for i in range(-3, 4):
        for j in range(-3, 4):
            sx = 125 * (i + 0.5 * (j % 2))
            sy = 108 * j
            strings.append((sx, sy))
    geo = np.array([[sx, sy, z] for (sx, sy) in strings for z in dom_z])

    # tau direction and the two vertices.
    direction = np.array([0.6, 0.2, -0.3])
    direction = direction / np.linalg.norm(direction)
    c = 0.299792458  # m/ns
    # tau decay length ~ (E_tau/m_tau)*c*tau ~ 50 m per PeV.
    e_tau = (1 - inelasticity) * e_nu_gev
    L = 50.0 * (e_tau / 1.0e6)  # metres
    v1 = np.array([0.0, 0.0, 50.0])
    v2 = v1 + L * direction

    def cascade(vertex, energy, t0):
        # Light up modules near the vertex; brightness ~ log(E), time ~ distance.
        d = np.linalg.norm(geo - vertex, axis=1)
        near = d < 250
        xx = geo[near]
        dist = d[near]
        npe = np.maximum(0, np.log10(energy) * 3 - 0.02 * dist) + rng.normal(0, 0.3, dist.shape)
        npe = np.clip(npe, 0, None)
        keep = npe > 0.2
        t = t0 + dist[keep] / (c / 1.33) + rng.normal(0, 5, keep.sum())  # n_ice~1.33
        return xx[keep], npe[keep], t

    x1, q1, t1 = cascade(v1, inelasticity * e_nu_gev, t0=0.0)
    x2, q2, t2 = cascade(v2, 0.5 * e_tau, t0=L / direction.dot(direction) / c)

    X = np.vstack([x1, x2])
    npe = np.concatenate([q1, q2])
    t = np.concatenate([t1, t2])
    hits = {"x": X[:, 0], "y": X[:, 1], "z": X[:, 2], "t": t, "npe": npe}
    meta = dict(e_nu_gev=e_nu_gev, inelasticity=inelasticity,
                e_tau_gev=e_tau, decay_length_m=L, vertices=(v1, v2),
                synthetic=True)
    return hits, geo, meta


def load_prometheus_event(filename="prometheus_nutau_example.parquet", event=0):
    """Load one event from a cached Prometheus output file into the hit dict
    format used by plot_event_display(). Expects columns for module position,
    hit time and (optionally) charge. Adjust the column names to match your
    Prometheus version's photon output schema."""
    import pandas as pd  # noqa: PLC0415

    path = cached_path(filename)
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
