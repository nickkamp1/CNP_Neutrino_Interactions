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
# Light + detector  (Part 3)  -- Prometheus event displays
# ---------------------------------------------------------------------------
# The REAL Prometheus output is an *awkward* array (one record per event) with
# two fields:
#   mc_truth : {initial_state_type/energy/x/y/z, final_state_*, interaction,
#               bjorken_x/y, column_depth, ...}        (the MC truth of the event)
#   photons  : {sensor_pos_x, sensor_pos_y, sensor_pos_z, string_id, sensor_id,
#               t, id_idx}                              (one entry PER detected photon)
# So each *photon* already carries the (x,y,z) of the module it hit -- there is NO
# separate sensor_x/y/z truth table and NO per-event "event_id" column. The full
# detector geometry (every module, hit or not) lives in the .geo file referenced
# by the embedded `config_prometheus` JSON metadata.
#
# An event display wants ONE dot per *module* (not per photon): we group the
# photons by (string_id, sensor_id), count them (the NPE / total charge proxy),
# and take the earliest arrival time per module.

# PDG codes -> short human label, for titling the four IceCube_HE signatures.
_PDG_LABEL = {
    12: "nu_e", -12: "nubar_e",
    14: "nu_mu", -14: "nubar_mu",
    16: "nu_tau", -16: "nubar_tau",
    11: "e-", -11: "e+",
    13: "mu-", -13: "mu+",
    15: "tau-", -15: "tau+",
}


def read_geo_file(geo_path):
    """Parse a Prometheus detector `.geo` file into an (N, 3) array of module xyz.

    The `.geo` format is a small header (``### Metadata ###`` / ``### Modules ###``)
    followed by whitespace rows ``x y z string_id sensor_id`` (metres). We only
    need the positions here, to draw the faint full-detector outline behind an
    event. Returns ``None`` if the path is missing so callers can degrade
    gracefully (display the hit modules without the outline)."""
    if not geo_path or not os.path.exists(geo_path):
        return None
    xyz = []
    started = False
    with open(geo_path) as fh:
        for line in fh:
            if line.startswith("### Modules"):
                started = True
                continue
            if not started:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    xyz.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    continue
    return np.asarray(xyz, float) if xyz else None


def _resolve_geo_path(config_geo_path):
    """Find a usable detector .geo file.

    Prefers the absolute path recorded in the parquet's `config_prometheus`
    metadata; if that machine path isn't present (e.g. on Colab), falls back to
    the same-named file in an installed/cloned Prometheus `resources/geofiles/`.
    Returns a path string or None."""
    if config_geo_path and os.path.exists(config_geo_path):
        return config_geo_path

    base = os.path.basename(config_geo_path) if config_geo_path else "icecube.geo"
    candidates = []
    # 1) an importable prometheus package -> its bundled resources
    try:
        import importlib.util  # noqa: PLC0415

        spec = importlib.util.find_spec("prometheus")
        if spec and spec.origin:
            pkg_dir = os.path.dirname(spec.origin)
            candidates.append(os.path.join(pkg_dir, "..", "resources", "geofiles", base))
            candidates.append(os.path.join(pkg_dir, "resources", "geofiles", base))
    except Exception:  # noqa: BLE001
        pass
    # 2) a cloned prometheus checkout sitting next to / under the runtime
    for root in (".", "prometheus", "REPO/prometheus", os.path.join(DATA_DIR, "..")):
        candidates.append(os.path.join(root, "resources", "geofiles", base))
    # 3) committed alongside the data
    candidates.append(os.path.join(DATA_DIR, "geofiles", base))
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


def load_prometheus_event(path, event="brightest", geo=True):
    """Load one event from a real Prometheus parquet into the display format.

    Parameters
    ----------
    path : str
        Path to a ``Generation_*_photons.parquet`` file (the real Prometheus
        output). Read with ``awkward``; needs ``pyarrow``.
    event : int or "brightest"
        Event index, or "brightest" to auto-pick the event with the most photons
        (the visually clearest one for a display). Default "brightest".
    geo : bool
        If True, also load the full detector geometry from the `.geo` file named
        in the parquet's ``config_prometheus`` metadata (or a bundled fallback),
        for the faint outline. If the file can't be found, ``geo`` is returned as
        None and only the hit modules are drawn.

    Returns
    -------
    hits : dict with arrays x, y, z (module positions, m), t (earliest hit time
           per module, ns), npe (photon count per module = charge proxy).
    geo  : (N, 3) array of all module positions, or None.
    info : dict with mc-truth summary (initial_state_type/label/energy, ...).

    Backward-compatible note: the old signature took a cache filename + integer
    event and returned (hits, geo). This version returns a third ``info`` dict;
    older two-tuple unpacking still works if you ignore the extra value via
    ``hits, geo, *_ = load_prometheus_event(...)``.
    """
    import awkward as ak  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    arr = ak.from_parquet(path)

    nph = ak.to_numpy(ak.num(arr.photons.t))
    if event == "brightest":
        idx = int(np.argmax(nph))
    else:
        idx = int(event)
    rec = arr[idx]

    sx = ak.to_numpy(rec.photons.sensor_pos_x).astype(float)
    sy = ak.to_numpy(rec.photons.sensor_pos_y).astype(float)
    sz = ak.to_numpy(rec.photons.sensor_pos_z).astype(float)
    tt = ak.to_numpy(rec.photons.t).astype(float)
    sid = ak.to_numpy(rec.photons.string_id).astype(np.int64)
    did = ak.to_numpy(rec.photons.sensor_id).astype(np.int64)

    if sx.size == 0:
        raise ValueError(
            f"Event {idx} in {path} has no photon hits "
            f"(of {len(arr)} events, max hits = {int(nph.max())}). "
            "Pick another event or use event='brightest'."
        )

    # Collapse photons -> one row per module: charge = #photons, time = earliest.
    key = sid.astype(np.int64) * 100000 + did.astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    npe = np.bincount(inv).astype(float)
    # earliest arrival per module
    order = np.argsort(tt)
    t_first = np.full(uniq.size, np.inf)
    seen = np.zeros(uniq.size, bool)
    for j in order:
        m = inv[j]
        if not seen[m]:
            t_first[m] = tt[j]
            seen[m] = True
    # representative module position (first photon's position for that module)
    first_photon = np.zeros(uniq.size, dtype=int)
    seen2 = np.zeros(uniq.size, bool)
    for j in range(inv.size):
        m = inv[j]
        if not seen2[m]:
            first_photon[m] = j
            seen2[m] = True
    hits = {
        "x": sx[first_photon],
        "y": sy[first_photon],
        "z": sz[first_photon],
        "t": t_first,
        "npe": npe,
    }

    # MC truth summary
    pdg = int(rec.mc_truth.initial_state_type)
    info = {
        "event_index": idx,
        "n_photons": int(nph[idx]),
        "n_modules_hit": int(uniq.size),
        "initial_state_type": pdg,
        "initial_state_label": _PDG_LABEL.get(pdg, str(pdg)),
        "initial_state_energy_gev": float(rec.mc_truth.initial_state_energy),
        "interaction": int(rec.mc_truth.interaction),
    }

    geo_arr = None
    if geo:
        cfg_geo = None
        try:
            import json  # noqa: PLC0415

            meta = pq.read_metadata(path).metadata or {}
            raw = meta.get(b"config_prometheus")
            if raw is not None:
                cfg = json.loads(raw)
                cfg_geo = cfg.get("detector", {}).get("geo file")
        except Exception:  # noqa: BLE001
            cfg_geo = None
        geo_path = _resolve_geo_path(cfg_geo)
        geo_arr = read_geo_file(geo_path)

    return hits, geo_arr, info


def plot_event_display(hits, ax=None, title=None, geo=None, max_dots=4000,
                       color_by="t"):
    """3-D event display of the detector modules hit in one event.

    Parameters
    ----------
    hits : dict of arrays with fields x, y, z (module positions, m), t (hit time,
           ns) and optionally npe (charge / photon count per module).
    geo  : optional (N, 3) array of ALL module positions, drawn as a faint outline
           so the event sits inside the detector for context.
    color_by : "t" (default; early=red, late=blue) or "npe" (charge map).
    max_dots : cap on plotted modules for responsiveness.
    """
    import matplotlib.pyplot as plt

    x = np.asarray(hits["x"], float)
    y = np.asarray(hits["y"], float)
    z = np.asarray(hits["z"], float)
    t = np.asarray(hits["t"], float)
    q = np.asarray(hits.get("npe", np.ones_like(x)), float)

    if x.size > max_dots:  # keep the display responsive for big events
        # keep the brightest modules -- they define the topology
        keep = np.argsort(q)[::-1][:max_dots]
        x, y, z, t, q = x[keep], y[keep], z[keep], t[keep], q[keep]

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

    if geo is not None:
        geo = np.asarray(geo, float)
        ax.scatter(geo[:, 0], geo[:, 1], geo[:, 2], s=1, c="0.85",
                   alpha=0.25, depthshade=False)

    # marker size ~ sqrt(charge) so a few huge modules don't swamp the rest
    size = 8 + 80 * np.sqrt(q / (q.max() + 1e-9))
    if color_by == "npe":
        c, cmap, clabel = q, "viridis", "photons per module (NPE proxy)"
    else:
        c, cmap, clabel = t, "rainbow_r", "hit time [ns]  (early = red)"
    sc = ax.scatter(x, y, z, c=c, s=size, cmap=cmap, alpha=0.85,
                    depthshade=False)
    cb = ax.get_figure().colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
    cb.set_label(clabel)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    if title:
        ax.set_title(title)
    return ax
