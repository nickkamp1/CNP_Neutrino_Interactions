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

# Flavour part of the label, by |PDG| of the initial neutrino.
_FLAVOR = {12: "nu_e", 14: "nu_mu", 16: "nu_tau"}


def class_label(pdg, interaction):
    """The event class the game checks against.

    Charged-current (``interaction == 1``) keeps the flavour ("nu_e CC", "nu_mu CC",
    "nu_tau CC"); neutral-current (``interaction == 2``) is a flavour-blind hadronic
    cascade, labelled simply "NC".
    """
    if interaction == 2:
        return "NC"
    return f"{_FLAVOR.get(abs(int(pdg)), 'nu')} CC"


# Reading a Prometheus parquet pulls every photon of every event into memory, so it
# is slow (seconds) for the big files. The "guess the event" game samples many events,
# so we memo-cache the parsed awkward array (and the per-event photon count) per path.
_ARR_CACHE = {}
_NPH_CACHE = {}


def _load_arr(path):
    """Parse a Prometheus parquet to an awkward array, memo-cached by path."""
    import awkward as ak  # noqa: PLC0415

    arr = _ARR_CACHE.get(path)
    if arr is None:
        arr = ak.from_parquet(path)
        _ARR_CACHE[path] = arr
    return arr


def n_photons_array(path):
    """Return an int array of #detected photons per event (cached)."""
    import awkward as ak  # noqa: PLC0415

    nph = _NPH_CACHE.get(path)
    if nph is None:
        nph = ak.to_numpy(ak.num(_load_arr(path).photons.t))
        _NPH_CACHE[path] = nph
    return nph


# Folder name (the injected final state) -> the four IceCube_HE signatures. Used to
# enumerate the dataset for the "guess the event" game.
SIGNATURE_FOLDERS = ("EMinus", "MuMinus", "TauMinus", "NuMu")


def signature_parquet(detector, signature, data_dir=None):
    """Path to the committed ``Generation_*_photons.parquet`` for one signature."""
    base = data_dir or os.path.join(DATA_DIR, "Prometheus_simulation")
    return os.path.join(base, detector, signature,
                        "Generation_00000-000_photons.parquet")


def pick_random_event(detector="IceCube_HE", rng=None, min_photons=1):
    """Pick a random (path, event_index) with >= ``min_photons`` detected photons.

    Draws a signature folder uniformly, then a random event in it that actually lit
    up the detector -- the pool the "guess the event" game samples from. Returns
    ``(path, index)`` suitable for :func:`load_prometheus_event`.
    """
    rng = rng or np.random.default_rng()
    for _ in range(100):  # almost always succeeds first try
        sig = SIGNATURE_FOLDERS[rng.integers(len(SIGNATURE_FOLDERS))]
        path = signature_parquet(detector, sig)
        if not os.path.exists(path):
            continue
        nph = n_photons_array(path)
        ok = np.flatnonzero(nph >= min_photons)
        if ok.size:
            return path, int(rng.choice(ok))
    raise RuntimeError(f"No event with >={min_photons} photons under {detector}.")


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

    arr = _load_arr(path)

    nph = n_photons_array(path)
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
    interaction = int(rec.mc_truth.interaction)   # 1 = CC, 2 = NC (LeptonInjector code)
    info = {
        "event_index": idx,
        "n_photons": int(nph[idx]),
        "n_modules_hit": int(uniq.size),
        "initial_state_type": pdg,
        "initial_state_label": _PDG_LABEL.get(pdg, str(pdg)),
        "initial_state_energy_gev": float(rec.mc_truth.initial_state_energy),
        "interaction": interaction,
        # human label the "guess the event" game checks against:
        #   nu_tau CC / nu_mu CC / nu_e CC  vs.  NC (any flavour, hadronic cascade)
        "class_label": class_label(pdg, interaction),
        "bjorken_y": float(rec.mc_truth.bjorken_y),
        # true interaction vertex + neutrino direction (for the reveal arrow):
        "vertex_x": float(rec.mc_truth.initial_state_x),
        "vertex_y": float(rec.mc_truth.initial_state_y),
        "vertex_z": float(rec.mc_truth.initial_state_z),
        "zenith": float(rec.mc_truth.initial_state_zenith),
        "azimuth": float(rec.mc_truth.initial_state_azimuth),
        "first_cascade_x": float(rec.mc_truth.final_state_x[-1]),
        "first_cascade_y": float(rec.mc_truth.final_state_y[-1]),
        "first_cascade_z": float(rec.mc_truth.final_state_z[-1]),
        "second_cascade_x": float(rec.mc_truth.final_state_x[1]),
        "second_cascade_y": float(rec.mc_truth.final_state_y[1]),
        "second_cascade_z": float(rec.mc_truth.final_state_z[1]),
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
    size = 10*q**(1./3.)
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
    ax.set_xlim(-500,500)
    ax.set_ylim(-500,500)
    ax.set_zlim(-2500,-1500)
    if title:
        ax.set_title(title)
    return ax


def direction_unit_vector(zenith, azimuth):
    """Unit vector of the incoming neutrino from (zenith, azimuth), radians.

    Prometheus stores the direction the particle *comes from*; the momentum points
    the opposite way, which is what we draw as the travel arrow."""
    sz = np.sin(zenith)
    return -np.array([sz * np.cos(azimuth), sz * np.sin(azimuth), np.cos(zenith)])


def add_truth_overlay(ax, info, length=120.0, show_cascades=True):
    """Overlay the MC-truth interaction vertex + travel-direction arrow on a display.

    Drops a star at the interaction vertex and draws an arrow along the lepton's
    travel direction (opposite the stored incoming direction). For a tau double-bang
    the two cascade points are starred too. Used by the formalised tau display and by
    the "guess the event" reveal button."""
    vx, vy, vz = info["vertex_x"], info["vertex_y"], info["vertex_z"]
    d = direction_unit_vector(info["zenith"], info["azimuth"])
    ax.scatter([vx], [vy], [vz], marker="*", s=320, c="red",
               edgecolor="k", depthshade=False, label="interaction vertex", zorder=6)
    ax.quiver(vx, vy, vz, d[0], d[1], d[2], length=length, color="red",
              linewidth=2, arrow_length_ratio=0.3)
    if show_cascades:
        ax.scatter([info["first_cascade_x"], info["second_cascade_x"]],
                   [info["first_cascade_y"], info["second_cascade_y"]],
                   [info["first_cascade_z"], info["second_cascade_z"]],
                   marker="*", s=160, c="magenta", edgecolor="k",
                   depthshade=False, zorder=6)
    return ax


def dom_waveforms(path, event, n_brightest=12):
    """Per-DOM photon arrival-time arrays for the brightest modules in one event.

    Groups the event's photons by module ``(string_id, sensor_id)`` and returns the
    ``n_brightest`` modules (most photons first). This is the waveform each optical
    module records -- a tau double-bang shows two arrival-time clusters (one per
    cascade) on modules between the bangs.

    Returns a list of dicts ``{string_id, sensor_id, n_photons, times}`` sorted by
    photon count, descending.
    """
    import awkward as ak  # noqa: PLC0415

    rec = _load_arr(path)[int(event)]
    sid = ak.to_numpy(rec.photons.string_id).astype(np.int64)
    did = ak.to_numpy(rec.photons.sensor_id).astype(np.int64)
    tt = ak.to_numpy(rec.photons.t).astype(float)

    key = sid * 100000 + did
    _, inv = np.unique(key, return_inverse=True)
    counts = np.bincount(inv)
    order = np.argsort(counts)[::-1][:n_brightest]

    out = []
    for m in order:
        sel = inv == m
        out.append({
            "string_id": int(sid[sel][0]),
            "sensor_id": int(did[sel][0]),
            "n_photons": int(counts[m]),
            "times": tt[sel],
        })
    return out
