"""
Build the cached data files the notebook falls back to.

Run this ONCE on a machine / container with internet + the working toolchain
(SIREN, Prometheus, ...). It writes a handful of small files into ./data
which you then upload to your CACHE_BASE_URL (e.g. a GitHub release). The Colab
notebook downloads these so every stage works even if a live install fails.

    python make_cache.py            # build everything that's available
    python make_cache.py --flux     # just the oscillated-flux array
    python make_cache.py --events   # just the Prometheus nu_tau events

Each builder is wrapped in try/except so a missing tool skips that file instead
of crashing the whole run.
"""

import argparse
import os

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA, exist_ok=True)


SIREN_FLUX_RECORD = "20129082"   # Zenodo record shipping SIREN's flux tables


def _zenodo_download(record_id, filename, dest):
    """Self-contained Zenodo file fetch (no SIREN/compiled deps). Uses the
    stable public URL pattern zenodo.org/records/<id>/files/<name>?download=1."""
    import urllib.request  # noqa: PLC0415

    os.makedirs(dest, exist_ok=True)
    out = os.path.join(dest, filename)
    if os.path.exists(out):
        return out
    url = f"https://zenodo.org/records/{record_id}/files/{filename}?download=1"
    print(f"[flux] downloading {filename} ({record_id}) — this is ~1.2 GB ...")
    urllib.request.urlretrieve(url, out)
    return out


def build_flux_subset():
    """Pull SIREN's flux archive from Zenodo and extract a SMALL atmospheric
    table (surface + at-detector) into data/atmo_flux_siren.npz.

    The at-detector tables already fold in nuSQuIDS oscillation + Earth
    absorption, so this replaces daemonflux/nuflux/nuSQuIDS for the lecture.
    The notebook only ever downloads the small .npz, never the 1.2 GB archive.
    """
    import zipfile  # noqa: PLC0415

    work = os.path.join(DATA, "_siren_fluxes")
    archive = _zenodo_download(SIREN_FLUX_RECORD, "fluxes.zip", work)
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        atmo = [n for n in names if "atmos" in n.lower()]
        print(f"[flux] archive has {len(names)} files; atmospheric matches:")
        for n in atmo:
            print("   ", n)
        zf.extractall(work)

    # ---- TODO: adapt to the real table format/paths printed above ----
    # Read the surface and at-detector atmospheric tables and put them on a
    # common (energy x coszen x flavor x nu/nubar) grid. Filenames/parsing
    # depend on the SIREN flux schema — wire to the matches listed above.
    raise NotImplementedError(
        "Inspect the atmospheric matches printed above, then read the surface "
        "and detector tables and np.savez them as data/atmo_flux_siren.npz with "
        "keys: energy_gev, coszen, flux_surface, flux_detector, flavors."
    )


def build_prometheus_events(n_events=200, geo_file=None):
    """Generate a small set of nu_tau CC events with Prometheus (CPU photon
    propagator) and copy the photon-hit output to data/ for the §5 display.

    Mirrors the illustrative config in the notebook. Adjust keys to your build.
    """
    import shutil  # noqa: PLC0415
    from prometheus import Prometheus  # noqa: PLC0415
    from prometheus.config import config  # noqa: PLC0415

    print(f"[events] generating {n_events} nu_tau events with Prometheus ...")
    config["run"]["nevents"] = n_events
    config["injection"]["name"] = "SIREN"                 # or "LeptonInjector"
    config["photon propagator"]["name"] = "olympus"       # CPU; avoids GPU PPC
    if geo_file:
        config["detector"]["geo file"] = geo_file
    # ---- set primary = NuTau + the SIREN injection paths to match build_siren_events ----
    p = Prometheus(config)
    p.sim()
    out = config["run"].get("outfile") or config.get("out", {}).get("output file")
    dest = os.path.join(DATA, "prometheus_nutau_example.parquet")
    if out and os.path.exists(out):
        shutil.copy(out, dest)
        print(f"[events] wrote {dest}")
    else:
        print("[events] set the Prometheus output path and copy it to", dest)


def build_siren_events(n_events=int(2e4), e_min=1e4, e_max=1e7, experiment="IceCube"):
    """SIREN nu_tau CC-DIS injection -> data/siren_nutau_injection.parquet.

    Follows resources/examples/example1/DIS_IceCube.py, swapping NuMu->NuTau and
    widening the energy range. Runs end-to-end through SaveEvents; the only piece
    left is converting that output into the small flat table the notebook reads.
    """
    import siren  # noqa: PLC0415
    from siren._util import GenerateEvents, SaveEvents  # noqa: PLC0415

    print(f"[siren] injecting {n_events} nu_tau events ...")
    detector_model = siren.utilities.load_detector(experiment)
    primary_type = siren.dataclasses.Particle.ParticleType.NuTau
    Nucleon = siren.dataclasses.Particle.ParticleType.Nucleon

    primary_processes, _ = siren.utilities.load_processes(
        "CSMSDISSplines", primary_types=[primary_type], target_types=[Nucleon],
        isoscalar=True, process_types=["CC"],
    )
    primary_cross_sections = primary_processes[primary_type]

    injector = siren.injection.Injector()
    injector.number_of_events = n_events
    injector.detector_model = detector_model
    injector.primary_type = primary_type
    injector.primary_interactions = primary_cross_sections
    injector.primary_injection_distributions = [
        siren.distributions.PrimaryMass(0),
        siren.distributions.PowerLaw(2, e_min, e_max),
        siren.distributions.IsotropicDirection(),
        siren.distributions.ColumnDepthPositionDistribution(
            600, 600.0, siren.distributions.LeptonDepthFunction()),
    ]
    events, gen_times = GenerateEvents(injector)

    weighter = siren.injection.Weighter()
    weighter.injectors = [injector]
    weighter.detector_model = detector_model
    weighter.primary_type = primary_type
    weighter.primary_interactions = primary_cross_sections
    weighter.primary_physical_distributions = [
        siren.distributions.PowerLaw(2, e_min, e_max),
        siren.distributions.IsotropicDirection(),
    ]

    outdir = os.path.join(DATA, "output")
    os.makedirs(outdir, exist_ok=True)
    SaveEvents(events, weighter, gen_times,
               output_filename=os.path.join(outdir, "siren_nutau"))

    # TODO: read the SaveEvents output above and write a small flat table with
    #       columns energy_gev, bjorken_y, cos_zen, gen_weight ->
    #       data/siren_nutau_injection.parquet  (final-state schema is yours to map).
    print("[siren] SaveEvents written to", outdir,
          "-> convert to data/siren_nutau_injection.parquet")


BUILDERS = {
    "flux": build_flux_subset,
    "events": build_prometheus_events,
    "siren": build_siren_events,
}


def main():
    ap = argparse.ArgumentParser()
    for name in BUILDERS:
        ap.add_argument(f"--{name}", action="store_true")
    args = ap.parse_args()
    selected = [n for n in BUILDERS if getattr(args, n)] or list(BUILDERS)
    for name in selected:
        try:
            BUILDERS[name]()
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {name}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
