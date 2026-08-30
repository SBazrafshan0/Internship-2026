"""
problems/fragmentation.py
=========================
The **parametric studies**, as opposed to the single runs the other problem
files provide.

``dynamic.py`` and ``thermal.py`` each answer "what happens for *this* parameter
set".  The next question -- "what happens *as* a parameter is varied" -- is a
different kind of object: a sweep, with caching,
returning plain arrays that a figure function can consume without touching
FEniCSx.  Keeping them here means:

* every study is reproducible from one call with one dictionary of knobs;
* results are cached as ``.npz`` under ``output/cache/``, so re-drawing a figure
  after changing its style costs nothing;
* the figure layer never has to import dolfinx.

Studies
-------
:func:`run_bar_benchmark`   AT1 vs AT2 on the reference bar, with the damage
                            snapshots around nucleation.
:func:`run_verification`    does one crack cost ``G_c``; is the elastic limit
                            the analytic one; does the energy balance close.
:func:`run_spacing_study`   crack count and mean spacing against the foundation
                            stiffness -- the pattern-selection result.
:func:`run_pattern_2d`      the same in 2D, for the crack *pattern*.
:func:`run_crack_site_study` where the crack actually lands, with and without a
                            seeded imperfection -- the non-uniqueness result.

All of them accept ``force=True`` to ignore the cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.parameters import get_defaults
from tools.solvers import MODELS

CACHE = ROOT / "output" / "cache"


# =============================================================================
# Small shared helpers
# =============================================================================
def _deep_update(cfg: dict, overrides: dict) -> dict:
    for section, vals in (overrides or {}).items():
        if isinstance(vals, dict) and isinstance(cfg.get(section), dict):
            cfg[section].update(vals)
        else:
            cfg[section] = vals
    return cfg


def _cfg(physics, over):
    return _deep_update(get_defaults(physics), over)


def crack_positions(x, alpha, thr: float = 0.5):
    """Centres of the connected damaged regions of a 1-D damage profile.

    A "crack" is a maximal run of nodes with ``alpha > thr``; its position is
    the damage-weighted centroid of that run, which is stable under mesh
    refinement in a way that the arg-max is not.
    """
    x = np.asarray(x, float)
    a = np.asarray(alpha, float)
    o = np.argsort(x)
    x, a = x[o], a[o]
    hot = a > thr
    if not hot.any():
        return np.empty(0)
    out, i, n = [], 0, hot.size
    while i < n:
        if hot[i]:
            j = i
            while j + 1 < n and hot[j + 1]:
                j += 1
            w = a[i:j + 1]
            out.append(float(np.sum(w * x[i:j + 1]) / np.sum(w)))
            i = j + 1
        else:
            i += 1
    return np.asarray(out)


def mean_spacing(positions, L=1.0, include_ends: bool = True):
    """Mean distance between neighbouring cracks.

    With ``include_ends`` the two free ends count as fragment boundaries, which
    is the right convention for the thermal problem: the quantity of interest
    is the mean *fragment length*, and a film with one crack has two fragments,
    not zero spacings.
    """
    p = np.sort(np.asarray(positions, float))
    if p.size == 0:
        return np.nan
    q = np.r_[0.0, p, L] if include_ends else p
    d = np.diff(q)
    return float(np.mean(d)) if d.size else np.nan


class _Snapper:
    """``field_recorder`` that keeps the damage field of every step."""

    def __init__(self):
        self.load = {"qs": [], "dyn": []}
        self.alpha = {"qs": [], "dyn": []}

    def __call__(self, branch, load, u_arr, a_arr):
        self.load[branch].append(float(load))
        self.alpha[branch].append(np.asarray(a_arr).copy())

    def arrays(self, branch="qs"):
        return np.asarray(self.load[branch]), np.asarray(self.alpha[branch])


def _cached(name, force=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}.npz"
    if f.exists() and not force:
        with np.load(f, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}, f
    return None, f


def _store(f, data: dict):
    """Cache a study.  Ragged entries (one array per parameter value) are kept
    as object arrays, which is why every loader passes ``allow_pickle=True``."""
    packed = {}
    for k, v in data.items():
        try:
            packed[k] = np.asarray(v, dtype=float)
        except (ValueError, TypeError):
            packed[k] = np.asarray(v, dtype=object)
    np.savez_compressed(f, **packed)
    return f


# =============================================================================
# 1 -- the reference bar: AT1 against AT2
# =============================================================================
_BAR_BASE = {
    "model_parameters":   {"l_hat": 0.04, "Lambda": 0.0, "eta": 0.05,
                           "c1": 0.0, "c2": 1.0e-2, "c3": 1.0e-3},
    "mesh_parameters":    {"physics": "1D", "mesh_per_lhat": 5, "Lx": 1.0},
    "loading_parameters": {"U_max": 1.0, "N_steps_qs": 80, "N_steps_dyn": 120},
}


def run_bar_benchmark(models=("AT1", "AT2"), overrides=None, defect=(0.5, 0.03, 0.05),
                      force=False, verbose=True) -> dict:
    """The benchmark every later result is measured against.

    A homogeneous bar with **no foundation**: pulled at one end, one crack, and
    after it the reaction is exactly zero -- nothing holds the halves together.
    That is the clean case in which the analytic elastic limit
    ``e_crit = sqrt(w'(0)/(c_w E))`` and the Griffith energy ``G_c = l_hat``
    can both be read straight off the curves.

    Returns, per model, the two load-response curves, the energy histories,
    the final damage profile, and three damage snapshots bracketing nucleation.
    """
    from problems import dynamic

    out = {}
    for name in models:
        cache, f = _cached(f"bar_{name}", force)
        if cache is not None:
            out[name] = cache
            if verbose:
                print(f"[bar] {name}: cached")
            continue
        cfg = _cfg("mechanical", {**_BAR_BASE, "solver_parameters": {"model": name}})
        _deep_update(cfg, overrides or {})
        snap = _Snapper()
        if verbose:
            print(f"[bar] {name}: running ...")
        r = dynamic.run_problem(**cfg, plot=False, paraview=False,
                                verbose=verbose, defect=defect,
                                field_recorder=snap)
        loads, alphas = snap.arrays("qs")
        amax = alphas.max(axis=1)
        # the nucleation step: the first load at which damage escapes the seed
        seed = float(amax[0])
        i_nuc = int(np.argmax(amax > max(2.0 * seed, 0.2)))
        idx = [max(i_nuc - 1, 0), i_nuc, len(loads) - 1]
        d = {
            "qs_U": r["qs"]["U"], "qs_F": r["qs"]["F"],
            "qs_P_el": r["qs"]["P_el"], "qs_S": r["qs"]["S"],
            "qs_total": r["qs"]["total"],
            "dyn_U": r["dyn"]["U"], "dyn_F": r["dyn"]["F"],
            "x_alpha": np.sort(r["x_alpha"]),
            "alpha_qs_final": r["alpha_qs_final"][np.argsort(r["x_alpha"])],
            "qs_snapshots": alphas[idx][:, np.argsort(r["x_alpha"])],
            "qs_snapshot_U": loads[idx],
            "l_hat": cfg["model_parameters"]["l_hat"],
            "U_crit": loads[i_nuc],
        }
        _store(f, d)
        out[name] = d
    if verbose:
        ec = np.sqrt(1.0 / MODELS["AT1"]["c_w"])
        print(f"  AT1 elastic limit: measured U_crit = {float(out['AT1']['U_crit']):.4f}"
              f"  vs  sqrt(3/8) = {ec:.4f}")
    return out


# =============================================================================
# 2 -- verification: G_c, the elastic limit, and the energy balance
# =============================================================================
def _crack_energy_and_ecrit(cfg, defect, verbose, tol=1.0e-3):
    """One quasi-static bar run -> (S at the end, elastic limit, history).

    The elastic limit is read as the **first load at which damage becomes
    non-zero at all**, which for AT1 is a sharp event: below it ``alpha`` is
    identically zero, so any tolerance in a wide range gives the same answer.
    It is measured with ``defect=None`` -- a seeded imperfection lowers the
    local threshold and would bias the number downwards.
    """
    from problems import dynamic
    snap = _Snapper()
    r = dynamic.run_problem(**cfg, plot=False, paraview=False, verbose=False,
                            defect=defect, branches=("qs",),
                            field_recorder=snap)
    loads, alphas = snap.arrays("qs")
    amax = alphas.max(axis=1)
    seed = float(amax[0])
    hit = np.flatnonzero(amax > seed + tol)
    i_nuc = int(hit[0]) if hit.size else int(np.argmax(amax))
    return float(r["qs"]["S"][-1]), float(loads[i_nuc]), r["qs"]


def run_verification(mesh_per_lhats=(2, 3, 4, 6, 8),
                     l_hats=(0.08, 0.04, 0.02, 0.01),
                     defect=None, force=False,
                     verbose=True) -> dict:
    """The three checks that decide whether anything else can be believed.

    1. **Griffith.**  With the normalisation used here a fully developed AT1
       band costs exactly ``G_c = \\hat\\ell``.  Refining the mesh at fixed
       ``\\hat\\ell`` must drive the measured surface energy onto that value --
       from *above*, since a coarse mesh over-resolves the band.
    2. **The elastic limit does not move with the internal length.**  This is
       the property that distinguishes the normalisation used here (specific
       fracture energy fixed) from the one that fixes ``G_c``; it is also what
       Leon Baldelli & Maurini (JMPS 2021) state for this parametrisation.
    3. **Energy balance.**  Before the crack the stored energy must equal the
       work done by the loading device, ``\\int \\hat F\\, d\\hat U``.
    """
    cache, f = _cached("verification", force)
    if cache is not None:
        if verbose:
            print("[verification] cached")
        return {"mesh": {"mesh_per_lhat": cache["mpl"], "S_over_Gc": cache["S_over_Gc"]},
                "lhat": {"l_hat": cache["l_hat"], "e_crit_measured": cache["e_crit"],
                         "S_over_Gc": cache["S_over_Gc_lhat"]},
                "balance": {"U": cache["bU"], "stored": cache["bstored"],
                            "work": cache["bwork"],
                            "U_crack": float(cache["bU_crack"]),
                            "rel_err": float(cache["brel"])}}

    base = {**_BAR_BASE, "solver_parameters": {"model": "AT1"}}
    seed = (0.5, 0.03, 0.05)          # pins the crack for the energy measurement

    # ---- 1. mesh convergence at fixed l_hat --------------------------------
    # Measured with a seeded imperfection: a homogeneous bar has no preferred
    # crack site, and the point here is the energy of ONE band, not where it is.
    # The load stops just past the crack so that no further diffuse damage is
    # accumulated on top of it.
    S_mesh = []
    for mpl_ in mesh_per_lhats:
        cfg = _cfg("mechanical", base)
        cfg["mesh_parameters"]["mesh_per_lhat"] = int(mpl_)
        cfg["loading_parameters"].update(U_max=0.80, N_steps_qs=64)
        S, _, _ = _crack_energy_and_ecrit(cfg, seed, verbose)
        S_mesh.append(S / cfg["model_parameters"]["l_hat"])
        if verbose:
            print(f"[verification] h = l_hat/{mpl_}:  S/Gc = {S_mesh[-1]:.4f}")

    # ---- 2. the elastic limit against l_hat --------------------------------
    # Measured WITHOUT an imperfection: a seed lowers the local threshold, and
    # the quantity wanted here is the material one.
    ec, S_lh = [], []
    for lh in l_hats:
        cfg = _cfg("mechanical", base)
        cfg["model_parameters"]["l_hat"] = float(lh)
        cfg["mesh_parameters"]["mesh_per_lhat"] = 5
        # a finer load grid, or the measured threshold is quantised by dU
        cfg["loading_parameters"].update(U_max=0.80, N_steps_qs=256)
        S, U_c, _ = _crack_energy_and_ecrit(cfg, defect, verbose)
        ec.append(U_c); S_lh.append(S / lh)
        if verbose:
            print(f"[verification] l_hat = {lh:g}:  e_crit = {U_c:.4f}, "
                  f"S/Gc = {S_lh[-1]:.4f}")

    # ---- 3. energy balance on the reference run ----------------------------
    cfg = _cfg("mechanical", base)
    _, _, qs = _crack_energy_and_ecrit(cfg, seed, verbose)
    U = np.asarray(qs["U"], float)
    F = np.asarray(qs["F"], float)
    stored = np.asarray(qs["total"], float)
    work = np.r_[0.0, np.cumsum(0.5 * (F[1:] + F[:-1]) * np.diff(U))]
    i = int(np.argmax(np.abs(np.diff(stored)))) if stored.size > 2 else stored.size - 1
    pre = slice(0, max(i, 2))
    rel = float(np.max(np.abs((stored - stored[0])[pre] - work[pre]))
                / max(np.max(np.abs(stored)), 1e-30))
    if verbose:
        print(f"[verification] energy balance closes to {rel:.2%}")

    _store(f, {"mpl": mesh_per_lhats, "S_over_Gc": S_mesh,
               "l_hat": l_hats, "e_crit": ec, "S_over_Gc_lhat": S_lh,
               "bU": U, "bstored": stored - stored[0], "bwork": work,
               "bU_crack": U[i], "brel": rel})
    return {"mesh": {"mesh_per_lhat": np.asarray(mesh_per_lhats, float),
                     "S_over_Gc": np.asarray(S_mesh)},
            "lhat": {"l_hat": np.asarray(l_hats, float),
                     "e_crit_measured": np.asarray(ec),
                     "S_over_Gc": np.asarray(S_lh)},
            "balance": {"U": U, "stored": stored - stored[0], "work": work,
                        "U_crack": float(U[i]), "rel_err": rel}}


# =============================================================================
# 3 -- pattern selection: crack spacing against the foundation stiffness
# =============================================================================
_THERM_BASE = {
    "model_parameters":   {"l_hat": 0.005, "eta": 0.05,
                           "c1": 0.0, "c2": 1.0e-2, "c3": 1.0e-3},
    "mesh_parameters":    {"physics": "1D", "mesh_per_lhat": 4, "Lx": 1.0},
    "loading_parameters": {"theta_max": 3.0, "N_steps_qs": 80,
                           "N_steps_dyn": 120, "N_snapshots": 8},
    "solver_parameters":  {"model": "AT1"},
}


#: A run whose damaged fraction exceeds this has lost its pattern: the bands
#: have merged and the crack count is an artefact, not a measurement.
MERGE_FRACTION = 0.5


def saturation_spacing(Lambda, theta, e_crit=None, L=1.0):
    """Closed-form fragment length at which the film stops cracking.

    A fragment of length ``d``, free at both ends, on a Winkler foundation with
    a uniform eigenstrain ``theta`` satisfies (with ``E=1``)

        u'' = Lambda^2 u,      u'(+-d/2) = theta,

    hence ``u = theta sinh(Lambda x)/(Lambda cosh(Lambda d/2))`` and an elastic
    strain ``e = u' - theta`` whose magnitude is largest at the *centre*:

        |e|_max = theta [ 1 - 1/cosh(Lambda d / 2) ].

    The fragment cracks again while that exceeds the nucleation threshold, so
    fragmentation stops at

        d* = (2/Lambda) arccosh[ 1 / (1 - e_crit/theta) ].

    Two things follow, and both are testable: the spacing is **inversely
    proportional to Lambda** (i.e. proportional to the transfer length), and the
    prefactor is set by how far past the threshold the film is driven -- it is
    not a free constant.  Returns ``inf`` when ``theta <= e_crit`` (the film
    never cracks) and is capped at ``L``.
    """
    import math
    from tools.solvers import MODELS
    if e_crit is None:
        e_crit = math.sqrt(1.0 / MODELS["AT1"]["c_w"])
    Lambda = np.asarray(Lambda, float)
    if theta <= e_crit:
        return np.full_like(Lambda, np.inf)
    d = (2.0 / Lambda) * np.arccosh(1.0 / (1.0 - e_crit / theta))
    return np.minimum(d, L)


def run_spacing_study(Lambdas=(4.0, 6.0, 8.0, 12.0, 16.0, 24.0),
                      l_hats=(0.005, 0.01),
                      overrides=None, profile_at=(4.0, 8.0, 24.0),
                      profile_l_hat=0.005,
                      force=False, verbose=True) -> dict:
    """What sets the crack spacing?

    Two candidate laws, and the sweep decides between them.

    **(i) Sequential halving.**  If cracks appeared one at a time, each fragment
    would split at its centre until it was short enough to stay below the
    threshold, giving the saturation length :func:`saturation_spacing`,
    ``d* = (2/Lambda) arccosh[1/(1-e_c/theta)]``, i.e. ``d ~ 1/Lambda`` and
    **independent of the internal length**.

    **(ii) A single instability.**  But at the elastic limit the stress is
    uniform to within a boundary layer, so the criterion is met *everywhere at
    once*: the pattern is selected by which perturbation grows, not by a
    sequence of events.  That mechanism mixes the two lengths of the problem and
    predicts ``d ~ sqrt(l_hat * ell_e) = sqrt(l_hat/Lambda)`` -- a **-1/2**
    exponent in ``Lambda``, and a spacing that *does* move with ``l_hat``.

    Sweeping ``Lambda`` alone cannot separate them (both give a decreasing
    curve), so the study sweeps ``l_hat`` as well: only (ii) predicts that the
    two sets of points collapse when divided by ``sqrt(l_hat/Lambda)``.

    This is the mechanism Leon Baldelli & Maurini (JMPS 2021) describe for the
    thin-film strip -- "the damage criterion is met almost uniformly ... this,
    coupled with the existence of the additional length scale, is responsible
    for the appearance of structured crack patterns beyond the elastic limit".

    Only the quasi-static branch is integrated: the pattern is a property of the
    equilibrium path, not of the loading rate.
    """
    cache, f = _cached("spacing", force)
    if cache is not None:
        if verbose:
            print("[spacing] cached")
        return {k: (list(cache[k]) if cache[k].dtype == object else cache[k])
                for k in cache}

    from problems import thermal

    Lambdas = np.asarray(Lambdas, float)
    l_hats = np.asarray(l_hats, float)
    spac = np.full((l_hats.size, Lambdas.size), np.nan)
    ncr = np.zeros_like(spac)
    # Fraction of the film that ends up damaged.  When the bands (width ~4
    # l_hat) get as close as the spacing they merge into a continuum: the
    # crack counter then sees ONE region and the measurement is meaningless.
    # Recording the fraction lets a figure exclude those points for a stated
    # reason instead of silently fitting through them.
    frac = np.zeros_like(spac)
    theta_hist, n_hist, profiles, prof_lam = [], [], [], []
    x_alpha = None
    theta_max = None

    for h, lh in enumerate(l_hats):
        for j, L in enumerate(Lambdas):
            cfg = _cfg("thermal", {
                **_THERM_BASE,
                "model_parameters": {**_THERM_BASE["model_parameters"],
                                     "Lambda": float(L), "l_hat": float(lh)}})
            _deep_update(cfg, overrides or {})
            theta_max = cfg["loading_parameters"]["theta_max"]
            if verbose:
                print(f"[spacing] l_hat = {lh:g}, Lambda = {L:g} ...")
            r = thermal.run_problem(**cfg, plot=False, paraview=False,
                                    verbose=False, branches=("qs",))
            o = np.argsort(r["x_alpha"])
            xs, prof = r["x_alpha"][o], r["alpha_qs_final"][o]
            pos = crack_positions(xs, prof)
            Lx = cfg["mesh_parameters"]["Lx"]
            spac[h, j] = mean_spacing(pos, L=Lx)
            ncr[h, j] = len(pos)
            frac[h, j] = float(np.mean(prof > 0.5))
            if abs(lh - profile_l_hat) < 1e-12:
                x_alpha = xs
                theta_hist.append(np.asarray(r["qs"]["theta"], float))
                n_hist.append(np.asarray(r["qs"]["n_cracks"], float))
                if any(abs(L - v) < 1e-9 for v in profile_at):
                    profiles.append(prof); prof_lam.append(float(L))
            if verbose:
                d_star = float(saturation_spacing(L, theta_max))
                print(f"           -> {len(pos)} cracks, d = {spac[h, j]:.4f}"
                      f"  | halving {d_star:.4f}"
                      f"  | sqrt(l_hat/Lambda) {np.sqrt(lh / L):.4f}"
                      f"  | damaged {frac[h, j]:.0%}"
                      + ("   <-- bands merged, excluded"
                         if frac[h, j] > MERGE_FRACTION else ""))

    d = {"Lambda": Lambdas, "l_hat": l_hats, "spacing": spac, "n_final": ncr,
         "damaged_fraction": frac, "merge_fraction": np.asarray([MERGE_FRACTION]),
         "theta_max": np.asarray([theta_max]),
         "profile_Lambda_all": Lambdas,
         "theta": np.asarray(theta_hist, dtype=object),
         "n_cracks": np.asarray(n_hist, dtype=object),
         "x_alpha": x_alpha,
         "profiles": np.asarray(profiles, dtype=object),
         "profile_Lambda": np.asarray(prof_lam, float)}
    np.savez_compressed(f, **d)
    for k in ("theta", "n_cracks", "profiles"):
        d[k] = list(d[k])
    return d


def run_pattern_2d(Lambdas=(4.0, 12.0, 32.0), overrides=None, mesh_per_lhat=3,
                   l_hat=0.03, Ly=0.25, theta_max=6.0, verbose=True):
    """The 2-D crack pattern at three foundation stiffnesses.

    Not cached: the meshes differ between runs, so the natural cache unit is
    the ``.npz`` a caller writes itself.  Returns a list of dicts with the
    matplotlib triangulation and the final damage field, ready to plot.
    """
    from problems import thermal

    runs = []
    for L in Lambdas:
        cfg = _cfg("thermal", {
            **_THERM_BASE,
            "model_parameters": {**_THERM_BASE["model_parameters"],
                                 "Lambda": float(L), "l_hat": float(l_hat)},
            "mesh_parameters": {"physics": "2D", "mesh_per_lhat": mesh_per_lhat,
                                "Lx": 1.0, "Ly": Ly},
            "loading_parameters": {**_THERM_BASE["loading_parameters"],
                                   "theta_max": theta_max},
        })
        _deep_update(cfg, overrides or {})
        if verbose:
            print(f"[pattern 2D] Lambda = {L:g} ...")
        r = thermal.run_problem(**cfg, plot=False, paraview=False,
                                verbose=verbose, branches=("qs",))
        runs.append({"Lambda": float(L), "triang": r["triang"],
                     "alpha": r["alpha_qs_final"]})
    return runs


# =============================================================================
# 4 -- non-uniqueness: where does the crack actually land?
# =============================================================================
def run_crack_site_study(etas=(0.3, 0.1, 0.03, 0.01),
                         defect=(0.5, 0.03, 0.05), overrides=None,
                         force=False, verbose=True) -> dict:
    """The result the Sobolev norm found and no energy could.

    A homogeneous bar with ``\\Lambda=0`` is translation invariant: every crack
    position costs exactly the same energy, so the evolution is **non-unique**.
    An energy-based comparison of two branches is blind to this -- translating
    a crack changes no energy at all -- and will happily call two completely
    different solutions identical.  Measuring where the crack lands makes the
    degeneracy visible, and shows that seeding a small imperfection removes it
    only once the loading is slow enough to beat inertia.
    """
    cache, f = _cached("crack_site", force)
    if cache is not None:
        if verbose:
            print("[crack site] cached")
        return {"free": {"etas": cache["etas"], "x_qs": float(cache["free_xqs"]),
                         "x_dyn": list(cache["free_xdyn"])},
                "seeded": {"etas": cache["etas"], "x_qs": float(cache["seed_xqs"]),
                           "x_dyn": list(cache["seed_xdyn"])}}

    from problems import dynamic

    base = {**_BAR_BASE, "solver_parameters": {"model": "AT1"},
            "model_parameters": {**_BAR_BASE["model_parameters"], "l_hat": 0.04}}
    result = {}
    for key, dfc in (("free", None), ("seeded", defect)):
        x_dyn, x_qs = [], None
        for eta in etas:
            cfg = _cfg("mechanical", base)
            cfg["model_parameters"]["eta"] = float(eta)
            _deep_update(cfg, overrides or {})
            if verbose:
                print(f"[crack site] {key}, eta = {eta:g} ...")
            r = dynamic.run_problem(**cfg, plot=False, paraview=False,
                                    verbose=False, defect=dfc)
            o = np.argsort(r["x_alpha"])
            x = r["x_alpha"][o]
            if x_qs is None:
                p = crack_positions(x, r["alpha_qs_final"][o])
                x_qs = float(p[0]) if p.size else np.nan
            p = crack_positions(x, r["alpha_dyn_final"][o])
            x_dyn.append(p if p.size else np.array([np.nan]))
            if verbose:
                print(f"             QS at {x_qs:.3f}; dynamic at "
                      f"{np.round(x_dyn[-1], 3)}")
        result[key] = {"etas": np.asarray(etas, float), "x_qs": x_qs,
                       "x_dyn": x_dyn}

    np.savez_compressed(f, etas=np.asarray(etas, float),
                        free_xqs=result["free"]["x_qs"],
                        free_xdyn=np.asarray(result["free"]["x_dyn"], dtype=object),
                        seed_xqs=result["seeded"]["x_qs"],
                        seed_xdyn=np.asarray(result["seeded"]["x_dyn"], dtype=object))
    return result


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "bar"):
        run_bar_benchmark()
    if which in ("all", "verification"):
        run_verification()
    if which in ("all", "spacing"):
        run_spacing_study()
    if which in ("all", "site"):
        run_crack_site_study()
