"""
problems/eta_convergence.py
===========================
**Norm-based** measurement of the dynamic -> quasi-static limit: as the loading
time-scale ``eta`` is reduced, how close does the dynamic *state* get to the
quasi-static one?

Why norms and not energies
--------------------------
The first version of this study compared the two branches through their
**total energy**.  That turned out to depend on how the energy is bookkept
rather than on the solutions themselves:

* the dynamic total ``K + P_el + P_f + S`` *excludes* the viscous dissipation
  ``D``, while the quasi-static branch has no loss channel at all -- an
  asymmetry that by itself produces a floor in the "gap";
* with a foundation (``Lambda != 0``) the loading device keeps doing work on
  the broken specimen, so the quasi-static total *rises* after the crack and
  the comparison mixes stored energy with external work.

Measuring the **distance between the states** removes all of that.  Following
the meeting of 27/07, the state is the pair

.. math::  y_t = (u_t, \\alpha_t),

and finiteness of the energy

.. math::
   \\int_\\Omega g(\\alpha)|u'|^2 + |u|^2 + w(\\alpha) + |\\alpha'|^2 \\;<\\; \\infty

places both fields in the Sobolev space :math:`H^1(\\Omega)`, so the natural
distance is the :math:`H^1\\times H^1` norm

.. math::
   \\|v\\|_{L^2}=\\Big(\\int_\\Omega |v|^2\\Big)^{1/2},\\qquad
   \\|v\\|_{H^1}=\\Big(\\int_\\Omega |v|^2+\\int_\\Omega |v'|^2\\Big)^{1/2},\\qquad
   \\|y\\|^2 = \\|u\\|^2_{H^1}+\\|\\alpha\\|^2_{H^1}.

All the integrals are evaluated **exactly** for P1 fields (see
:func:`l2_norm_sq` / :func:`h1_semi_norm_sq`), so no quadrature error enters.

Norm equivalence
----------------
Two norms are *equivalent* when ``C||u||_1 <= ||u||_2 <= (1/C)||u||_1`` for
some ``C > 0``; on a fixed finite-element space every norm is equivalent to
every other, so the *conclusion* of a convergence study must not depend on
which one is used.  :func:`run_eta_study` therefore reports the whole family
-- ``L2`` and ``H1`` distances of ``u``, of ``alpha``, and of the state ``y``
-- and the figure overlays them precisely to show that they all fall together.

Method
------
Both branches are solved on the **same mesh** (only ``eta`` differs), so the
nodal vectors are directly comparable and no spatial interpolation is needed.
The two branches merely visit *different load levels*, so for each field the
nodal values are interpolated **in the load** onto a common grid, and then

    gap(eta) = || y_dyn(.) - y_qs(.) ||   measured over the loading path,

reported both as the worst case along the path (``sup``) and as the RMS along
it (a discrete Bochner ``L^2`` in the load).

Outputs
-------
``output/eta_norm_<physics>.png`` / ``.pdf`` -- the figure, and
``output/eta_norm_<physics>.npz`` -- the raw curves, so the figure can be
restyled without re-running the FEM.

Usage
-----
    from problems.eta_convergence import run_eta_study
    study = run_eta_study("mechanical", etas=[0.5, 0.2, 0.1, 0.05, 0.02])
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.parameters import get_defaults
from tools.norms import (l2_norm, h1_norm, state_norm,   # noqa: F401
                         l2_norm_sq, h1_semi_norm_sq, sort_dofs)
from tools.plotting import plot_eta_norm_study

# loading axis and human labels for each physics
_LOAD_KEY   = {"mechanical": "U",     "thermal": "theta"}
_LOAD_LABEL = {"mechanical": r"imposed displacement $\hat U$",
               "thermal":    r"thermal load $\theta$"}
_PHYS_ARG   = {"mechanical": "mechanical", "thermal": "thermal"}

# A light-but-still-cracking configuration so a full eta sweep runs in minutes.
# NOTE (27/07): the mechanical case is run with **Lambda = 0** -- no elastic
# foundation -- so that once the bar breaks there is no load path left and the
# loading device stops doing work on it.
_LIGHT_OVERRIDES = {
    "thermal": {
        "model_parameters":   {"l_hat": 0.04, "Lambda": 2.0,
                               "c1": 0.0, "c2": 1.0e-3, "c3": 1.0e-3},
        "mesh_parameters":    {"physics": "1D", "mesh_per_lhat": 3},
        "loading_parameters": {"theta_max": 3.0, "N_steps_qs": 60,
                               "N_steps_dyn": 90, "N_snapshots": 6},
        "solver_parameters":  {"model": "AT1"},
    },
    "mechanical": {
        "model_parameters":   {"l_hat": 0.04, "Lambda": 0.0,
                               "c1": 0.0, "c2": 1.0e-3, "c3": 1.0e-3},
        "mesh_parameters":    {"physics": "1D", "mesh_per_lhat": 3},
        "loading_parameters": {"U_max": 1.0, "N_steps_qs": 60,
                               "N_steps_dyn": 90, "N_snapshots": 6},
        "solver_parameters":  {"model": "AT1"},
    },
}


# =============================================================================
# Helpers
# =============================================================================
def _deep_update(cfg: dict, overrides: dict) -> dict:
    """In-place nested-dict update of the get_defaults() config."""
    for section, vals in overrides.items():
        if isinstance(vals, dict) and isinstance(cfg.get(section), dict):
            cfg[section].update(vals)
        else:
            cfg[section] = vals
    return cfg


class _Recorder:
    """Collects ``(load, u, alpha)`` at every converged step of both branches."""

    def __init__(self):
        self.qs = {"load": [], "u": [], "a": []}
        self.dyn = {"load": [], "u": [], "a": []}

    def __call__(self, branch, load, u_arr, a_arr):
        d = self.qs if branch == "qs" else self.dyn
        d["load"].append(float(load)); d["u"].append(u_arr); d["a"].append(a_arr)

    def arrays(self, branch):
        d = self.qs if branch == "qs" else self.dyn
        return (np.asarray(d["load"], float),
                np.asarray(d["u"], float),        # (n_steps, n_dofs)
                np.asarray(d["a"], float))


def _interp_fields(load, F, grid):
    """Interpolate a field history ``F`` (n_steps, n_dofs) in the LOAD, onto
    ``grid``.  Both branches share the mesh, so this is a nodewise 1-D interp."""
    o = np.argsort(load)
    load, F = load[o], F[o]
    return np.column_stack([np.interp(grid, load, F[:, j])
                            for j in range(F.shape[1])])


# =============================================================================
# Main entry point
# =============================================================================
def run_eta_study(physics: str,
                  etas=(0.5, 0.2, 0.1, 0.05, 0.02),
                  base_overrides: dict | None = None,
                  n_grid: int = 200,
                  defect: tuple | None = (0.5, 0.03, 0.05),
                  output_dir: str | Path | None = None,
                  plot: bool = True,
                  verbose: bool = True) -> dict:
    """Measure ``|| y_dyn(eta) - y_qs ||`` over the loading path, for each eta.

    ``defect = (x0, width, alpha0)`` seeds a small imperfection so that the
    crack site is unique.  **This is not cosmetic.**  A homogeneous bar with
    ``Lambda = 0`` has no preferred crack location, so quasi-static and dynamic
    localise at different places and the distance between the *fields* is
    dominated by that mismatch rather than by ``eta`` -- it does not converge.
    (An energy-based comparison is blind to this, because the energy does not
    change when the crack is translated: the norm measurement is what exposes
    the non-uniqueness.)  Pass ``defect=None`` to see the pathological case.

    Returns a dict with ``etas``, the load ``grid``, the per-eta distance
    curves ``dist[metric][k]`` along the path and the summary numbers
    ``sup[metric]`` / ``rms[metric]`` (both normalised by the corresponding
    norm of the quasi-static state), for the metrics

        ``u_L2``, ``u_H1``, ``a_L2``, ``a_H1``, ``y_H1``.
    """
    if physics not in _LOAD_KEY:
        raise ValueError(f"physics must be one of {list(_LOAD_KEY)}")
    from problems import dynamic, thermal
    module = {"mechanical": dynamic, "thermal": thermal}[physics]

    etas = sorted((float(e) for e in etas), reverse=True)   # fast -> slow
    out_dir = Path(output_dir or ROOT / "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    recs, x_u, x_a = {}, None, None
    for i, eta in enumerate(etas):
        cfg = get_defaults(_PHYS_ARG[physics])
        _deep_update(cfg, _LIGHT_OVERRIDES[physics])
        if base_overrides:
            _deep_update(cfg, base_overrides)
        cfg["model_parameters"]["eta"] = eta
        if verbose:
            print(f"[eta norm | {physics}] ({i+1}/{len(etas)}) eta={eta:g} ...")
        rec = _Recorder()
        kw = {} if defect is None else {"defect": defect}
        res = module.run_problem(**cfg, plot=False, verbose=False,
                                 field_recorder=rec, **kw)
        recs[eta] = rec
        if x_u is None:
            x_u, x_a = np.asarray(res["x_u"]), np.asarray(res["x_alpha"])

    # dofs come back in solver order -- sort them along the bar once and for all
    ou, oa = sort_dofs(x_u), sort_dofs(x_a)
    x_u, x_a = x_u[ou], x_a[oa]

    # ---- the quasi-static reference (eta-independent) -----------------------
    Lq, Uq, Aq = recs[etas[0]].arrays("qs")
    Uq, Aq = Uq[:, ou], Aq[:, oa]
    grid = np.linspace(float(np.min(Lq)), float(np.max(Lq)), n_grid)
    Uq_g, Aq_g = _interp_fields(Lq, Uq, grid), _interp_fields(Lq, Aq, grid)

    # norms of the reference along the path (used to normalise the distances)
    ref = {"u_L2": np.array([l2_norm(x_u, Uq_g[k]) for k in range(n_grid)]),
           "u_H1": np.array([h1_norm(x_u, Uq_g[k]) for k in range(n_grid)]),
           "a_L2": np.array([l2_norm(x_a, Aq_g[k]) for k in range(n_grid)]),
           "a_H1": np.array([h1_norm(x_a, Aq_g[k]) for k in range(n_grid)]),
           "y_H1": np.array([state_norm(x_u, Uq_g[k], x_a, Aq_g[k])
                             for k in range(n_grid)])}
    metrics = list(ref)
    scale = {m: max(float(np.max(ref[m])), 1e-30) for m in metrics}

    # ---- distance of each dynamic branch to it ------------------------------
    dist = {m: [] for m in metrics}
    dyn_state = []
    for eta in etas:
        Ld, Ud, Ad = recs[eta].arrays("dyn")
        Ud, Ad = Ud[:, ou], Ad[:, oa]
        Ud_g, Ad_g = _interp_fields(Ld, Ud, grid), _interp_fields(Ld, Ad, grid)
        du, da = Ud_g - Uq_g, Ad_g - Aq_g
        dist["u_L2"].append(np.array([l2_norm(x_u, du[k]) for k in range(n_grid)]))
        dist["u_H1"].append(np.array([h1_norm(x_u, du[k]) for k in range(n_grid)]))
        dist["a_L2"].append(np.array([l2_norm(x_a, da[k]) for k in range(n_grid)]))
        dist["a_H1"].append(np.array([h1_norm(x_a, da[k]) for k in range(n_grid)]))
        dist["y_H1"].append(np.array([state_norm(x_u, du[k], x_a, da[k])
                                      for k in range(n_grid)]))
        dyn_state.append((Ud_g, Ad_g))
    dist = {m: np.array(v) for m, v in dist.items()}          # (n_eta, n_grid)

    sup = {m: dist[m].max(axis=1) / scale[m] for m in metrics}
    rms = {m: np.sqrt((dist[m] ** 2).mean(axis=1)) / scale[m] for m in metrics}

    # ---- the crack event splits the path into two very different regimes ----
    # Before it, the two branches solve the *same smooth* problem and the
    # distance is controlled by inertia alone -> it must vanish with eta.
    # At the crack the state jumps, and a sub-step difference in *when* or
    # *where* it jumps costs an O(1) distance no matter how small eta is.
    # Measuring them together hides the convergence, so separate them.
    # The crack is the biggest JUMP of ||y_qs|| -- upward, not downward: the
    # damage band raises ||alpha||_{H1} and the displacement develops a steep
    # gradient across it, so the state norm grows when the bar breaks.
    # The window must be elastic for EVERY branch: if one of them cracks a
    # little earlier than the reference, the "pre-crack" part is contaminated
    # by an O(1) jump and the measurement is meaningless.  So take the
    # earliest crack over the quasi-static branch AND all dynamic ones.
    ref_y = ref["y_H1"]
    loads_crack = [float(grid[int(np.argmax(np.abs(np.diff(ref_y)))) + 1])]
    for k in range(len(etas)):
        yk = np.array([state_norm(x_u, dyn_state[k][0][i], x_a, dyn_state[k][1][i])
                       for i in range(n_grid)])
        loads_crack.append(float(grid[int(np.argmax(np.abs(np.diff(yk)))) + 1]))
    load_crack = min(loads_crack)
    span = grid[-1] - grid[0]
    pre = grid < load_crack - 0.05 * span
    if pre.sum() < 5:                       # degenerate detection -> safe half
        pre = grid < 0.5 * (grid[0] + grid[-1])
        load_crack = float(grid[pre.sum()])
    pre_sup = {m: dist[m][:, pre].max(axis=1) / scale[m] for m in metrics}
    pre_rms = {m: np.sqrt((dist[m][:, pre] ** 2).mean(axis=1)) / scale[m]
               for m in metrics}
    # observed order of convergence of the pre-crack distance in eta
    # A vanishing distance is not a failure: with AT1 the damage field is
    # identically zero in BOTH branches throughout the elastic phase, so
    # ||alpha_dyn - alpha_qs|| is *exactly* zero there.  Flag that instead of
    # fitting a slope through zeros.
    order, exact_zero = {}, {}
    e_arr = np.array(etas)
    for m in metrics:
        y = pre_sup[m]
        good = y > 0
        exact_zero[m] = bool((~good).any())
        order[m] = (float(np.polyfit(np.log(e_arr[good]), np.log(y[good]), 1)[0])
                    if good.sum() > 1 else float("nan"))

    study = {"physics": physics, "etas": np.array(etas), "grid": grid,
             "x_u": x_u, "x_alpha": x_a, "ref": ref, "scale": scale,
             "dist": dist, "sup": sup, "rms": rms, "metrics": metrics,
             "pre_sup": pre_sup, "pre_rms": pre_rms, "order": order,
             "exact_zero": exact_zero,
             "load_crack": load_crack, "pre_mask": pre, "defect": defect,
             "Lambda": _LIGHT_OVERRIDES[physics]["model_parameters"]["Lambda"]}

    if verbose:
        print(f"\n[eta norm | {physics}] crack event at load = {load_crack:.4f}")
        print("relative distance to the quasi-static state, BEFORE the crack "
              "(sup over the elastic part of the path):")
        print(f"{'eta':>9} " + "".join(f"{m:>11}" for m in metrics))
        for k, eta in enumerate(etas):
            print(f"{eta:9.4g} " + "".join(f"{pre_sup[m][k]:11.3e}" for m in metrics))
        print("observed order in eta: " +
              "  ".join(f"{m}=" + ("exactly 0 (elastic phase)" if np.isnan(order[m])
                                   else f"{order[m]:.2f}") for m in metrics))
        print("\nsame, over the WHOLE path (dominated by the crack jump):")
        print(f"{'eta':>9} " + "".join(f"{m:>11}" for m in metrics))
        for k, eta in enumerate(etas):
            print(f"{eta:9.4g} " + "".join(f"{sup[m][k]:11.4f}" for m in metrics))

    npz = out_dir / f"eta_norm_{physics}.npz"
    np.savez(npz, etas=study["etas"], grid=grid, load_crack=load_crack,
             ref_y=ref["y_H1"],
             **{f"pre_sup_{m}": pre_sup[m] for m in metrics},
             **{f"sup_{m}": sup[m] for m in metrics},
             **{f"rms_{m}": rms[m] for m in metrics},
             **{f"dist_{m}": dist[m] for m in metrics})
    if verbose:
        print(f"  saved {npz}")
    if plot:
        plot_eta_norm_study(study, out_dir, verbose=verbose)
    return study


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "mechanical"
    etas = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
    if which in ("mechanical", "both"):
        run_eta_study("mechanical", etas=etas)
    if which in ("thermal", "both"):
        run_eta_study("thermal", etas=etas)


# =============================================================================
# The first attempt, kept on purpose: the ENERGY-based measurement
# =============================================================================
# This is how the study was done before the norms.  It is retained because
# that is the order the argument runs in -- the natural thing to compare is
# the energy, and it is only by seeing *how* it fails that the move to Sobolev
# norms of the state is motivated.  Do not use it to draw conclusions; see
# :func:`run_eta_study` for the measurement that does.
def run_energy_study(physics: str,
                     etas=(0.5, 0.2, 0.1, 0.05, 0.02),
                     base_overrides: dict | None = None,
                     n_grid: int = 300,
                     defect: tuple | None = (0.5, 0.03, 0.05),
                     output_dir: str | Path | None = None,
                     plot: bool = True,
                     verbose: bool = True) -> dict:
    """Compare the two branches through their **total energy** (the first try).

    For each ``eta`` the dynamic total ``K + P_el + P_f + S`` is interpolated
    onto the quasi-static load grid with :func:`numpy.interp` and the residual
    measured as

        gap(eta) = || E_dyn - E_qs ||_RMS / max|E_qs| .

    Two things make this quantity a poor measure of "how close are the two
    solutions", and both are visible in the figure it produces:

    * the dynamic total **excludes** the cumulative viscous dissipation ``D``
      while the quasi-static branch has no dissipation channel at all, so the
      comparison is not symmetric and the crack event burns a finite amount of
      energy however slowly one loads -- a floor that does not vanish;
    * with a foundation the loading device keeps doing work on the *broken*
      specimen, so the quasi-static total rises after the crack and the
      comparison mixes stored energy with external work.

    Neither is a bug in the solver: the energy balance itself is exact (checked
    to 0.00 %).  They are defects of the *comparison*.
    """
    if physics not in _LOAD_KEY:
        raise ValueError(f"physics must be one of {list(_LOAD_KEY)}")
    from problems import dynamic, thermal
    from tools.plotting import plot_eta_energy_study
    module = {"mechanical": dynamic, "thermal": thermal}[physics]
    key = _LOAD_KEY[physics]

    etas = sorted((float(e) for e in etas), reverse=True)
    out_dir = Path(output_dir or ROOT / "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, qs_ref = {}, None
    for i, eta in enumerate(etas):
        cfg = get_defaults(_PHYS_ARG[physics])
        _deep_update(cfg, _LIGHT_OVERRIDES[physics])
        if base_overrides:
            _deep_update(cfg, base_overrides)
        cfg["model_parameters"]["eta"] = eta
        if verbose:
            print(f"[eta energy | {physics}] ({i+1}/{len(etas)}) eta={eta:g} ...")
        kw = {} if defect is None else {"defect": defect}
        res = module.run_problem(**cfg, plot=False, verbose=False, **kw)
        raw[eta] = {"qs": res["qs"], "dyn": res["dyn"]}
        if qs_ref is None:
            qs_ref = res["qs"]                       # QS is eta-independent

    Lq = np.asarray(qs_ref[key], float)
    Eq = np.asarray(qs_ref["total"], float)
    o = np.argsort(Lq); Lq, Eq = Lq[o], Eq[o]
    grid = np.linspace(Lq.min(), Lq.max(), n_grid)
    Eq_g = np.interp(grid, Lq, Eq)
    norm = max(float(np.max(np.abs(Eq_g))), 1e-30)

    E_dyn, gaps, D_end = [], [], []
    for eta in etas:
        d = raw[eta]["dyn"]
        Ld = np.asarray(d[key], float); Ed = np.asarray(d["total"], float)
        o = np.argsort(Ld); Ld, Ed = Ld[o], Ed[o]
        Ed_g = np.interp(grid, Ld, Ed)
        E_dyn.append(Ed_g)
        m = (grid >= Ld.min()) & (grid <= Ld.max())
        gaps.append(float(np.sqrt(np.mean((Ed_g[m] - Eq_g[m]) ** 2)) / norm))
        D_end.append(float(np.asarray(d["D"])[-1]))

    study = {"physics": physics, "etas": np.array(etas), "load_grid": grid,
             "E_qs": Eq_g, "E_dyn": E_dyn, "gaps": np.array(gaps),
             "D_end": np.array(D_end), "norm": norm, "load_key": key,
             "Lambda": _LIGHT_OVERRIDES[physics]["model_parameters"]["Lambda"]}

    if verbose:
        print(f"\n[eta energy | {physics}] normalised energy gap, and the "
              f"dissipation the comparison leaves out:")
        print(f"{'eta':>9} {'gap':>12} {'D(end)':>12}")
        for eta, g, D in zip(etas, gaps, D_end):
            print(f"{eta:9.4g} {g:12.4e} {D:12.4e}")

    npz = out_dir / f"eta_energy_{physics}.npz"
    np.savez(npz, etas=study["etas"], load_grid=grid, E_qs=Eq_g,
             E_dyn=np.array(E_dyn), gaps=study["gaps"], D_end=study["D_end"])
    if verbose:
        print(f"  saved {npz}")
    if plot:
        plot_eta_energy_study(study, out_dir, verbose=verbose)
    return study
