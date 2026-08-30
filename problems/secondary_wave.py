"""
problems/secondary_wave.py
==========================
Direct space-time simulation of the post-crack wave problem **with real crack
nucleation and domain splitting** -- the piece the modal solution of
:mod:`problems.secondary` structurally cannot provide.

Why a new solver
----------------
The semi-analytic model of ``secondary.py`` expands the transient on the
eigenfunctions of a *fixed* domain, so its boundary conditions can never
change: it can say *whether* the strain would reach ``e_crit``, but it keeps
propagating the wave as if no crack had appeared.  In reality the instant a
secondary crack nucleates at ``x_c``:

* the fragment **splits** into ``[0, x_c]`` and ``[x_c, L]``;
* two **new traction-free faces** are created at ``x_c``;
* each face relaxes its own stress and launches a **fresh release wave** into
  its own sub-fragment;
* those waves can nucleate a **third generation** of cracks, and so on.

This module integrates the damped Klein-Gordon equation

.. math::  m u_{tt} + \\eta u_t - E_h u_{xx} + k u = 0

directly in time on a P1 finite-element grid with **lumped mass**.  A crack is
simply a *broken element*: its internal force is set to zero forever, which
decouples the two sides and makes both new faces traction-free automatically.
**Domain splitting therefore needs no special code at all** -- it falls out of
the discretisation.

Set-up (same half-specimen as the concept note)
-----------------------------------------------
* ``x = 0``    : the first crack -- traction-free from ``t = 0``;
* ``x = L``    : the symmetry plane of the full specimen -- ``u = 0`` while it
  is intact.  If the criterion is met there, the symmetry plane itself cracks
  (a genuine crack at the centre of the full specimen) and the constraint is
  *released* rather than an element being broken, so the fragment becomes
  free-free.  This is the usual first secondary crack, because ``x = L`` is a
  strain antinode;
* initial state: uniform pre-stress ``e = u_x - theta = -theta``, ``u = v = 0``.

Nucleation rule
---------------
An intact element cracks when ``|e| >= e_crit``, provided it is at least one
damage-band width ``ell_d`` away from every existing crack (two cracks cannot
overlap).  With AT1 and ``Gc = E_h = 1`` the band width is tied to the
threshold, ``ell_d = 1/e_crit**2`` -- **no new free parameter** (see §10 of
``secondary_theory.ipynb``).

Control parameters (identical to the note)
------------------------------------------
``Lambda_bar = ell_e/L``,  ``Gamma = gamma*tau_rt`` with
``tau_rt = 2*sqrt(2)*L/c``,  and the threshold ``e_crit``.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.solvers import damage_band_width
# Figures live in tools.plotting, like every other figure in the repository;
# they are re-exported here so a caller only needs this one module.
from tools.plotting import (plot_wave_spacetime, plot_wave_energy,   # noqa: F401
                            animate_wave_run)


def simulate(Lambda_bar: float,
             Gamma: float,
             e_crit: float = 1.3,
             theta: float = 1.0,
             ell_e: float = 1.0,
             c: float = 1.0,
             E_h: float = 1.0,
             Gc: float = 1.0,
             n_roundtrips: float = 4.0,
             cells_per_ell_e: int = 60,
             n_snap: int = 700,
             min_spacing_in_ell_d: float = 1.0,
             cfl: float = 0.4,
             allow_cracks: bool = True,
             verbose: bool = False) -> dict:
    """Integrate the release problem, letting new cracks form and split it.

    Returns a dict with the space-time strain snapshots ``e_xt`` (shape
    ``(n_el, n_snap)``), the element centres ``x_el``, the snapshot times
    ``t_snap``, the list of ``cracks`` (position, time, generation) and the
    energy histories.
    """
    # ---- geometry / material from the note's control numbers ---------------
    L      = ell_e / float(Lambda_bar)
    tau_rt = 2.0 * np.sqrt(2.0) * L / c
    gamma  = float(Gamma) / tau_rt
    m      = E_h / c ** 2
    k      = E_h / ell_e ** 2                 # Winkler foundation
    eta    = 2.0 * m * gamma                  # note's viscous coefficient
    ell_d  = damage_band_width(e_crit)
    min_sp = min_spacing_in_ell_d * ell_d

    # ---- grid ---------------------------------------------------------------
    n_el = max(60, int(round(cells_per_ell_e * L / ell_e)))
    h    = L / n_el
    n_nd = n_el + 1
    x_nd = np.linspace(0.0, L, n_nd)
    x_el = 0.5 * (x_nd[:-1] + x_nd[1:])

    # lumped nodal volumes (half a cell at each end)
    V = np.full(n_nd, h); V[0] = V[-1] = 0.5 * h
    M, C, Kf = m * V, eta * V, k * V

    intact = np.ones(n_el)                    # 1 = load-carrying, 0 = cracked
    sym_intact = True                         # is the symmetry plane still u=0 ?

    u = np.zeros(n_nd)
    v = np.zeros(n_nd)

    # ---- explicit time step from the discrete cut-off frequency ------------
    omega0    = c / ell_e
    omega_max = np.sqrt(omega0 ** 2 + 4.0 * c ** 2 / h ** 2)
    t_end   = n_roundtrips * tau_rt
    dt      = cfl * 2.0 / omega_max
    n_steps = int(np.ceil(t_end / dt))
    dt      = t_end / n_steps

    def strain(u_):
        return (u_[1:] - u_[:-1]) / h - theta

    def internal_force(u_):
        N = E_h * strain(u_) * intact         # broken element carries nothing
        F = np.zeros(n_nd)
        F[:-1] -= N
        F[1:]  += N
        return F

    def accel(u_, v_):
        a_ = -(internal_force(u_) + Kf * u_ + C * v_) / M
        if sym_intact:
            a_[-1] = 0.0
        return a_

    # ---- crack bookkeeping --------------------------------------------------
    cracks = [{"x": 0.0, "t": 0.0, "gen": 0}]          # the first crack
    crack_x = [0.0]
    E_removed = 0.0        # elastic energy destroyed by breaking elements

    def _generation(xc):
        """Causal attribution: the *nearest* existing crack is the most likely
        source of the release wave that broke this spot, so the new crack sits
        one generation after it."""
        d = [abs(xc - cr["x"]) for cr in cracks]
        return cracks[int(np.argmin(d))]["gen"] + 1

    def try_crack(t_now):
        """Break every element that is over threshold and far enough from the
        existing cracks (strongest first).  Returns the number created."""
        nonlocal sym_intact, E_removed
        e_ = np.abs(strain(u)) * intact
        cand = np.nonzero(e_ >= e_crit)[0]
        if cand.size == 0:
            return 0
        made = 0
        for j in cand[np.argsort(-e_[cand])]:
            xc = x_el[j]
            if min(abs(xc - xx) for xx in crack_x) < min_sp:
                continue
            # a crack at the last element IS a crack on the symmetry plane:
            # release the constraint instead of breaking the element, so the
            # fragment becomes free-free (a real crack of the full specimen).
            if j == n_el - 1 and sym_intact:
                if min(abs(L - xx) for xx in crack_x) < min_sp:
                    continue                     # keep the constraint in place
                sym_intact, xc = False, L        # the symmetry plane cracks
            else:
                E_removed += 0.5 * E_h * (strain(u)[j]) ** 2 * h
                intact[j] = 0.0
            gen = _generation(xc)
            crack_x.append(xc)
            cracks.append({"x": float(xc), "t": float(t_now), "gen": gen})
            made += 1
        return made

    # ---- storage ------------------------------------------------------------
    every = max(1, n_steps // n_snap)
    e_snap, t_snap = [], []
    u_snap = []          # ADDED: the displacement field, for the figure
                         # that shows the jump opening at a crack face
    hist = {"t": [], "K": [], "P_el": [], "P_f": [], "S": [], "D": [], "total": [],
            "n_cracks": []}
    E_diss = 0.0

    def record(t_now):
        # a broken element carries no stress: its kinematic strain is
        # meaningless (it grows without bound as the fragments drift apart),
        # so report the *stress-producing* strain everywhere.
        e_ = strain(u) * intact
        P_el = 0.5 * E_h * np.sum(e_ ** 2) * h
        P_f  = 0.5 * k * np.sum(V * u ** 2)
        K    = 0.5 * m * np.sum(V * v ** 2)
        S    = E_removed                       # elastic energy destroyed at the crack faces
        hist["t"].append(t_now); hist["K"].append(K); hist["P_el"].append(P_el)
        hist["P_f"].append(P_f); hist["S"].append(S); hist["D"].append(E_diss)
        hist["total"].append(K + P_el + P_f + S + E_diss)
        hist["n_cracks"].append(len(cracks))
        e_snap.append(e_.copy()); t_snap.append(t_now)
        u_snap.append(u.copy())          # ADDED

    # ---- time loop (velocity Verlet, damping solved exactly) ---------------
    a = accel(u, v)
    record(0.0)
    t_cur = 0.0
    for step in range(1, n_steps + 1):
        v_half = v + 0.5 * dt * a
        u = u + dt * v_half
        if sym_intact:
            u[-1] = 0.0
        t_cur += dt

        # dissipated power  int eta*v^2 dx  (trapezoid in time via v_half)
        E_diss += dt * eta * float(np.sum(V * v_half ** 2))

        if allow_cracks:
            n_new = try_crack(t_cur)
            if n_new and verbose:
                print(f"    crack(s) at t/tau_rt={t_cur/tau_rt:.3f}: "
                      f"{[round(cr['x']/ell_e, 2) for cr in cracks[-n_new:]]}")

        F = internal_force(u)
        rhs = v_half - 0.5 * dt * (F + Kf * u) / M
        v = rhs / (1.0 + 0.5 * dt * C / M)
        if sym_intact:
            v[-1] = 0.0
        a = -(F + Kf * u + C * v) / M
        if sym_intact:
            a[-1] = 0.0

        if step % every == 0 or step == n_steps:
            record(t_cur)

    out = {"Lambda_bar": Lambda_bar, "Gamma": Gamma, "e_crit": e_crit,
           "L": L, "tau_rt": tau_rt, "gamma": gamma, "ell_d": ell_d,
           "ell_e": ell_e, "theta": theta, "n_el": n_el, "dt": dt,
           "x_el": x_el, "t_snap": np.array(t_snap),
           "e_xt": np.array(e_snap).T,                 # (n_el, n_snap)
           "x_nd": x_nd,                               # ADDED
           "u_xt": np.array(u_snap).T,                 # ADDED (n_nd, n_snap)
           "cracks": cracks,
           "n_secondary": len(cracks) - 1,
           "n_generations": (max(cr["gen"] for cr in cracks) if len(cracks) > 1 else 0),
           "hist": {kk: np.array(vv) for kk, vv in hist.items()}}
    if verbose:
        print(f"  [wave] Lambda={Lambda_bar:.3f} Gamma={Gamma:.3f} -> "
              f"{out['n_secondary']} secondary crack(s), "
              f"{out['n_generations']} generation(s)")
    return out


def verify_against_modal(Lambda_bar: float = 0.3, Gamma: float = 0.4,
                         **kw) -> dict:
    """Cross-check the direct solver against the modal solution of
    :mod:`problems.secondary` with cracking switched **off** (same problem)."""
    from problems import secondary as sc

    w = simulate(Lambda_bar, Gamma, allow_cracks=False, **kw)
    r = sc.run_problem({"Lambda_bar": Lambda_bar, "Gamma": Gamma, "e_crit": 1e9,
                        "n_modes": 400, "n_x": 600, "n_t": 1200,
                        "n_roundtrips": 4.0}, plot=False, verbose=False)
    # Compare the ELASTIC ENERGY history, 0.5*int e^2 dx.  It is smooth and
    # convergent, unlike the pointwise field: the two solutions differ visibly
    # only at the sharp release front, which is precisely where the truncated
    # cosine series rings (the Gibbs effect of section 10).
    trapz = getattr(np, "trapezoid", np.trapz)
    tg = np.linspace(0.0, min(w["t_snap"][-1], r["t"][-1]), 120)
    Ew = np.interp(tg, w["t_snap"], w["hist"]["P_el"])
    Em = np.interp(tg, r["t"], 0.5 * trapz(r["e_xt"] ** 2, r["x"], axis=0))
    rel = np.max(np.abs(Ew - Em)) / np.max(np.abs(Em))
    return {"t": tg, "E_wave": Ew, "E_modal": Em, "rel_err": float(rel)}


if __name__ == "__main__":
    chk = verify_against_modal()
    print("max relative difference to the modal solution:", f"{chk['rel_err']:.3%}")
    simulate(0.15, 0.1, verbose=True)
