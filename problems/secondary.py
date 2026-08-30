"""
problems/secondary.py
=====================
**Semi-analytic** study of secondary cracking after a sudden fracture event
(concept note by A.L.B., 1st June 2026).

Question
--------
When a crack suddenly nucleates in a thin film bonded to an elastic substrate
(Winkler foundation), can the stress-release shock generate further,
*secondary*, cracks?

Model (1D, half-specimen ``[0, ell]``, crack face at ``x = 0``)
---------------------------------------------------------------
Post-crack dynamics of the film displacement ``u``:

.. math::

    m u_{tt} + \\eta u_t - E_h u_{xx} + k u = 0,

with the derived quantities

.. math::

    c = \\sqrt{E_h/m}, \\quad \\ell_e = \\sqrt{E_h/k}, \\quad
    \\omega_0 = c/\\ell_e, \\quad \\gamma = \\eta/(2m).

Everything here is solved by **modal expansion** -- there is no finite-element
machinery in this file, only numpy.  The pipeline follows the note:

1.  *Static release*: relaxed one-crack equilibrium ``u^+(x)`` (traction-free
    crack face ``e(0)=0``, compatibility ``u(ell)=0``), released energy
    ``dE_1/2 = (1/2) E_h theta^2 ell_e tanh(ell/ell_e)``.
2.  *Modal decomposition*: project the initial mismatch ``v(x,0) = -u^+(x)``
    on ``phi_n = cos(q_n x)``, ``q_n = (n+1/2) pi / ell``; dispersion
    ``omega_n^2 = omega_0^2 + c^2 q_n^2``; modal energy quota ``Q_n``.
3.  *Propagation + damping*: damped modal evolution ``a_n(t)``; the damping
    number ``Gamma = gamma * tau_rt`` (round-trip time
    ``tau_rt ~ 2 sqrt(2) ell / c``) decides whether waves survive a reflection.
4.  *Secondary-cracking criterion*: the dynamic elastic strain
    ``e(x,t) = u_x - theta`` is compared with the damage-nucleation threshold
    ``e_crit`` (from the phase-field model: ``e_crit^2 = 2 G_c w'(0) /
    (E_h ell_d |g'(0)|)``, see :func:`tools.solvers.critical_strain`).
    A secondary crack is *triggered* when ``|e(x,t)| >= e_crit`` somewhere
    away from the (already relaxed) first crack.

Sign convention follows the corrected concept note
(``secondary_cracking_corrected.pdf``, ALB): the pre-crack film is uniformly
pre-stressed with ZERO displacement ``u_before == 0`` (so ``e_0 = -theta``,
``N_0 = -E_h theta``), and the relaxed post-crack profile satisfies
``u^+(ell) = 0`` (recovers the pre-crack state far from the crack).  Because
every energy is quadratic in ``theta`` and the criterion uses ``|e|``, none of
the final numbers depend on this sign -- but the formulas now read
line-for-line against the PDF (eqs 6, 7, 9, 16, 18, 20, 21).

Non-dimensionalisation: ``ell_e = c = theta = E_h = m = 1``; the three control
parameters are then

    * ``Lambda_bar = ell_e/ell``   -- transfer length vs specimen length
      (small ``Lambda_bar`` = long film, only a boundary layer relaxes);
    * ``Gamma``                    -- damping number ``gamma * tau_rt``;
    * ``l_hat = ell_d/ell``        -- damage internal length vs specimen length;
      it sets the nucleation threshold ``e_crit ~ 1/sqrt(ell_d)`` and hence how
      close the pre-stress already sits to cracking.

See ``problems/secondary_theory.ipynb`` for the full derivations.

NOTE on parameters: the parameter dictionary :data:`SECONDARY_PARAMETERS` is
defined *at the end of this file*, NOT in :mod:`tools.parameters`, because
several symbols of the note clash with repository-wide names (``eta``,
``Lambda``, ``gamma`` -- see the comment block above the dictionary).
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.solvers import critical_strain, damage_band_width
from tools.plotting import (plot_secondary_run, plot_secondary_regime_map,
                            plot_secondary_regime_summary)

# np.trapz is being renamed to np.trapezoid in recent numpy
_trapz = getattr(np, "trapezoid", np.trapz)


# =============================================================================
# Step 1 -- static release
# =============================================================================
def relaxed_displacement(x, theta, ell, ell_e):
    """Relaxed one-crack equilibrium ``u^+(x)`` of ``-E_h u_xx + k u = 0``.

    PDF eq (6).  Boundary conditions: traction-free crack face,
    ``u^+_x(0) = +theta`` (so the *elastic* strain ``e = u_x - theta`` vanishes
    at the crack), and far-end compatibility ``u^+(ell) = 0``::

        u^+(x) = -theta ell_e sinh((ell-x)/ell_e) / cosh(ell/ell_e).
    """
    return -theta * ell_e * np.sinh((ell - x) / ell_e) / np.cosh(ell / ell_e)


def relaxed_strain(x, theta, ell, ell_e):
    """Elastic strain ``e^+(x) = u^+_x - theta`` of the relaxed equilibrium.

    PDF eq (7).  ``e^+(0) = 0`` (traction-free, ``N = 0``) and ``e^+ -> -theta``
    over the transfer length ``ell_e`` (recovering the uniform pre-stress)::

        e^+(x) = theta [ cosh((ell-x)/ell_e) / cosh(ell/ell_e) - 1 ].
    """
    return theta * (np.cosh((ell - x) / ell_e) / np.cosh(ell / ell_e) - 1.0)


def released_energy(theta, ell, ell_e, E_h=1.0):
    """Released elastic energy on one half-specimen,

    ``dE_1/2 = (1/2) E_h theta^2 ell_e tanh(ell/ell_e)``.

    Regimes: ``~ (1/2) E_h theta^2 ell`` for ``ell << ell_e`` (whole film
    relaxes) and ``~ (1/2) E_h theta^2 ell_e`` for ``ell >> ell_e`` (only a
    boundary layer of size ``ell_e`` relaxes).
    """
    return 0.5 * E_h * theta ** 2 * ell_e * np.tanh(ell / ell_e)


# =============================================================================
# Step 2 -- modal decomposition
# =============================================================================
def modal_setup(ell, ell_e, c, n_modes):
    """Wavenumbers and frequencies of the eigenfunctions ``cos(q_n x)``.

    ``q_n = (n + 1/2) pi / ell`` (from ``v_x(0)=0``, ``v(ell)=0``) and the
    Winkler dispersion relation ``omega_n^2 = omega_0^2 + c^2 q_n^2`` with the
    substrate cut-off ``omega_0 = c / ell_e``.
    """
    n = np.arange(n_modes)
    q = (n + 0.5) * np.pi / ell
    omega0 = c / ell_e
    omega = np.sqrt(omega0 ** 2 + (c * q) ** 2)
    return q, omega


def modal_coefficients(theta, ell, ell_e, q, n_quad=4001):
    """Coefficients ``a_n(0)`` of the initial mismatch ``v(x,0) = -u^+(x)``.

    ``a_n(0) = (2/ell) int_0^ell  v(x,0) cos(q_n x) dx`` (the modes are
    orthogonal with norm ``ell/2``).  Quadrature is plain trapezoidal on a
    dense grid -- transparent and accurate for these smooth integrands.

    PDF eq (16): the integral is exact and gives the *positive* Lorentzian
    ``a_n(0) = 2 theta ell_e^2 / (ell (1 + (q_n ell_e)^2))`` -- no ``(-1)^n``
    factor, matching the note's filter ``A_n ~ 1/(1 + (q_n ell_e)^2)``.
    """
    x = np.linspace(0.0, ell, n_quad)
    v0 = -relaxed_displacement(x, theta, ell, ell_e)
    return np.array([(2.0 / ell) * _trapz(v0 * np.cos(qi * x), x) for qi in q])


def modal_energy_quota(omega, a0):
    """Modal energy quota ``Q_n = omega_n^2 a_n^2 / sum_j omega_j^2 a_j^2``.

    Plotted against ``beta_n = q_n ell_e`` it shows which wavelengths carry
    the released energy (the release profile is localised over ``ell_e``, so
    short wavelengths ``beta_n >> 1`` are strongly suppressed).
    """
    w = omega ** 2 * a0 ** 2
    return w / np.sum(w)


# =============================================================================
# Step 3 -- propagation and damping
# =============================================================================
def modal_evolution(a0, omega, gamma_bar, t):
    """Damped free oscillation of each mode, released from rest.

    ``a_n(t) = a_n(0) e^{-gamma t} [cos(omega_d t) + (gamma/omega_d)
    sin(omega_d t)]`` with ``omega_d = sqrt(omega_n^2 - gamma^2)``.  The
    complex square root makes the same formula valid in the overdamped case.
    Returns the array ``a_nt`` of shape ``(n_modes, n_t)``.
    """
    om_d = np.sqrt((omega ** 2 - gamma_bar ** 2).astype(complex))
    phase = om_d[:, None] * t[None, :]
    osc = np.cos(phase) + (gamma_bar / om_d)[:, None] * np.sin(phase)
    return a0[:, None] * np.exp(-gamma_bar * t)[None, :] * np.real(osc)


def strain_field(x, t, theta, ell, ell_e, q, a_nt):
    """Dynamic elastic strain ``e(x,t) = e^+(x) + v_x(x,t)`` on the grid.

    PDF eq (20).  ``v_x(x,t) = - sum_n q_n a_n(t) sin(q_n x)``.  Returns shape
    ``(n_x, n_t)``.  At ``t=0`` it reproduces the uniform pre-crack strain
    ``e_0 = -theta`` (the pre-stress, up to modal truncation).
    """
    sin_qx = np.sin(np.outer(q, x))                       # (n_modes, n_x)
    v_x = -np.einsum("n,nx,nt->xt", q, sin_qx, a_nt)
    return relaxed_strain(x, theta, ell, ell_e)[:, None] + v_x


# =============================================================================
# Step 4 -- secondary-cracking criterion
# =============================================================================
def effective_exclusion(p, ell):
    """Exclusion zone around the first crack, clamped to half the specimen.

    Nominally ``x_exclude * ell_e``; for short films (``ell`` comparable to
    ``ell_e``) that would swallow the whole half-specimen, so it is capped at
    ``ell / 2`` to keep the criterion well-defined for all ``Lambda_bar``.
    """
    return min(p["x_exclude"] * p["ell_e"], 0.5 * ell)



def detect_secondary_trigger(e_xt, e_crit, x, t, x_min=0.0):
    """First space-time point where ``|e(x,t)| >= e_crit``.

    The zone ``x < x_min`` (a few ``ell_e`` around the first crack, where the
    stress is already relaxed and the damage band already exists) is excluded.
    Returns ``{"t", "x", "e"}`` of the first crossing, or ``None`` if the wave
    never reaches the threshold (no secondary crack).
    """
    mask = x >= x_min
    if not mask.any():
        return None
    abs_e = np.abs(e_xt[mask, :])
    hit_t = np.nonzero(abs_e.max(axis=0) >= e_crit)[0]
    if hit_t.size == 0:
        return None
    jt = int(hit_t[0])
    jx = int(np.argmax(abs_e[:, jt]))
    x_allowed = x[mask]
    return {"t": float(t[jt]), "x": float(x_allowed[jx]),
            "e": float(abs_e[jx, jt])}


# =============================================================================
# Main entry point
# =============================================================================
def run_problem(parameters: dict | None = None,
                plot: bool = True,
                output_dir: str | Path | None = None,
                verbose: bool = True) -> dict:
    """Run one semi-analytic secondary-cracking computation.

    ``parameters`` overrides entries of :data:`SECONDARY_PARAMETERS`.
    Returns a result dictionary (grids, modal data, strain field, trigger).
    """
    p = dict(SECONDARY_PARAMETERS)
    if parameters:
        p.update(parameters)

    ell_e, c, theta, E_h = p["ell_e"], p["c"], p["theta"], p["E_h"]
    ell = ell_e / p["Lambda_bar"]           # Lambda_bar = ell_e / ell

    # Damage internal length as a fraction of the specimen: l_hat = ell_d / ell.
    l_hat = p["l_hat"]
    ell_d = l_hat * ell

    # Round-trip time of the energy packet (group velocity ~ c/sqrt(2) for
    # the representative component q ~ 1/ell_e) and modal damping rate.
    tau_rt = 2.0 * np.sqrt(2.0) * ell / c
    gamma_bar = p["Gamma"] / tau_rt

    # Damage-nucleation threshold: explicit value, or derived from the
    # phase-field model (AT1 has a finite elastic phase; AT2 gives 0).
    # e_crit ~ 1/sqrt(ell_d), so it moves with l_hat (and with Lambda_bar
    # through ell = ell_e / Lambda_bar).
    e_crit = p["e_crit"]
    if e_crit is None:
        e_crit = critical_strain(p["model"], Gc=p["Gc"], E_h=E_h,
                                 ell_d=ell_d)

    if verbose:
        print(f"\n[SECONDARY | semi-analytic]  Lambda_bar={p['Lambda_bar']:g},"
              f" Gamma={p['Gamma']:g}, l_hat={l_hat:g}  ->  ell={ell:g},"
              f" ell_d={ell_d:g}, tau_rt={tau_rt:.3g}, gamma={gamma_bar:.3g},"
              f" e_crit/theta={e_crit / theta:.3g}, n_modes={p['n_modes']}")

    # ---- steps 1-3 ----------------------------------------------------------
    q, omega = modal_setup(ell, ell_e, c, p["n_modes"])
    a0 = modal_coefficients(theta, ell, ell_e, q)
    Q = modal_energy_quota(omega, a0)

    t = np.linspace(0.0, p["n_roundtrips"] * tau_rt, p["n_t"])
    x = np.linspace(0.0, ell, p["n_x"])
    a_nt = modal_evolution(a0, omega, gamma_bar, t)
    e_xt = strain_field(x, t, theta, ell, ell_e, q, a_nt)

    # ---- step 4 -------------------------------------------------------------
    x_excl = effective_exclusion(p, ell)
    trigger = detect_secondary_trigger(e_xt, e_crit, x, t, x_min=x_excl)
    if verbose:
        if trigger:
            print(f"  SECONDARY CRACK triggered at t/tau_rt="
                  f"{trigger['t'] / tau_rt:.3f}, x/ell_e="
                  f"{trigger['x'] / ell_e:.2f} (|e|/e_crit="
                  f"{trigger['e'] / e_crit:.3f})")
        else:
            print("  no secondary crack (wave damped below threshold)")

    result = {
        "parameters":  p,
        "ell":         ell,
        "ell_d":       ell_d,
        "l_hat":       l_hat,
        "tau_rt":      tau_rt,
        "gamma_bar":   gamma_bar,
        "e_crit":      float(e_crit),
        "x":           x,
        "t":           t,
        "q":           q,
        "omega":       omega,
        "beta_n":      q * ell_e,
        "a0":          a0,
        "Q_n":         Q,
        "e_xt":        e_xt,
        "e_plus":      relaxed_strain(x, theta, ell, ell_e),
        "dE_half":     released_energy(theta, ell, ell_e, E_h),
        "x_excl":      x_excl,
        "trigger":     trigger,
    }

    if plot:
        png, pdf = plot_secondary_run(result, output_dir or ROOT / "output")
        if verbose:
            print(f"  saved {png}")
            print(f"        {pdf}")
    return result


def trigger_map(Lambda_bars, Gammas, l_hats=None,
                base_parameters: dict | None = None,
                plot: bool = True,
                output_dir: str | Path | None = None,
                verbose: bool = True) -> dict:
    """Sweep the ``(Lambda_bar, Gamma, l_hat)`` cube and record the *trigger
    margin*

    ``margin = max_{x,t} |e(x,t)| / e_crit``   (margin >= 1  <=>  secondary
    crack).  For each value of ``l_hat`` the ``(Lambda_bar, Gamma)`` plane is
    swept, so the result is a stack of regime maps.  Returns a dict with the
    grids and the 3-D margin array ``margin[h, i, j]`` (indices
    ``l_hat, Gamma, Lambda_bar``).

    When ``plot`` is true this emits **one regime-map figure per ``l_hat``**
    (nine files for nine values) plus a single *summary* figure that overlays
    the ``R_crack = 1`` boundaries of every ``l_hat`` in the same
    ``(Lambda_bar, Gamma)`` plane.
    """
    Lambda_bars = np.asarray(Lambda_bars, dtype=float)
    Gammas = np.asarray(Gammas, dtype=float)
    if l_hats is None:                       # back-compat: single default l_hat
        l_hats = [SECONDARY_PARAMETERS["l_hat"]]
    l_hats = np.atleast_1d(np.asarray(l_hats, dtype=float))
    margin = np.zeros((l_hats.size, Gammas.size, Lambda_bars.size))

    base = dict(base_parameters or {})
    # coarser resolution for the sweep unless the caller overrides it
    base.setdefault("n_modes", 120)
    base.setdefault("n_x", 200)
    base.setdefault("n_t", 400)

    for h, lh in enumerate(l_hats):
        for i, G in enumerate(Gammas):
            for j, L in enumerate(Lambda_bars):
                r = run_problem({**base, "Lambda_bar": float(L),
                                 "Gamma": float(G), "l_hat": float(lh)},
                                plot=False, verbose=False)
                xm = r["x"] >= r["x_excl"]
                margin[h, i, j] = np.abs(r["e_xt"][xm, :]).max() / r["e_crit"]
        if verbose:
            print(f"  [trigger_map] l_hat={lh:g} done "
                  f"({h + 1}/{l_hats.size})")

    out = {"Lambda_bar": Lambda_bars, "Gamma": Gammas, "l_hat": l_hats,
           "margin": margin, "base_parameters": base}
    if plot:
        out_dir = output_dir or ROOT / "output"
        # one (Lambda_bar, Gamma) regime map per l_hat
        for h, lh in enumerate(l_hats):
            single = {"Lambda_bar": Lambda_bars, "Gamma": Gammas,
                      "l_hat": float(lh), "margin": margin[h],
                      "base_parameters": base}
            png, pdf = plot_secondary_regime_map(single, out_dir)
            if verbose:
                print(f"  saved {png}")
                print(f"        {pdf}")
        # one paper-style summary overlaying every R_crack = 1 boundary
        png, pdf = plot_secondary_regime_summary(out, out_dir)
        if verbose:
            print(f"  saved {png}")
            print(f"        {pdf}")
    return out


# =============================================================================
# Parameters -- defined HERE, not in tools/parameters.py
# =============================================================================
# Several symbols of the concept note clash with repository-wide parameter
# names, so this problem keeps its own dictionary (as agreed):
#
#   * ``eta``    -- in tools.parameters: dynamic *loading time-scale*
#                   (tau = eta * t).  In the note: *viscous damping
#                   coefficient* of  m u_tt + eta u_t - E_h u_xx + k u = 0.
#                   Here we only ever use gamma = eta/(2m), via Gamma.
#   * ``Lambda`` -- in tools.parameters: foundation stiffness.  In the note:
#                   Lambda = ell / ell_e.  Here named ``Lambda_bar``.
#   * ``gamma``  -- in tools.parameters (Newmark): time-integration
#                   coefficient.  In the note: modal damping rate eta/(2m).
#   * ``ell``    -- the half-specimen length, NOT the phase-field
#                   regularisation length ``l_hat`` (the latter corresponds
#                   to ``ell_d`` in the note's damage criterion).
SECONDARY_PARAMETERS = {
    # ---- non-dimensionalisation -----------
    "ell_e":   1.0,     # elastic transfer length  sqrt(E_h / k)
    "c":       1.0,     # axial wave speed         sqrt(E_h / m)
    "theta":   1.0,     # imposed pre-crack homogeneous strain
    "E_h":     1.0,     # axial film stiffness     E_f * h_f

    # ---- the three control parameters of the note ---------------------------
    "Lambda_bar":   1.05,     # Lambda = ell_e / ell  (transfer vs specimen length)
    "Gamma":        0.6,    # damping number Gamma = gamma * tau_rt
    "l_hat":        0.1,     # l_hat = ell_d / ell  (damage vs specimen length)

    # ---- damage-nucleation threshold ----------------------------------------
    # e_crit / theta < 1 would have cracked statically; the interesting window
    # is 1 < e_crit/theta < ~2 (dynamic overshoot from reflection).  When
    # e_crit is None it is derived from l_hat via ell_d = l_hat * ell.
    "e_crit":  None,        # explicit threshold; set to None to derive it from
    "model":   "AT1",       # the phase-field model via critical_strain(...)
    "Gc":      1.0,         # fracture toughness   (used only when e_crit=None)

    # ---- numerics ------------------------------------------------------------
    "n_modes":       200,   # modal truncation
    "n_x":           400,   # spatial samples on [0, ell]
    "n_t":           800,   # time samples
    "n_roundtrips":  4.0,   # simulated time in units of tau_rt
    "x_exclude":     2.0,   # exclusion zone near the first crack, in ell_e
}


# =============================================================================
# Stand-alone execution
# =============================================================================
if __name__ == "__main__":
    # Single run with the default (Lambda_bar, Gamma, l_hat)
    run_problem()

    # Regime maps: does the released wave re-crack the film?  One
    # (Lambda_bar, Gamma) map per l_hat (nine files) + one summary figure.
    trigger_map(Lambda_bars=np.linspace(0.1, 2.0, 17),
                Gammas=np.linspace(0.0, 2.0, 17),
                l_hats=np.geomspace(0.01, 1.0, 9))


# =============================================================================
# Refinements of the criterion  (see secondary_theory.ipynb §10-§12)
# =============================================================================
# The pointwise test of `detect_secondary_trigger` has a defect: its maximum
# always sits at the symmetry end x = ell, where sin(q_n ell) = +-1 for EVERY
# mode while the strain coefficients decay only like 1/q_n.  The series is then
# merely conditionally convergent (a Gibbs phenomenon), so `max|e|` keeps
# drifting as modes are added and no converged number can be quoted.
#
# The cure is the physics the pointwise test left out: the phase-field model
# also pays the gradient term Gc*ell_d*|alpha_x|^2, so a real crack is a damage
# BAND of width ~ell_d and a spike narrower than that cannot nucleate anything.
# Filtering the transient over ell_d removes the offending high-q tail and
# makes the margin converge exactly.  With AT1 and Gc = E_h = 1 the width is
# tied to the threshold, ell_d = 1/e_crit^2, so this adds NO free parameter.

# cheap-but-converged resolution (the ell_d filter kills the high modes)
FAST_PARAMETERS = {"n_modes": 80, "n_x": 400, "n_t": 2000, "n_roundtrips": 4.0}


def strain_nonlocal(result, ell_d=None):
    """Strain field low-pass filtered over the damage-band width ``ell_d``.

    The filter is applied in **modal space** -- mode ``n`` is multiplied by
    ``exp(-(q_n ell_d/2)^2 / 2)``, the transform of a Gaussian band of standard
    deviation ``ell_d/2``.  Doing it modally (rather than convolving on the
    grid) avoids any boundary artefact at the symmetry end ``x = ell``, which
    is precisely where the peak sits.  Only the *transient* is filtered: the
    static profile ``e_plus`` varies over ``ell_e`` and carries no high-q part.
    """
    if ell_d is None:
        ell_d = damage_band_width(result["e_crit"])
    q, x, t = result["q"], result["x"], result["t"]
    a_nt = modal_evolution(result["a0"], result["omega"], result["gamma_bar"], t)
    filt = np.exp(-0.5 * (q * (ell_d / 2.0)) ** 2)
    v_x = -np.einsum("n,nx,nt->xt", q * filt, np.sin(np.outer(q, x)), a_nt)
    return result["e_plus"][:, None] + v_x


def R_local(result):
    """Pointwise trigger margin ``max|e|/e_crit`` -- NOT modally convergent."""
    m = result["x"] >= result["x_excl"]
    return float(np.abs(result["e_xt"][m, :]).max() / result["e_crit"])


def R_nonlocal(result, ell_d=None):
    """Trigger margin on the ``ell_d``-filtered strain -- converges exactly."""
    e = strain_nonlocal(result, ell_d)
    m = result["x"] >= result["x_excl"]
    return float(np.abs(e[m, :]).max() / result["e_crit"])


def crack_site(result, ell_d=None):
    """Position ``x`` of the non-local overshoot maximum (the next crack)."""
    e = strain_nonlocal(result, ell_d)
    m = result["x"] >= result["x_excl"]
    env = np.abs(e[m, :]).max(axis=1)
    return float(result["x"][m][int(np.argmax(env))])


def run_fast(Lambda_bar, Gamma, e_crit=1.3, **over):
    """One run at the cheap-but-converged resolution of :data:`FAST_PARAMETERS`."""
    p = {**FAST_PARAMETERS, "Lambda_bar": float(Lambda_bar),
         "Gamma": float(Gamma), "e_crit": float(e_crit)}
    p.update(over)
    return run_problem(p, plot=False, verbose=False)


def margin_fields(Lambda_bars, Gammas, e_crit=1.3, verbose=True, **over):
    """Both margins over the ``(Lambda, Gamma)`` plane -> ``(R_loc, R_nl)``,
    each of shape ``(len(Gammas), len(Lambda_bars))``.

    ``**over`` overrides the resolution of :data:`FAST_PARAMETERS`.  A map
    is a sweep, not a converged single point: it is worth trading time
    samples for grid points, and the ``ell_d`` filter removes exactly the
    high modes that would demand the finer time step."""
    Lambda_bars = np.asarray(Lambda_bars, float)
    Gammas = np.asarray(Gammas, float)
    Rl = np.zeros((Gammas.size, Lambda_bars.size))
    Rn = np.zeros_like(Rl)
    for i, G in enumerate(Gammas):
        for j, L in enumerate(Lambda_bars):
            r = run_fast(L, G, e_crit, **over)
            Rl[i, j] = R_local(r)
            Rn[i, j] = R_nonlocal(r)
        if verbose:
            print(f"  [margin_fields] Gamma row {i + 1}/{Gammas.size} done")
    return Rl, Rn


def overshoot_undamped(Lambda_bars, e_crit=1.3, nonlocal_=True, verbose=True,
                       **over):
    """``A(Lambda) = max|e|/theta`` at ``Gamma = 0`` -- the ceiling on the
    dynamic overshoot.  A second crack is possible **only if**
    ``e_crit/theta < A(Lambda)``; no damping can rescue a tougher film."""
    Lambda_bars = np.asarray(Lambda_bars, float)
    A = np.zeros_like(Lambda_bars)
    for k, L in enumerate(Lambda_bars):
        r = run_fast(L, 0.0, e_crit, **over)
        th = r["parameters"]["theta"]
        if nonlocal_:
            e = strain_nonlocal(r)
            m = r["x"] >= r["x_excl"]
            A[k] = np.abs(e[m, :]).max() / th
        else:
            A[k] = R_local(r) * r["e_crit"] / th
        if verbose and (k + 1) % 5 == 0:
            print(f"  [overshoot] {k + 1}/{Lambda_bars.size}")
    return A


def l_hat_min(Lambda_bars, A):
    """Smallest damage-length ratio for which a second crack is still possible.

    With ``ell_e = theta = 1``: ``ell = 1/Lambda``, ``ell_d = l_hat*ell`` and
    ``e_crit = 1/sqrt(ell_d) = sqrt(Lambda/l_hat)``.  Requiring
    ``e_crit < A`` gives ``l_hat > Lambda / A**2``.
    """
    return np.asarray(Lambda_bars, float) / np.asarray(A, float) ** 2


def cascade(F0=32.0, gamma_phys=0.005, e_crit=1.3, max_gen=8,
            ell_e=1.0, c=1.0, verbose=True):
    """Recursive fragmentation from one fragment of **full** length ``F0``.

    Fixed: ``ell_e``, the physical damping rate ``gamma_phys`` and ``e_crit``.
    A fragment of full length ``F`` is free at both ends, so its mirror plane
    sits at ``F/2`` and it is analysed as a half-specimen of length ``L = F/2``:
    ``Lambda = ell_e/L`` and ``Gamma = gamma_phys * 2*sqrt(2) * L / c``.
    A shrinking fragment therefore has a *larger* ``Lambda`` (less stored
    energy) but a *smaller* ``Gamma`` (less damping per round trip); the
    cascade stops when the first effect wins.

    NOTE: each generation is analysed as a *fresh* problem (uniform pre-stress,
    zero velocity).  In reality the daughters inherit the parent's vibration,
    so this predicts the saturation size well but not a literal time history --
    see :mod:`problems.secondary_wave` for the direct simulation.
    """
    ell_d = damage_band_width(e_crit)
    F_min = 4.0 * ell_d
    frags = [float(F0)]
    history = [list(frags)]
    events = []

    for gen in range(max_gen):
        new, cracked = [], 0
        for F in frags:
            if F < F_min:
                new.append(F)
                continue
            L = 0.5 * F
            Lam = ell_e / L
            Gam = gamma_phys * 2.0 * np.sqrt(2.0) * L / c
            r = run_fast(Lam, Gam, e_crit)
            R = R_nonlocal(r)
            if R >= 1.0:
                xs = crack_site(r)
                if L - xs < ell_d:              # one crack on the mirror plane
                    pieces = [L, L]
                else:                           # two mirror cracks
                    pieces = [xs, F - 2.0 * xs, xs]
                new += [p for p in pieces if p > 1e-9]
                cracked += 1
                events.append({"gen": gen, "F": F, "Lambda": Lam, "Gamma": Gam,
                               "R": R, "x_crack": xs})
            else:
                new.append(F)
        history.append(list(new))
        if verbose:
            print(f"  [cascade] generation {gen}: {len(frags)} -> {len(new)} "
                  f"fragments ({cracked} cracked)")
        if cracked == 0:
            break
        frags = new

    return {"F0": F0, "gamma_phys": gamma_phys, "e_crit": e_crit,
            "ell_d": ell_d, "history": history, "final": np.array(frags),
            "events": events}
