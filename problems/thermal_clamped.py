"""
problems/thermal_clamped.py
===========================
`Internship-2026-main/problems/thermal.py` with **both ends clamped**, so that
the thermal problem can be run *without* the elastic foundation.

Why this file exists.  The trusted thermal problem leaves the displacement free
at both ends and relies on the foundation to remove the rigid-body mode.  Set
Lambda = 0 there and the bar expands freely: u' = theta, the elastic strain
e = u' - theta vanishes, and no damage appears at any load (checked -- alpha is
identically zero up to theta = 2).  The step in the narrative that this file
reproduces is the *constrained* bar: clamped at both ends, heated by a ramp, no
foundation.  The stress is then uniform, e = -theta, and AT1 stays on its
homogeneous branch

    alpha_h(theta) = 1 - e_c^2 / theta^2 ,   e_c = sqrt(3/8) ,

which is a *plateau*: damage grows everywhere at once and nothing localises.
That is the observation the foundation then destroys -- it is the foundation,
not the thermal load, that produces a crack pattern.

This file is a verbatim copy of the trusted `thermal.py` with ONE hunk changed
(the boundary-condition block, marked in place).  The clamp is ON by default;
pass `model_parameters["clamped_ends"] = False` to recover the free bar of
`thermal.py` (which produces no stress and no damage at Lambda = 0).
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.imports import (
    np, sp, ufl, fem, MPI, PETSc, comm,
    Path, tqdm,
)
from tools.helpers   import (
    SNESProblem, make_linear_snes, make_damage_snes, alt_min_loop, print_mesh_info,
    make_crack_counter, detect_crack_events,
)
from tools.meshing   import create_mesh_and_tags, grad_sq, strain, elastic_strain
from tools.solvers   import fracture_energy_density, g_degradation
from tools.parameters import get_defaults
from tools.plotting  import (
    plot_thermal_run, export_paraview, triangulation_from_domain,
)


# =============================================================================
# Problem-specific geometry  (lives in tools.meshing.GEOMETRY_BUILDERS)
# =============================================================================
# Geometry of this problem.  Hard-coded here on purpose: if you ever need a
# different shape (notched bar, L-shape, plate-with-hole, ...), do NOT edit
# this file -- write a new file in problems/ that uses a different value
# from tools.meshing.GEOMETRY_BUILDERS instead.
_PROBLEM_SHAPE = "rectangle"


# =============================================================================
# Loading ramp
# =============================================================================
def _make_thermal_loaders(loading_parameters, eta):
    """Return ``(Theta_qs, Theta_dyn)`` as numpy callables of ``t``:

    * ``Theta_qs(t)  = theta_max * t``                           (linear ramp)
    * ``Theta_dyn(t) = theta_max * (tau/2) * (1 + tanh(tau/T0))``, ``tau = eta*t``
    """
    t_sp     = sp.Symbol("t", real=True)
    T0_v     = loading_parameters["T0"]
    ThMax_v  = loading_parameters["theta_max"]

    Theta_qs_sp  = ThMax_v * t_sp

    tau          = eta * t_sp
    Theta_dyn_sp = ThMax_v * (tau / 2.0) * (1.0 + sp.tanh(tau / T0_v))

    return (sp.lambdify(t_sp, Theta_qs_sp, "numpy"),
            sp.lambdify(t_sp, Theta_dyn_sp, "numpy"))


def _final_times(loading_parameters):
    """Return ``(t_final_qs, tau_final_dyn)``, the pseudo-times at which the
    QS and dynamic ramps stop.

    Without ``theta_target`` this is the old, hard-coded behaviour: QS stops
    at ``t=1`` (load exactly ``theta_max``) and the dynamic ramp stops at
    ``tau=1`` (load ``~94.6%`` of ``theta_max``) -- unchanged, so every
    existing run reproduces bit for bit.

    With ``theta_target`` set (a *load*, not a time, and required to satisfy
    ``0 < theta_target < theta_max`` since the dynamic ramp only
    *approaches* ``theta_max`` as ``tau -> infinity``) both branches instead
    stop at the first pseudo-time each ramp reaches that load:
    ``t_final_qs = theta_target/theta_max`` exactly (the QS ramp is
    linear), and ``tau_final_dyn`` is solved numerically by inverting the
    smoothed ramp -- the ramp's own amplitude ``theta_max`` and shape are
    untouched, only *how far along it* each branch runs changes, so the two
    branches keep stopping at the same physical load as each other.
    """
    theta_target = loading_parameters.get("theta_target")
    if theta_target is None:
        return 1.0, 1.0

    ThMax_v = loading_parameters["theta_max"]
    if not (0.0 < theta_target < ThMax_v):
        raise ValueError(
            f"theta_target={theta_target!r} must satisfy 0 < theta_target < "
            f"theta_max={ThMax_v!r}: the dynamic ramp only approaches "
            f"theta_max as tau -> infinity, it never reaches it at finite "
            f"time."
        )
    t_final_qs = theta_target / ThMax_v

    T0_v = loading_parameters["T0"]

    def _ramp(tau):
        return ThMax_v * (tau / 2.0) * (1.0 + np.tanh(tau / T0_v))

    tau_hi = 1.0
    while _ramp(tau_hi) < theta_target:
        tau_hi *= 2.0
        if tau_hi > 1.0e6:
            raise RuntimeError(
                f"could not bracket tau for theta_target={theta_target!r}; "
                f"check theta_max and T0."
            )
    from scipy.optimize import brentq
    tau_final_dyn = brentq(lambda tau: _ramp(tau) - theta_target,
                            0.0, tau_hi, xtol=1e-13, rtol=1e-13)
    return t_final_qs, tau_final_dyn


def _setup_function_spaces(domain, physics):
    V_alpha = fem.functionspace(domain, ("Lagrange", 1))
    if physics == "1D":
        V_u = fem.functionspace(domain, ("Lagrange", 1))
    else:
        V_u = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    return V_u, V_alpha


def _alpha_max(alpha, comm_):
    local = float(alpha.x.array.max()) if alpha.x.array.size else 0.0
    return float(comm_.allreduce(local, op=MPI.MAX))


# =============================================================================
# Main entry point
# =============================================================================
def run_problem(
    model_parameters: dict,
    mesh_parameters: dict,
    loading_parameters: dict,
    solver_parameters: dict,
    AltMin_parameters: dict,
    Newmark_parameters: dict,
    plot: bool = True,
    paraview: bool = True,
    output_dir: str | Path | None = None,
    verbose: bool = True,
    field_recorder=None,
) -> dict:
    """One *quasi-static + dynamic* thermal fragmentation run.

    ``field_recorder``, when given, is called at every converged step as
    ``field_recorder(branch, load, u_array, alpha_array)`` with ``branch`` in
    ``{"qs", "dyn"}`` -- the hook the run animations of
    :mod:`tools.animations` are built on.
    """
    # Geometry is a property of the *problem*, not a user-tunable knob.
    mesh_parameters["shape"] = _PROBLEM_SHAPE

    physics    = mesh_parameters["physics"]
    model_name = solver_parameters["model"]
    n_qs       = loading_parameters["N_steps_qs"]
    n_dyn      = loading_parameters["N_steps_dyn"]
    # Final pseudo-times for the two ramps.  Defaults (no theta_target given)
    # to the canonical t=1 / tau=1 pair, unchanged from before; see
    # _final_times() for what theta_target changes.
    t_final_qs, tau_final_dyn = _final_times(loading_parameters)
    N_snap      = loading_parameters.get("N_snapshots", 6)

    if verbose and comm.rank == 0:
        print(
            f"\n[THERMAL | {physics} | {model_name}]  "
            f"l_hat={model_parameters['l_hat']}, "
            f"Lambda={model_parameters['Lambda']}, "
            f"eta={model_parameters['eta']}  |  "
            f"mesh_per_lhat={mesh_parameters.get('mesh_per_lhat', 4)}  |  "
            f"n_QS={n_qs}, n_Dyn={n_dyn} (t_dyn_final={tau_final_dyn/float(model_parameters['eta']):.3g})"
        )

    # -------------------------------------------------------------------------
    # Mesh / spaces
    # -------------------------------------------------------------------------
    domain, mt, dx, ds = create_mesh_and_tags(mesh_parameters, model_parameters, comm)
    V_u, V_alpha = _setup_function_spaces(domain, physics)
    count_cracks = make_crack_counter(domain, V_alpha, thr=0.5)
    if verbose:
        print_mesh_info(domain, V_u, V_alpha,
                        label=f"THERM {physics}", comm_=domain.comm)

    u     = fem.Function(V_u, name="Displacement")
    v     = fem.Function(V_u, name="Velocity")
    a     = fem.Function(V_u, name="Acceleration")
    u_new = fem.Function(V_u)
    v_new = fem.Function(V_u)
    a_new = fem.Function(V_u)

    alpha          = fem.Function(V_alpha, name="Damage")
    alpha_old_iter = fem.Function(V_alpha)
    alpha_lb       = fem.Function(V_alpha)
    alpha_ub       = fem.Function(V_alpha)

    # ---- THE ONLY DIFFERENCE FROM problems/thermal.py -----------------------
    # The trusted thermal problem leaves the ends free and relies on the
    # foundation to remove the rigid-body mode.  With Lambda = 0 that mode is
    # back: the bar expands freely, u' = theta, so e = u' - theta = 0 and NO
    # damage ever appears (verified: alpha == 0 for every step up to
    # theta = 2).  Clamping both ends restores a uniform stress e = -theta and
    # with it the homogeneous damage branch -- the "damage plateau".
    bcs_u, bcs_v, bcs_a, bcs_alpha = [], [], [], []
    # Defaults to True: this file exists *only* for the clamped bar, and
    # with the clamp off it silently solves the free bar instead --
    # u' = theta, e = 0, no stress and no damage at any load, which
    # reads as "nothing happens" rather than as a missing flag.
    if model_parameters.get("clamped_ends", True):
        Lx = mesh_parameters.get("Lx", 1.0)
        def _both_ends(x):
            return np.isclose(x[0], 0.0) | np.isclose(x[0], Lx)
        dofs_end = fem.locate_dofs_geometrical(V_u, _both_ends)
        zero = (PETSc.ScalarType(0.0) if physics == "1D"
                else np.zeros(domain.geometry.dim, dtype=PETSc.ScalarType))
        bcs_u = [fem.dirichletbc(zero, dofs_end, V_u)]
        bcs_v = [fem.dirichletbc(zero, dofs_end, V_u)]
        bcs_a = [fem.dirichletbc(zero, dofs_end, V_u)]
        if verbose and comm.rank == 0:
            print(f"  [BC] both ends clamped: {len(dofs_end)} dofs")
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Constants + energies
    # -------------------------------------------------------------------------
    E_c       = fem.Constant(domain, PETSc.ScalarType(model_parameters.get("E_ref", 1.0)))
    nu_c      = fem.Constant(domain, PETSc.ScalarType(model_parameters.get("nu", 0.3)))
    Lambda_c  = fem.Constant(domain, PETSc.ScalarType(model_parameters["Lambda"]))
    l_hat_c   = fem.Constant(domain, PETSc.ScalarType(model_parameters["l_hat"]))
    # eta is the dynamic loading time-scale (tau = eta * t); it is applied in
    # the imposed thermal load only, NOT as a mass / kinetic-energy multiplier.
    c1_c      = fem.Constant(domain, PETSc.ScalarType(model_parameters.get("c1", 0.0)))
    c2_c      = fem.Constant(domain, PETSc.ScalarType(model_parameters.get("c2", 0.0)))
    c3_c      = fem.Constant(domain, PETSc.ScalarType(model_parameters.get("c3", 0.0)))
    delta_t_c = fem.Constant(domain, PETSc.ScalarType(1.0 / n_dyn))
    theta_c   = fem.Constant(domain, PETSc.ScalarType(0.0))

    # Lame parameters (only used for 2D elasticity, but define them here for convenience since they appear in multiple places in the variational forms):
    mu_c    = E_c / (2.0 * (1.0 + nu_c))
    lmbda_c = E_c * nu_c / ((1.0 + nu_c) * (1.0 - 2.0 * nu_c))

    g = g_degradation(alpha)
    eps_e = elastic_strain(u, theta_c, physics)

    if physics == "1D":
        eps_v = strain(v, "1D")     # strain rate
        strain_e_density         = 0.5 * g * E_c * eps_e ** 2
        foundation_e_density     = 0.5 * Lambda_c ** 2 * u ** 2
        kinetic_e_density        = 0.5 * v ** 2
        # c1: local velocity damping, c2: Kelvin-Voigt strain-rate damping
        # c3 damps the damage rate and enters the damage sub-problem only (not here)
        dissipated_power_density = 0.5 * (c1_c * v ** 2 + c2_c * E_c * eps_v ** 2)
        mean_stress_expr         = g * E_c * eps_e + c2_c * E_c * eps_v
    else:
        eps_v = strain(v, "2D")
        
        # Note: the elastic energy density is g*psi_0 where psi_0 is the *undamaged* elastic energy density.  This means that the viscous part of the stress (which depends on eps_v) also gets degraded by g -- in other words, we are assuming that damage reduces not only the elastic stiffness but also the viscous damping.  This is a modeling choice; if you want to assume that damage only reduces the elastic stiffness but leaves the viscous damping unaffected, then you can simply remove the factor of g from the definition of sigma_vis below.
        elastic_energy_term      = 0.5 * lmbda_c * ufl.tr(eps_e)**2 + mu_c * ufl.inner(eps_e, eps_e)
        strain_e_density         = g * elastic_energy_term
        foundation_e_density     = 0.5 * Lambda_c ** 2 * ufl.dot(u, u)
        kinetic_e_density        = 0.5 * ufl.dot(v, v)

        # The viscous energy term is defined such that its time derivative gives the dissipated power density.  In particular, the contribution from the c2 term is chosen so that when you take the time derivative, you get a term of the form c2 * (lmbda_c * tr(eps) * tr(eps_v) + 2 * mu_c * inner(eps, eps_v)), which matches the form of sigma_vis below and ensures that the viscous stress contributes correctly to the dissipated power.
        viscous_energy_term      = 0.5 * lmbda_c * ufl.tr(eps_v)**2 + mu_c * ufl.inner(eps_v, eps_v)
        # c1: local velocity damping, c2: Kelvin-Voigt strain-rate damping
        # c3 damps the damage rate and enters the damage sub-problem only (not here)
        dissipated_power_density = (0.5 * c1_c * ufl.dot(v, v) + c2_c * viscous_energy_term)

        # Total stress is elastic + viscous (c2 only); c3 acts on the damage field
        # and is not reported as a Cauchy stress.
        # The reported mean stress is the xx component of the total stress.
        sigma_el  = g * (lmbda_c * ufl.tr(eps_e) * ufl.Identity(domain.geometry.dim) + 2.0 * mu_c * eps_e)
        sigma_vis = c2_c * (lmbda_c * ufl.tr(eps_v) * ufl.Identity(domain.geometry.dim) + 2.0 * mu_c * eps_v)
        sigma     = sigma_el + sigma_vis
        mean_stress_expr         = sigma[0, 0]

    fracture_e_density = fracture_energy_density(alpha, grad_sq(alpha, physics),
                                                 l_hat_c, model_name)

    strain_energy     = strain_e_density     * dx
    foundation_energy = foundation_e_density * dx
    potential_energy  = strain_energy + foundation_energy
    fracture_energy   = fracture_e_density   * dx
    kinetic_energy    = kinetic_e_density    * dx
    dissipated_power  = dissipated_power_density * dx

    strain_energy_form     = fem.form(strain_energy)
    foundation_energy_form = fem.form(foundation_energy)
    fracture_energy_form   = fem.form(fracture_energy)
    kinetic_energy_form    = fem.form(kinetic_energy)
    dissipated_power_form  = fem.form(dissipated_power)
    mean_stress_form       = fem.form(mean_stress_expr * dx)
    error_L2_alpha_form    = fem.form((alpha - alpha_old_iter) ** 2 * dx)

    # c3 damage-rate dissipation tracked separately (alpha_rate = (alpha-alpha_lb)/dt)
    alpha_rate = fem.Function(V_alpha)
    dissipated_power_alpha_form = fem.form(0.5 * c3_c * alpha_rate * alpha_rate * dx)

    u_test     = ufl.TestFunction(V_u)
    alpha_test = ufl.TestFunction(V_alpha)

    Res_u_qs     = ufl.derivative(potential_energy, u, u_test)
    Res_alpha_qs = ufl.derivative(potential_energy + fracture_energy, alpha, alpha_test)
    if physics == "1D":
        inertia_term = a_new * u_test * dx
    else:
        inertia_term = ufl.dot(a_new, u_test) * dx
    Q_dv    = ufl.derivative(dissipated_power, v, u_test)
    Res_acc = inertia_term + Q_dv + ufl.derivative(potential_energy, u, u_test)

    Theta_qs_fn, Theta_dyn_fn = _make_thermal_loaders(
        loading_parameters, float(model_parameters["eta"]))

    # -------------------------------------------------------------------------
    # SNES setup (QS)
    # -------------------------------------------------------------------------
    J_u_qs     = ufl.derivative(Res_u_qs,     u,     ufl.TrialFunction(V_u))
    J_alpha_qs = ufl.derivative(Res_alpha_qs, alpha, ufl.TrialFunction(V_alpha))
    elastic_problem_qs = SNESProblem(Res_u_qs,     u,     bcs_u,     J=J_u_qs)
    damage_problem_qs  = SNESProblem(Res_alpha_qs, alpha, bcs_alpha, J=J_alpha_qs)
    solver_u_qs     = make_linear_snes(elastic_problem_qs, V_u)
    solver_alpha_qs = make_damage_snes(damage_problem_qs,  alpha_lb, alpha_ub)

    # -------------------------------------------------------------------------
    # QS loop -- runs t = 0 -> t_final_qs (load reaches theta_max)
    # -------------------------------------------------------------------------
    u.x.array[:] = 0.0; v.x.array[:] = 0.0; a.x.array[:] = 0.0
    alpha.x.array[:] = 0.0
    alpha_lb.x.array[:] = 0.0
    alpha_ub.x.array[:] = 1.0
    theta_c.value = 0.0

    qs = {"t": [], "theta": [], "sigma_bar": [], "P_el": [], "P_f": [], "S": [], "total": [], "n_cracks": []}
    qs_snapshots = []
    paraview_alpha_qs = []
    paraview_u_qs     = []
    dt_qs = t_final_qs / n_qs
    n_qs_max = n_qs            # QS runs exactly n_qs steps (t = 0 -> t_final_qs)
    snap_every_qs = max(1, n_qs_max // max(1, N_snap))
    pbar_qs = tqdm(total=n_qs_max,
                   desc=f"QS  [{physics}|{model_name}]",
                   dynamic_ncols=True,
                   disable=not (verbose and comm.rank == 0))

    t_cur = 0.0
    i = 0
    while t_cur + 1e-12 < t_final_qs:
        i += 1
        t_cur += dt_qs
        theta_c.value = float(Theta_qs_fn(t_cur))
        n_alt = alt_min_loop(solver_u_qs, u.x.petsc_vec,
                             solver_alpha_qs, alpha,
                             alpha_old_iter, error_L2_alpha_form,
                             AltMin_parameters["max_iter"], AltMin_parameters["tol"],
                             comm_=domain.comm)
        alpha_lb.x.array[:] = alpha.x.array

        qs["t"].append(float(t_cur))
        qs["theta"].append(float(theta_c.value))
        qs["sigma_bar"].append(domain.comm.allreduce(fem.assemble_scalar(mean_stress_form),    op=MPI.SUM))
        qs["P_el"].append(domain.comm.allreduce(fem.assemble_scalar(strain_energy_form),       op=MPI.SUM))
        qs["P_f"].append(domain.comm.allreduce(fem.assemble_scalar(foundation_energy_form),    op=MPI.SUM))
        qs["S"].append(domain.comm.allreduce(fem.assemble_scalar(fracture_energy_form),        op=MPI.SUM))
        qs["total"].append(qs["P_el"][-1] + qs["P_f"][-1] + qs["S"][-1])
        qs["n_cracks"].append(count_cracks(alpha.x.array))
        if field_recorder is not None:
            field_recorder("qs", float(theta_c.value),
                           u.x.array.copy(), alpha.x.array.copy())

        a_max = _alpha_max(alpha, domain.comm)
        pbar_qs.update(1)
        pbar_qs.set_postfix(t=f"{t_cur:.3f}", theta=f"{float(theta_c.value):.3f}",
                            a_max=f"{a_max:.3f}", n_cr=qs["n_cracks"][-1], altmin=n_alt)

        if i % snap_every_qs == 0 or i == n_qs_max:
            qs_snapshots.append({"step": i, "t": float(t_cur),
                                 "theta": float(theta_c.value),
                                 "alpha": alpha.x.array.copy()})
            if paraview and physics == "2D":
                alpha_snap = fem.Function(V_alpha, name="Damage")
                alpha_snap.x.array[:] = alpha.x.array
                paraview_alpha_qs.append((t_cur, alpha_snap))
                u_snap = fem.Function(V_u, name="Displacement")
                u_snap.x.array[:] = u.x.array
                paraview_u_qs.append((t_cur, u_snap))
    pbar_qs.close()

    alpha_qs_final = alpha.x.array.copy()
    for k in qs:
        qs[k] = np.asarray(qs[k])
    qs_events = detect_crack_events(qs["theta"], qs["S"], qs["n_cracks"])

    # -------------------------------------------------------------------------
    # SNES setup (Dyn)
    # -------------------------------------------------------------------------
    beta_v  = Newmark_parameters["beta"]
    gamma_v = Newmark_parameters["gamma"]

    def u_newmark(u_, v_, a_, a_new_, dt):
        return u_ + dt * v_ + 0.5 * dt ** 2 * ((1.0 - 2.0 * beta_v) * a_ + 2.0 * beta_v * a_new_)

    def v_newmark(v_, a_, a_new_, dt):
        return v_ + dt * ((1.0 - gamma_v) * a_ + gamma_v * a_new_)

    Res_acc_newmark = ufl.replace(Res_acc, {
        u: u_newmark(u, v, a, a_new, delta_t_c),
        v: v_newmark(v,    a, a_new, delta_t_c),
    })
    # c3 damage-rate term: D_alpha Q[alpha_hat] = int c3 * alpha_dot * alpha_hat dx
    # with alpha_dot ~ (alpha - alpha_lb) / dt
    Res_alpha_dyn   = (ufl.replace(Res_alpha_qs, {u: u_newmark(u, v, a, a_new, delta_t_c)})
                       + c3_c / delta_t_c * (alpha - alpha_lb) * alpha_test * dx)

    J_acc_newmark = ufl.derivative(Res_acc_newmark, a_new, ufl.TrialFunction(V_u))
    J_alpha_dyn   = ufl.derivative(Res_alpha_dyn,   alpha, ufl.TrialFunction(V_alpha))

    acc_problem        = SNESProblem(Res_acc_newmark, a_new, bcs_a,     J=J_acc_newmark)
    damage_problem_dyn = SNESProblem(Res_alpha_dyn,   alpha, bcs_alpha, J=J_alpha_dyn)
    solver_acc       = make_linear_snes(acc_problem,        V_u)
    solver_alpha_dyn = make_damage_snes(damage_problem_dyn, alpha_lb, alpha_ub)

    # -------------------------------------------------------------------------
    # Dynamic loop -- while max(alpha) < alpha_break and t < t_final_dyn
    # -------------------------------------------------------------------------
    for fn in (u, u_new, v, v_new, a, a_new):
        fn.x.array[:] = 0.0
    alpha.x.array[:] = 0.0
    alpha_lb.x.array[:] = 0.0
    alpha_ub.x.array[:] = 1.0
    theta_c.value = 0.0

    dyn = {"t": [], "theta": [], "sigma_bar": [], "K": [], "P_el": [], "P_f": [], "S": [],
           "D": [], "total": [], "n_cracks": []}
    dyn_snapshots = []
    paraview_alpha = []
    paraview_u     = []
    # Time-scaling: the dynamic load uses tau = eta*t, so t_final_dyn is
    # tau_final_dyn/eta (tau_final_dyn=1 by default; see _final_times()).
    # We keep the requested number of steps (N_steps_dyn) and stretch the
    # step size accordingly.
    eta_v          = float(model_parameters["eta"])
    t_final_dyn    = tau_final_dyn / eta_v
    dt             = t_final_dyn / n_dyn
    delta_t_c.value = dt
    n_dyn_max      = n_dyn
    snap_every_dyn = max(1, n_dyn_max // max(1, N_snap))

    diss_energy = 0.0

    pbar_dyn = tqdm(total=n_dyn_max,
                    desc=f"Dyn [{physics}|{model_name}]",
                    dynamic_ncols=True,
                    disable=not (verbose and comm.rank == 0))

    t_cur = 0.0
    step = 0
    while t_cur + 1e-12 < t_final_dyn:
        step += 1
        t_cur += dt
        theta_c.value = float(Theta_dyn_fn(t_cur))

        n_alt = alt_min_loop(solver_acc, a_new.x.petsc_vec,
                             solver_alpha_dyn, alpha,
                             alpha_old_iter, error_L2_alpha_form,
                             AltMin_parameters["max_iter"], AltMin_parameters["tol"],
                             comm_=domain.comm)

        u_new.x.array[:] = (
            u.x.array
            + dt * v.x.array
            + 0.5 * dt ** 2 * ((1.0 - 2.0 * beta_v) * a.x.array + 2.0 * beta_v * a_new.x.array)
        )
        v_new.x.array[:] = (
            v.x.array
            + dt * ((1.0 - gamma_v) * a.x.array + gamma_v * a_new.x.array)
        )
        u_new.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        v_new.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        u.x.array[:] = u_new.x.array
        v.x.array[:] = v_new.x.array
        a.x.array[:] = a_new.x.array
        alpha_rate.x.array[:] = (alpha.x.array - alpha_lb.x.array) / dt
        alpha_lb.x.array[:] = alpha.x.array

        dyn["t"].append(t_cur)
        dyn["theta"].append(float(theta_c.value))
        dyn["sigma_bar"].append(domain.comm.allreduce(fem.assemble_scalar(mean_stress_form), op=MPI.SUM))
        K_now = domain.comm.allreduce(fem.assemble_scalar(kinetic_energy_form), op=MPI.SUM)
        dyn["K"].append(K_now)
        dyn["P_el"].append(domain.comm.allreduce(fem.assemble_scalar(strain_energy_form),     op=MPI.SUM))
        dyn["P_f"].append(domain.comm.allreduce(fem.assemble_scalar(foundation_energy_form), op=MPI.SUM))
        dyn["S"].append(domain.comm.allreduce(fem.assemble_scalar(fracture_energy_form),     op=MPI.SUM))
        # Cumulative dissipated energy.  The assembled forms are the Rayleigh
        # POTENTIAL Q (quadratic in the rates: c1+c2 on u, c3 on alpha).  The power
        # actually removed is D_ydot Q . ydot = 2 Q by Euler's theorem for a
        # degree-2 homogeneous function, so the accumulator carries a factor 2.
        # The residual uses D_ydot Q directly and is unaffected -- do NOT move this
        # factor into the forms above or the damping forces would double.
        diss_power_now = (domain.comm.allreduce(fem.assemble_scalar(dissipated_power_form),       op=MPI.SUM)
                        + domain.comm.allreduce(fem.assemble_scalar(dissipated_power_alpha_form), op=MPI.SUM))
        diss_energy += 2.0 * dt * float(diss_power_now)
        dyn["D"].append(diss_energy)
        # Stored dynamic total (K + P_el + P_f + S).  Dissipated energy D is a
        # separate curve, not folded into the stored total.
        dyn["total"].append(K_now + dyn["P_el"][-1] + dyn["P_f"][-1] + dyn["S"][-1])
        dyn["n_cracks"].append(count_cracks(alpha.x.array))
        if field_recorder is not None:
            field_recorder("dyn", float(theta_c.value),
                           u.x.array.copy(), alpha.x.array.copy())

        a_max = _alpha_max(alpha, domain.comm)

        pbar_dyn.update(1)
        pbar_dyn.set_postfix(t=f"{t_cur:.3f}",
                             theta=f"{float(theta_c.value):.3f}",
                             K=f"{K_now:.2e}",
                             a_max=f"{a_max:.3f}",
                             n_cr=dyn["n_cracks"][-1],
                             altmin=n_alt)

        if step % snap_every_dyn == 0 or step == n_dyn_max:
            dyn_snapshots.append({"step": step, "t": float(t_cur),
                                  "theta": float(theta_c.value),
                                  "alpha": alpha.x.array.copy()})
            if paraview:
                alpha_snap = fem.Function(V_alpha, name="Damage")
                alpha_snap.x.array[:] = alpha.x.array
                paraview_alpha.append((t_cur, alpha_snap))
                u_snap = fem.Function(V_u, name="Displacement")
                u_snap.x.array[:] = u.x.array
                paraview_u.append((t_cur, u_snap))
    pbar_dyn.close()

    alpha_dyn_final = alpha.x.array.copy()
    for k in dyn:
        dyn[k] = np.asarray(dyn[k])
    dyn_events = detect_crack_events(dyn["theta"], dyn["S"], dyn["n_cracks"])

    x_alpha = V_alpha.tabulate_dof_coordinates()[:, 0]

    for s in (solver_u_qs, solver_alpha_qs, solver_acc, solver_alpha_dyn):
        s.destroy()

    result = {
        "physics":          physics,
        "model":            model_name,
        "qs":               qs,
        "dyn":              dyn,
        "x_alpha":          x_alpha,
        "alpha_qs_final":   alpha_qs_final,
        "alpha_dyn_final":  alpha_dyn_final,
        "qs_snapshots":     qs_snapshots,
        "dyn_snapshots":    dyn_snapshots,
        "qs_events":        qs_events,
        "dyn_events":       dyn_events,
        "dissipated_total": diss_energy,
        "triang":           triangulation_from_domain(domain) if physics == "2D" else None,
    }

    if output_dir is None:
        output_dir = ROOT / "output"


    if plot and comm.rank == 0:
        png, pdf = plot_thermal_run(result, model_parameters, mesh_parameters,
                                    loading_parameters, solver_parameters, output_dir)
        if verbose:
            print(f"  saved {png}")
            print(f"        {pdf}")

    if paraview and physics == "2D" and comm.rank == 0:
        xdmf_qs = export_paraview(domain, paraview_alpha_qs, paraview_u_qs,
                                  "thermal",
                                  model_parameters, mesh_parameters,
                                  loading_parameters, solver_parameters,
                                  output_dir, tag="QS")
        xdmf_dyn = export_paraview(domain, paraview_alpha, paraview_u,
                                   "thermal",
                                   model_parameters, mesh_parameters,
                                   loading_parameters, solver_parameters,
                                   output_dir, tag="dyn")
        if verbose:
            for xdmf in (xdmf_qs, xdmf_dyn):
                if xdmf:
                    print(f"        {xdmf}")

    return result


# =============================================================================
# Stand-alone execution
# =============================================================================
if __name__ == "__main__":
    # model / physics / mesh_per_lhat come straight from tools/parameters.py
    # (DEFAULT_SOLVER_PARAMETERS, DEFAULT_MESH_PARAMETERS) -- edit *there* to
    # switch them, rather than overriding here.
    run_problem(**get_defaults("thermal"))
