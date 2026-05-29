"""
problems/dynamic.py
===================
**Mechanical / dynamic** phase-field fragmentation problem.

Geometry & loading
------------------
* Bar :math:`[0, L_x]` (1D) or *unstructured* triangulation of
  :math:`[0, L_x]\\times[0, L_y]` (2D, Gmsh-generated).  Mesh resolution is
  controlled by a single knob ``mesh_per_lhat`` (number of cells per
  regularisation length).
* Left end clamped:  ``u = 0``  (in 2D both components).
* Right end pulled:  ``u_x = U(t)``.  Two histories are used:
  - quasi-static branch: linear ramp ``U_qs(t) = U_max * t`` (so ``U = t``
    when ``U_max = 1``), reaching ``U_max`` at ``t = 1``;
  - dynamic branch: smoothed linear ramp with ``tau = eta * t``,
    ``U_dyn(t) = U_max * (tau/2) * (1 + tanh(tau/T0))``.  The imposed
    velocity ``V`` and acceleration ``A`` are obtained by exact symbolic
    differentiation w.r.t. ``t``, so ``V`` carries one factor of ``eta``
    and ``A`` carries ``eta**2``.  ``eta`` is now *only* the dynamic
    loading time-scale (``tau = eta t``); it no longer multiplies the mass
    term.  To reach ``U_max`` the dynamic run must extend to
    ``t_final_dyn = 1 / eta`` (since the load reaches ``U_max`` at ``tau=1``).
* Damage Dirichlet:  ``alpha = 0`` on both end-edges (no crack can nucleate
  at the loading device).
* Elastic foundation everywhere:  energy density
  ``0.5 * Lambda^2 * |u|^2``.

Termination
-----------
Both loops are purely time-driven (a fixed number of steps): the QS loop
runs ``t = 0 -> 1`` (``N_steps_qs`` steps, load reaches ``U_max`` at ``t=1``)
and the Dyn loop runs ``t = 0 -> 1/eta`` (``N_steps_dyn`` steps, load reaches
``U_max`` at ``tau=eta*t=1``).  Crack-nucleation *generations* are detected
afterwards (surface-energy jumps, labelled by the connected damaged-region
count) and marked on the energy plot; they do not stop the run.

Switches
--------
``solver_parameters["model"]`` -- ``"AT1"`` or ``"AT2"``.
``mesh_parameters["physics"]`` -- ``"1D"`` or ``"2D"``.
(The geometry is fixed per problem file via ``_PROBLEM_SHAPE`` -- not a switch.)
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.imports import (
    np, sp, ufl, fem, df_mesh, MPI, PETSc, comm,
    Path, tqdm,
)
from tools.helpers   import (
    SNESProblem, make_linear_snes, make_damage_snes, alt_min_loop, print_mesh_info,
    make_crack_counter, detect_crack_events,
)
from tools.meshing   import create_mesh_and_tags, grad_sq, strain
from tools.solvers   import fracture_energy_density, g_degradation
from tools.parameters import get_defaults
from tools.plotting  import (
    plot_mechanical_run, export_paraview, triangulation_from_domain,
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
def _make_displacement_loaders(loading_parameters, eta):
    """Return the quasi-static and dynamic boundary-displacement loaders.

    Returns ``(U_qs, (U_dyn, V_dyn, A_dyn))`` as numpy callables of ``t``:

    * ``U_qs(t)   = U_max * t``                              (linear ramp)
    * ``U_dyn(t)  = U_max * (tau/2) * (1 + tanh(tau/T0))``,  ``tau = eta*t``
    * ``V_dyn``, ``A_dyn`` are the exact symbolic ``t``-derivatives of
      ``U_dyn`` (so they carry the corresponding powers of ``eta``).
    """
    t_sp   = sp.Symbol("t", real=True)
    T0_v   = loading_parameters["T0"]
    Umax_v = loading_parameters["U_max"]

    U_qs_sp  = Umax_v * t_sp

    tau      = eta * t_sp
    U_dyn_sp = Umax_v * (tau / 2.0) * (1.0 + sp.tanh(tau / T0_v))
    V_dyn_sp = sp.diff(U_dyn_sp, t_sp)
    A_dyn_sp = sp.diff(V_dyn_sp, t_sp)

    return (sp.lambdify(t_sp, U_qs_sp, "numpy"),
            (sp.lambdify(t_sp, U_dyn_sp, "numpy"),
             sp.lambdify(t_sp, V_dyn_sp, "numpy"),
             sp.lambdify(t_sp, A_dyn_sp, "numpy")))


def _setup_function_spaces(domain, physics):
    V_alpha = fem.functionspace(domain, ("Lagrange", 1))
    if physics == "1D":
        V_u = fem.functionspace(domain, ("Lagrange", 1))
    else:
        V_u = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    return V_u, V_alpha


def _alpha_max(alpha, comm_):
    """Global maximum of the damage field across MPI ranks."""
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
) -> dict:
    """Run one *quasi-static + dynamic* mechanical simulation."""
    # Geometry is a property of the *problem*, not a user-tunable knob.
    mesh_parameters["shape"] = _PROBLEM_SHAPE

    physics    = mesh_parameters["physics"]
    model_name = solver_parameters["model"]
    n_qs       = loading_parameters["N_steps_qs"]
    n_dyn      = loading_parameters["N_steps_dyn"]
    # Canonical final pseudo-time: the QS linear ramp U_qs(t)=U_max*t reaches
    # U_max at t=1, and the dynamic ramp reaches U_max at tau=eta*t=1 (t=1/eta).
    t_final_qs  = 1.0
    N_snap      = loading_parameters.get("N_snapshots", 6)

    if verbose and comm.rank == 0:
        print(
            f"\n[DYNAMIC | {physics} | {model_name}]  "
            f"l_hat={model_parameters['l_hat']}, "
            f"Lambda={model_parameters['Lambda']}, "
            f"eta={model_parameters['eta']}  |  "
            f"mesh_per_lhat={mesh_parameters.get('mesh_per_lhat', 4)}  |  "
            f"n_QS={n_qs}, n_Dyn={n_dyn} (t_dyn_final={t_final_qs/float(model_parameters['eta']):.3g})"
        )

    # -------------------------------------------------------------------------
    # Mesh / tags / spaces
    # -------------------------------------------------------------------------
    domain, mt, dx, ds = create_mesh_and_tags(mesh_parameters, model_parameters, comm)
    V_u, V_alpha = _setup_function_spaces(domain, physics)
    gdim = domain.geometry.dim
    fdim = domain.topology.dim - 1
    count_cracks = make_crack_counter(domain, V_alpha, thr=0.5)
    if verbose:
        print_mesh_info(domain, V_u, V_alpha,
                        label=f"MECH {physics}", comm_=domain.comm)

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

    # -------------------------------------------------------------------------
    # Boundary conditions
    # -------------------------------------------------------------------------
    left_facets  = mt.find(1)
    right_facets = mt.find(2)

    if physics == "1D":
        u_left_val  = fem.Constant(domain, PETSc.ScalarType(0.0))
        u_right_val = fem.Constant(domain, PETSc.ScalarType(0.0))
        v_left_val  = fem.Constant(domain, PETSc.ScalarType(0.0))
        v_right_val = fem.Constant(domain, PETSc.ScalarType(0.0))
        a_left_val  = fem.Constant(domain, PETSc.ScalarType(0.0))
        a_right_val = fem.Constant(domain, PETSc.ScalarType(0.0))

        left_dofs_u  = fem.locate_dofs_topological(V_u, fdim, left_facets)
        right_dofs_u = fem.locate_dofs_topological(V_u, fdim, right_facets)

        bcs_u = [fem.dirichletbc(u_left_val,  left_dofs_u,  V_u),
                 fem.dirichletbc(u_right_val, right_dofs_u, V_u)]
        bcs_v = [fem.dirichletbc(v_left_val,  left_dofs_u,  V_u),
                 fem.dirichletbc(v_right_val, right_dofs_u, V_u)]
        bcs_a = [fem.dirichletbc(a_left_val,  left_dofs_u,  V_u),
                 fem.dirichletbc(a_right_val, right_dofs_u, V_u)]

    else:  # 2D
        zero_vec = fem.Function(V_u)
        zero_vec.x.array[:] = 0.0
        left_dofs_full = fem.locate_dofs_topological(V_u, fdim, left_facets)
        bc_left_clamp = fem.dirichletbc(zero_vec, left_dofs_full)

        u_right_val = fem.Constant(domain, PETSc.ScalarType(0.0))
        v_right_val = fem.Constant(domain, PETSc.ScalarType(0.0))
        a_right_val = fem.Constant(domain, PETSc.ScalarType(0.0))

        V_ux_sub = V_u.sub(0)
        right_dofs_ux = fem.locate_dofs_topological(V_ux_sub, fdim, right_facets)
        bc_u_right = fem.dirichletbc(u_right_val, right_dofs_ux, V_ux_sub)
        bc_v_right = fem.dirichletbc(v_right_val, right_dofs_ux, V_ux_sub)
        bc_a_right = fem.dirichletbc(a_right_val, right_dofs_ux, V_ux_sub)

        bcs_u = [bc_left_clamp, bc_u_right]
        bcs_v = [bc_left_clamp, bc_v_right]
        bcs_a = [bc_left_clamp, bc_a_right]

    left_dofs_alpha  = fem.locate_dofs_topological(V_alpha, fdim, left_facets)
    right_dofs_alpha = fem.locate_dofs_topological(V_alpha, fdim, right_facets)
    bcs_alpha = [
        fem.dirichletbc(PETSc.ScalarType(0.0), left_dofs_alpha,  V_alpha),
        fem.dirichletbc(PETSc.ScalarType(0.0), right_dofs_alpha, V_alpha),
    ]

    # -------------------------------------------------------------------------
    # Material + energies
    # -------------------------------------------------------------------------
    E_c       = fem.Constant(domain, PETSc.ScalarType(model_parameters.get("E_ref", 1.0)))
    nu_c      = fem.Constant(domain, PETSc.ScalarType(model_parameters.get("nu", 0.3)))
    Lambda_c  = fem.Constant(domain, PETSc.ScalarType(model_parameters["Lambda"]))
    l_hat_c   = fem.Constant(domain, PETSc.ScalarType(model_parameters["l_hat"]))
    # eta is the dynamic loading time-scale (tau = eta * t); it is applied in
    # the imposed boundary load only, NOT as a mass / kinetic-energy multiplier.
    c1_c      = fem.Constant(domain, PETSc.ScalarType(model_parameters.get("c1", 0.0)))
    c2_c      = fem.Constant(domain, PETSc.ScalarType(model_parameters.get("c2", 0.0)))
    c3_c      = fem.Constant(domain, PETSc.ScalarType(model_parameters.get("c3", 0.0)))
    delta_t_c = fem.Constant(domain, PETSc.ScalarType(1.0 / n_dyn))

    # Lame parameters (only used for 2D elasticity, but define them here for convenience since they appear in multiple places in the variational forms):
    mu_c    = E_c / (2.0 * (1.0 + nu_c))
    lmbda_c = E_c * nu_c / ((1.0 + nu_c) * (1.0 - 2.0 * nu_c))

    g = g_degradation(alpha)

    if physics == "1D":
        eps     = strain(u, "1D")
        eps_v   = strain(v, "1D")           # strain-rate
        strain_e_density         = 0.5 * g * E_c * eps ** 2
        foundation_e_density     = 0.5 * Lambda_c ** 2 * u ** 2
        kinetic_e_density        = 0.5 * v ** 2
        # c1: local velocity, c2: strain-rate, c3: full velocity-gradient filter
        dissipated_power_density = 0.5 * (c1_c * v ** 2
                                          + c2_c * E_c * eps_v ** 2
                                          + c3_c * ufl.inner(ufl.grad(v), ufl.grad(v)))
        # c3 is a weak-form filter only -- it is NOT reported as a Cauchy stress
        reaction_form_expr       = g * E_c * eps + c2_c * E_c * eps_v
    else:
        eps     = strain(u, "2D")
        eps_v   = strain(v, "2D")
        
        # Note: the elastic energy density is g*psi_0 where psi_0 is the *undamaged* elastic energy density.  This means that the viscous part of the stress (which depends on eps_v) also gets degraded by g -- in other words, we are assuming that damage reduces not only the elastic stiffness but also the viscous damping.  This is a modeling choice; if you want to assume that damage only reduces the elastic stiffness but leaves the viscous damping unaffected, then you can simply remove the factor of g from the definition of sigma_vis below.
        elastic_energy_term      = 0.5 * lmbda_c * ufl.tr(eps)**2 + mu_c * ufl.inner(eps, eps)
        strain_e_density         = g * elastic_energy_term
        foundation_e_density     = 0.5 * Lambda_c ** 2 * ufl.dot(u, u)
        kinetic_e_density        = 0.5 * ufl.dot(v, v)

        # The viscous energy term is defined such that its time derivative gives the dissipated power density.  In particular, the contribution from the c2 term is chosen so that when you take the time derivative, you get a term of the form c2 * (lmbda_c * tr(eps) * tr(eps_v) + 2 * mu_c * inner(eps, eps_v)), which matches the form of sigma_vis below and ensures that the viscous stress contributes correctly to the dissipated power.
        viscous_energy_term      = 0.5 * lmbda_c * ufl.tr(eps_v)**2 + mu_c * ufl.inner(eps_v, eps_v)
        # c1: local velocity, c2: strain-rate, c3: full velocity-gradient high-order filter
        dissipated_power_density = (0.5 * c1_c * ufl.dot(v, v)
                                    + c2_c * viscous_energy_term
                                    + 0.5 * c3_c * ufl.inner(ufl.grad(v), ufl.grad(v)))

        # Total stress is elastic + viscous (c2 only).  The c3 gradient term is a
        # weak-form filter and is deliberately NOT included in the Cauchy stress.
        # The reaction force on the right edge is the normal component of the total
        # stress, which here reduces to the xx component since the normal is (1,0).
        sigma_el  = g * (lmbda_c * ufl.tr(eps) * ufl.Identity(domain.geometry.dim) + 2.0 * mu_c * eps)
        sigma_vis = c2_c * (lmbda_c * ufl.tr(eps_v) * ufl.Identity(domain.geometry.dim) + 2.0 * mu_c * eps_v)
        sigma     = sigma_el + sigma_vis
        reaction_form_expr   = sigma[0, 0]

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
    reaction_right_form    = fem.form(reaction_form_expr * ds(2))
    error_L2_alpha_form    = fem.form((alpha - alpha_old_iter) ** 2 * dx)

    u_test     = ufl.TestFunction(V_u)
    alpha_test = ufl.TestFunction(V_alpha)

    Res_u_qs     = ufl.derivative(potential_energy, u, u_test)
    Res_alpha_qs = ufl.derivative(potential_energy + fracture_energy, alpha, alpha_test)
    if physics == "1D":
        inertia_term = a_new * u_test * dx
    else:
        inertia_term = ufl.dot(a_new, u_test) * dx
    # Viscous term  Q_dv = d(dissipated_power)/dv . u_test
    Q_dv    = ufl.derivative(dissipated_power, v, u_test)
    Res_acc = inertia_term + Q_dv + ufl.derivative(potential_energy, u, u_test)

    U_qs_fn, (U_dyn_fn, V_dyn_fn, A_dyn_fn) = _make_displacement_loaders(
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
    # Quasi-static loop -- runs t = 0 -> t_final_qs (load reaches U_max)
    # -------------------------------------------------------------------------
    u.x.array[:] = 0.0; v.x.array[:] = 0.0; a.x.array[:] = 0.0
    alpha.x.array[:] = 0.0
    alpha_lb.x.array[:] = 0.0
    alpha_ub.x.array[:] = 1.0

    qs = {"t": [], "U": [], "F": [], "P_el": [], "P_f": [], "S": [], "total": [], "n_cracks": []}
    paraview_alpha_qs = []
    paraview_u_qs     = []
    dt_qs = 1.0 / n_qs
    n_qs_max = n_qs            # QS runs exactly n_qs steps (t = 0 -> t_final_qs=1)
    snap_every_qs = max(1, n_qs_max // max(1, N_snap))
    step_qs = 0
    pbar_qs = tqdm(total=n_qs_max,
                   desc=f"QS  [{physics}|{model_name}]",
                   dynamic_ncols=True,
                   disable=not (verbose and comm.rank == 0))

    t_cur = 0.0
    while t_cur + 1e-12 < t_final_qs:
        t_cur += dt_qs
        u_right_val.value = float(U_qs_fn(t_cur))
        n_alt = alt_min_loop(solver_u_qs, u.x.petsc_vec,
                             solver_alpha_qs, alpha,
                             alpha_old_iter, error_L2_alpha_form,
                             AltMin_parameters["max_iter"], AltMin_parameters["tol"],
                             comm_=domain.comm)
        alpha_lb.x.array[:] = alpha.x.array

        qs["t"].append(float(t_cur))
        qs["U"].append(float(u_right_val.value))
        qs["F"].append(domain.comm.allreduce(fem.assemble_scalar(reaction_right_form),  op=MPI.SUM))
        qs["P_el"].append(domain.comm.allreduce(fem.assemble_scalar(strain_energy_form),     op=MPI.SUM))
        qs["P_f"].append(domain.comm.allreduce(fem.assemble_scalar(foundation_energy_form), op=MPI.SUM))
        qs["S"].append(domain.comm.allreduce(fem.assemble_scalar(fracture_energy_form),     op=MPI.SUM))
        qs["total"].append(qs["P_el"][-1] + qs["P_f"][-1] + qs["S"][-1])
        qs["n_cracks"].append(count_cracks(alpha.x.array))

        a_max = _alpha_max(alpha, domain.comm)
        pbar_qs.update(1)
        pbar_qs.set_postfix(t=f"{t_cur:.3f}", U=f"{float(u_right_val.value):.3f}",
                            a_max=f"{a_max:.3f}", altmin=n_alt)

        step_qs += 1
        if paraview and physics == "2D" and (step_qs % snap_every_qs == 0 or step_qs == n_qs_max):
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
    qs_events = detect_crack_events(qs["U"], qs["S"], qs["n_cracks"])

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
    Res_alpha_dyn   = ufl.replace(Res_alpha_qs, {u: u_newmark(u, v, a, a_new, delta_t_c)})

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

    dyn = {"t": [], "U": [], "F": [], "K": [], "P_el": [], "P_f": [], "S": [],
           "D": [], "total": [], "n_cracks": []}
    paraview_alpha = []
    paraview_u     = []

    # Time-scaling: the dynamic load uses tau = eta*t, so it reaches U_max at
    # tau = 1, i.e. t = t_final_qs / eta.  We keep the requested number of steps
    # (N_steps_dyn) and stretch the step size accordingly.
    eta_v       = float(model_parameters["eta"])
    t_final_dyn = t_final_qs / eta_v
    dt          = t_final_dyn / n_dyn
    delta_t_c.value = dt
    n_dyn_max   = n_dyn
    snap_every  = max(1, n_dyn_max // max(1, N_snap))

    diss_energy = 0.0

    pbar_dyn = tqdm(total=n_dyn_max,
                    desc=f"Dyn [{physics}|{model_name}]",
                    dynamic_ncols=True,
                    disable=not (verbose and comm.rank == 0))

    t_cur = 0.0
    step  = 0
    while t_cur + 1e-12 < t_final_dyn:
        step += 1
        t_cur += dt
        u_right_val.value = float(U_dyn_fn(t_cur))
        v_right_val.value = float(V_dyn_fn(t_cur))
        a_right_val.value = float(A_dyn_fn(t_cur))

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
        fem.set_bc(u_new.x.petsc_vec, bcs_u)
        fem.set_bc(v_new.x.petsc_vec, bcs_v)
        u_new.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        v_new.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        u.x.array[:] = u_new.x.array
        v.x.array[:] = v_new.x.array
        a.x.array[:] = a_new.x.array
        alpha_lb.x.array[:] = alpha.x.array

        dyn["t"].append(t_cur)
        dyn["U"].append(float(u_right_val.value))
        dyn["F"].append(domain.comm.allreduce(fem.assemble_scalar(reaction_right_form), op=MPI.SUM))
        K_now = domain.comm.allreduce(fem.assemble_scalar(kinetic_energy_form), op=MPI.SUM)
        dyn["K"].append(K_now)
        dyn["P_el"].append(domain.comm.allreduce(fem.assemble_scalar(strain_energy_form),     op=MPI.SUM))
        dyn["P_f"].append(domain.comm.allreduce(fem.assemble_scalar(foundation_energy_form), op=MPI.SUM))
        dyn["S"].append(domain.comm.allreduce(fem.assemble_scalar(fracture_energy_form),     op=MPI.SUM))
        # Cumulative dissipated energy: time integral of dissipated power
        diss_power_now = domain.comm.allreduce(
            fem.assemble_scalar(dissipated_power_form), op=MPI.SUM
        )
        diss_energy += dt * float(diss_power_now)
        dyn["D"].append(diss_energy)
        # Stored dynamic total (K + P_el + P_f + S).  Dissipated energy D is a
        # separate curve, not folded into the stored total.
        dyn["total"].append(K_now + dyn["P_el"][-1] + dyn["P_f"][-1] + dyn["S"][-1])
        dyn["n_cracks"].append(count_cracks(alpha.x.array))

        a_max = _alpha_max(alpha, domain.comm)

        pbar_dyn.update(1)
        pbar_dyn.set_postfix(t=f"{t_cur:.3f}",
                             U=f"{float(u_right_val.value):.3f}",
                             K=f"{K_now:.2e}",
                             a_max=f"{a_max:.3f}",
                             n_cr=dyn["n_cracks"][-1],
                             altmin=n_alt)

        if paraview and (step % snap_every == 0 or step == n_dyn_max):
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
    dyn_events = detect_crack_events(dyn["U"], dyn["S"], dyn["n_cracks"])

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
        "qs_events":        qs_events,
        "dyn_events":       dyn_events,
        "dissipated_total": diss_energy,
        "triang":           triangulation_from_domain(domain) if physics == "2D" else None,
    }

    if output_dir is None:
        output_dir = ROOT / "output"

    if plot and comm.rank == 0:
        png, pdf = plot_mechanical_run(result, model_parameters, mesh_parameters,
                                       loading_parameters, solver_parameters, output_dir)
        if verbose:
            print(f"  saved {png}")
            print(f"        {pdf}")

    if paraview and physics == "2D" and comm.rank == 0:
        xdmf_qs = export_paraview(domain, paraview_alpha_qs, paraview_u_qs,
                                  "mechanical",
                                  model_parameters, mesh_parameters,
                                  loading_parameters, solver_parameters,
                                  output_dir, tag="QS")
        xdmf_dyn = export_paraview(domain, paraview_alpha, paraview_u,
                                   "mechanical",
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
    cfg = get_defaults("mechanical")

    # ----- edit *here* to switch model / physics --------------------------
    cfg["solver_parameters"]["model"]   = "AT1"        # "AT1" or "AT2"
    cfg["mesh_parameters"]["physics"]   = "2D"         # "1D"  or "2D"
    cfg["mesh_parameters"]["mesh_per_lhat"] = 3        # cells per l_hat
    # ----------------------------------------------------------------------

    run_problem(**cfg)
