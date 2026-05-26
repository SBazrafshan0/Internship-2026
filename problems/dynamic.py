"""
problems/dynamic.py
===================
**Mechanical / dynamic** phase-field fragmentation problem.

Geometry & loading
------------------
* Bar :math:`[0, L_x]` (1D) or rectangular strip :math:`[0, L_x] \\times
  [0, L_y]` (2D, triangular mesh).
* Left end clamped:  ``u = 0``  (in 2D both components).
* Right end pulled:  ``u_x = U(t)`` with the smooth ramp
  ``U(t) = U_dot ( sqrt(T0^2 + t^2) - T0 )``,  ``U_dot`` chosen so that
  ``U(1) = U_max``.  In 2D the *y*-component of the displacement is left
  free on the right edge.
* Damage Dirichlet:  ``alpha = 0`` on both end-edges (no crack can nucleate
  at the loading device).
* Elastic foundation everywhere:  energy density
  ``0.5 * Lambda^2 * |u|^2``.

Two computations are run for every parameter set:

1. **Quasi-static** (Alternate Minimisation) along the ramp ``t in (0,1]``.
2. **Dynamic**  (implicit Newmark-:math:`\\beta`) on the *same* ramp.

To change the *model variant* (AT1/AT2) or the *physics* (1D/2D), set
``solver_parameters["model"]`` and ``mesh_parameters["physics"]`` at the
bottom of the file.  Nothing else has to change.

Entry point
-----------
``run_problem(model_parameters, mesh_parameters, loading_parameters,
solver_parameters, AltMin_parameters, Newmark_parameters, plot=True,
paraview=True, output_dir=...)``
"""

from __future__ import annotations
import sys
from pathlib import Path

# Make sure the repository root is on the import path when this file is run
# directly (``python problems/dynamic.py``).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.imports import (
    np, sp, ufl, fem, df_mesh, MPI, PETSc, comm,
    Path, tqdm,
)
from tools.helpers   import (
    SNESProblem, make_linear_snes, make_damage_snes, alt_min_loop, print_mesh_info,
)
from tools.meshing   import create_mesh_and_tags, grad_sq, strain
from tools.solvers   import fracture_energy_density, g_degradation
from tools.parameters import (
    get_defaults, filename_stub,
)
from tools.plotting  import (
    plot_mechanical_run, export_paraview, triangulation_from_domain,
)


# =============================================================================
# Helpers specific to the mechanical problem
# =============================================================================
def _make_displacement_loader(loading_parameters):
    """
    Build the symbolic ramp ``U(t)``, ``V(t) = U'(t)``, ``A(t) = U''(t)`` and
    return three numpy-callable functions.
    """
    t_sp   = sp.Symbol("t", real=True)
    T0_v   = loading_parameters["T0"]
    Umax_v = loading_parameters["U_max"]
    Udot   = Umax_v / (np.sqrt(T0_v ** 2 + 1.0 ** 2) - T0_v)
    U_sp = Udot * (sp.sqrt(T0_v ** 2 + t_sp ** 2) - T0_v)
    V_sp = sp.diff(U_sp, t_sp)
    A_sp = sp.diff(V_sp, t_sp)
    U_fn = sp.lambdify(t_sp, U_sp, "numpy")
    V_fn = sp.lambdify(t_sp, V_sp, "numpy")
    A_fn = sp.lambdify(t_sp, A_sp, "numpy")
    return U_fn, V_fn, A_fn


def _setup_function_spaces(domain, physics):
    """Scalar damage space + (scalar in 1D, vector in 2D) displacement space."""
    V_alpha = fem.functionspace(domain, ("Lagrange", 1))
    if physics == "1D":
        V_u = fem.functionspace(domain, ("Lagrange", 1))
    else:
        V_u = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    return V_u, V_alpha


def _u_component(u, physics, idx=0):
    """Return ``u`` (1D) or ``u[idx]`` (2D)."""
    if physics == "1D":
        return u
    return u[idx]


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
    """
    Run one *quasi-static + dynamic* mechanical simulation.

    Returns a dictionary that contains the energy/reaction histories of both
    runs, the final damage fields, and (for 2D) a matplotlib triangulation so
    that the plotter can ``tripcolor`` directly.
    """
    physics    = mesh_parameters["physics"]
    model_name = solver_parameters["model"]
    n_qs       = loading_parameters["N_steps_qs"]
    n_dyn      = loading_parameters["N_steps_dyn"]

    if verbose and comm.rank == 0:
        print(
            f"\n[DYNAMIC | {physics} | {model_name}]  "
            f"l_hat={model_parameters['l_hat']}, "
            f"Lambda={model_parameters['Lambda']}, "
            f"eta={model_parameters['eta']}  |  "
            f"N_qs={n_qs}, N_dyn={n_dyn}"
        )

    # -------------------------------------------------------------------------
    # Mesh / tags / spaces
    # -------------------------------------------------------------------------
    domain, mt, dx, ds = create_mesh_and_tags(mesh_parameters, comm)
    V_u, V_alpha = _setup_function_spaces(domain, physics)
    gdim = domain.geometry.dim
    fdim = domain.topology.dim - 1

    if verbose:
        print_mesh_info(domain, V_u, V_alpha,
                        label=f"MECH {physics}", comm_=domain.comm)

    # Kinematic fields
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
        # Scalar displacement -> direct Dirichlet on V_u
        u_left_val  = fem.Constant(domain, PETSc.ScalarType(0.0))
        u_right_val = fem.Constant(domain, PETSc.ScalarType(0.0))
        v_left_val  = fem.Constant(domain, PETSc.ScalarType(0.0))
        v_right_val = fem.Constant(domain, PETSc.ScalarType(0.0))
        a_left_val  = fem.Constant(domain, PETSc.ScalarType(0.0))
        a_right_val = fem.Constant(domain, PETSc.ScalarType(0.0))

        left_dofs_u  = fem.locate_dofs_topological(V_u, fdim, left_facets)
        right_dofs_u = fem.locate_dofs_topological(V_u, fdim, right_facets)

        bcs_u = [
            fem.dirichletbc(u_left_val,  left_dofs_u,  V_u),
            fem.dirichletbc(u_right_val, right_dofs_u, V_u),
        ]
        bcs_v = [
            fem.dirichletbc(v_left_val,  left_dofs_u,  V_u),
            fem.dirichletbc(v_right_val, right_dofs_u, V_u),
        ]
        bcs_a = [
            fem.dirichletbc(a_left_val,  left_dofs_u,  V_u),
            fem.dirichletbc(a_right_val, right_dofs_u, V_u),
        ]

    else:  # 2D ----------------------------------------------------------------
        # Left edge: clamp both components.  Right edge: only x-component.
        zero_vec = fem.Function(V_u)
        zero_vec.x.array[:] = 0.0

        # Clamp on the left edge (both components) -- pass Function + ndarray
        # (form 3 of dolfinx.fem.dirichletbc).
        left_dofs_full = fem.locate_dofs_topological(V_u, fdim, left_facets)
        bc_left_clamp = fem.dirichletbc(zero_vec, left_dofs_full)

        # Imposed x-displacement on the right edge -- pass Constant + ndarray +
        # sub-space (form 2 of dolfinx.fem.dirichletbc).  Locating dofs on the
        # *single* sub-space ``V_u.sub(0)`` returns a single ndarray of dofs in
        # the parent space, which is what form 2 expects.
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

    # Damage: alpha=0 on both end-edges (no nucleation at the loading device)
    left_dofs_alpha  = fem.locate_dofs_topological(V_alpha, fdim, left_facets)
    right_dofs_alpha = fem.locate_dofs_topological(V_alpha, fdim, right_facets)
    bcs_alpha = [
        fem.dirichletbc(PETSc.ScalarType(0.0), left_dofs_alpha,  V_alpha),
        fem.dirichletbc(PETSc.ScalarType(0.0), right_dofs_alpha, V_alpha),
    ]

    # -------------------------------------------------------------------------
    # Material constants & energies
    # -------------------------------------------------------------------------
    Lambda_c  = fem.Constant(domain, PETSc.ScalarType(model_parameters["Lambda"]))
    l_hat_c   = fem.Constant(domain, PETSc.ScalarType(model_parameters["l_hat"]))
    eta_c     = fem.Constant(domain, PETSc.ScalarType(model_parameters["eta"]))
    delta_t_c = fem.Constant(domain, PETSc.ScalarType(1.0 / n_dyn))

    g = g_degradation(alpha)

    if physics == "1D":
        eps = strain(u, "1D")
        strain_e_density     = 0.5 * g * eps ** 2
        foundation_e_density = 0.5 * Lambda_c ** 2 * u ** 2
        kinetic_e_density    = 0.5 * eta_c ** 2 * v ** 2
        # Reaction at the right end:  (1-alpha)^2 * eps  evaluated on ds(2).
        reaction_form_expr = g * eps
    else:
        eps = strain(u, "2D")
        strain_e_density     = 0.5 * g * ufl.inner(eps, eps)
        foundation_e_density = 0.5 * Lambda_c ** 2 * ufl.dot(u, u)
        kinetic_e_density    = 0.5 * eta_c ** 2 * ufl.dot(v, v)
        # Horizontal reaction:  sigma_xx on the right edge.
        sigma = g * (2.0 * eps)       # mu = 1, lambda_lame = 0 normalisation
        reaction_form_expr = sigma[0, 0]

    fracture_e_density = fracture_energy_density(alpha, grad_sq(alpha, physics),
                                                 l_hat_c, model_name)

    strain_energy     = strain_e_density     * dx
    foundation_energy = foundation_e_density * dx
    potential_energy  = strain_energy + foundation_energy
    fracture_energy   = fracture_e_density   * dx
    kinetic_energy    = kinetic_e_density    * dx

    strain_energy_form     = fem.form(strain_energy)
    foundation_energy_form = fem.form(foundation_energy)
    fracture_energy_form   = fem.form(fracture_energy)
    kinetic_energy_form    = fem.form(kinetic_energy)
    reaction_right_form    = fem.form(reaction_form_expr * ds(2))
    error_L2_alpha_form    = fem.form((alpha - alpha_old_iter) ** 2 * dx)

    # -------------------------------------------------------------------------
    # Variational equations
    # -------------------------------------------------------------------------
    u_test     = ufl.TestFunction(V_u)
    alpha_test = ufl.TestFunction(V_alpha)

    Res_u_qs     = ufl.derivative(potential_energy, u, u_test)
    Res_alpha_qs = ufl.derivative(potential_energy + fracture_energy, alpha, alpha_test)

    if physics == "1D":
        inertia_term = eta_c ** 2 * a_new * u_test * dx
    else:
        inertia_term = eta_c ** 2 * ufl.dot(a_new, u_test) * dx

    Res_acc = inertia_term + ufl.derivative(potential_energy, u, u_test)

    # -------------------------------------------------------------------------
    # Loading ramp
    # -------------------------------------------------------------------------
    U_fn, V_fn, A_fn = _make_displacement_loader(loading_parameters)

    # -------------------------------------------------------------------------
    # Quasi-static SNES
    # -------------------------------------------------------------------------
    J_u_qs     = ufl.derivative(Res_u_qs,     u,     ufl.TrialFunction(V_u))
    J_alpha_qs = ufl.derivative(Res_alpha_qs, alpha, ufl.TrialFunction(V_alpha))

    elastic_problem_qs = SNESProblem(Res_u_qs,     u,     bcs_u,     J=J_u_qs)
    damage_problem_qs  = SNESProblem(Res_alpha_qs, alpha, bcs_alpha, J=J_alpha_qs)
    solver_u_qs     = make_linear_snes(elastic_problem_qs, V_u)
    solver_alpha_qs = make_damage_snes(damage_problem_qs,  alpha_lb, alpha_ub)

    # -------------------------------------------------------------------------
    # Quasi-static loop
    # -------------------------------------------------------------------------
    u.x.array[:] = 0.0; v.x.array[:] = 0.0; a.x.array[:] = 0.0
    alpha.x.array[:]    = 0.0
    alpha_lb.x.array[:] = 0.0
    alpha_ub.x.array[:] = 1.0

    qs = {"t": [], "U": [], "F": [], "P_el": [], "P_f": [], "S": [], "total": []}
    t_grid_qs = np.linspace(0.0, 1.0, n_qs + 1)[1:]

    qs_iter = (
        tqdm(t_grid_qs, desc=f"QS  [{physics}|{model_name}]",
             total=n_qs, dynamic_ncols=True, disable=not (verbose and comm.rank == 0))
        if verbose else t_grid_qs
    )
    for ti in qs_iter:
        u_right_val.value = float(U_fn(ti))
        n_alt = alt_min_loop(solver_u_qs, u.x.petsc_vec,
                             solver_alpha_qs, alpha,
                             alpha_old_iter, error_L2_alpha_form,
                             AltMin_parameters["max_iter"], AltMin_parameters["tol"],
                             comm_=domain.comm)
        alpha_lb.x.array[:] = alpha.x.array
        if hasattr(qs_iter, "set_postfix"):
            alpha_max = float(domain.comm.allreduce(alpha.x.array.max(), op=MPI.MAX))
            qs_iter.set_postfix(U=f"{float(u_right_val.value):.3f}",
                                a_max=f"{alpha_max:.3f}", altmin=n_alt)

        qs["t"].append(float(ti))
        qs["U"].append(float(u_right_val.value))
        qs["F"].append(domain.comm.allreduce(fem.assemble_scalar(reaction_right_form),  op=MPI.SUM))
        qs["P_el"].append(domain.comm.allreduce(fem.assemble_scalar(strain_energy_form),     op=MPI.SUM))
        qs["P_f"].append(domain.comm.allreduce(fem.assemble_scalar(foundation_energy_form), op=MPI.SUM))
        qs["S"].append(domain.comm.allreduce(fem.assemble_scalar(fracture_energy_form),     op=MPI.SUM))
        qs["total"].append(qs["P_el"][-1] + qs["P_f"][-1] + qs["S"][-1])

    alpha_qs_final = alpha.x.array.copy()
    for k in qs:
        qs[k] = np.asarray(qs[k])

    # -------------------------------------------------------------------------
    # Dynamic (Newmark) SNES
    # -------------------------------------------------------------------------
    beta_v  = Newmark_parameters["beta"]
    gamma_v = Newmark_parameters["gamma"]

    def u_newmark(u_, v_, a_, a_new_, dt):
        return u_ + dt * v_ + 0.5 * dt ** 2 * ((1.0 - 2.0 * beta_v) * a_ + 2.0 * beta_v * a_new_)

    Res_acc_newmark = ufl.replace(Res_acc, {u: u_newmark(u, v, a, a_new, delta_t_c)})
    Res_alpha_dyn   = ufl.replace(Res_alpha_qs, {u: u_newmark(u, v, a, a_new, delta_t_c)})

    J_acc_newmark = ufl.derivative(Res_acc_newmark, a_new, ufl.TrialFunction(V_u))
    J_alpha_dyn   = ufl.derivative(Res_alpha_dyn,   alpha, ufl.TrialFunction(V_alpha))

    acc_problem        = SNESProblem(Res_acc_newmark, a_new, bcs_a,     J=J_acc_newmark)
    damage_problem_dyn = SNESProblem(Res_alpha_dyn,   alpha, bcs_alpha, J=J_alpha_dyn)
    solver_acc       = make_linear_snes(acc_problem,        V_u)
    solver_alpha_dyn = make_damage_snes(damage_problem_dyn, alpha_lb, alpha_ub)

    delta_t_c.value = 1.0 / n_dyn

    # Re-initialise
    for fn in (u, u_new, v, v_new, a, a_new):
        fn.x.array[:] = 0.0
    alpha.x.array[:] = 0.0
    alpha_lb.x.array[:] = 0.0
    alpha_ub.x.array[:] = 1.0

    dyn = {"t": [], "U": [], "F": [], "K": [], "P_el": [], "P_f": [], "S": [], "total": []}

    # Time-history for Paraview
    N_snap = loading_parameters.get("N_snapshots", 6)
    snap_idx_dyn = set(np.unique(np.linspace(0, n_dyn - 1, N_snap, dtype=int)).tolist())
    paraview_alpha = []
    paraview_u     = []

    dt    = 1.0 / n_dyn
    t_cur = 0.0
    dyn_iter = (
        tqdm(range(n_dyn), desc=f"Dyn [{physics}|{model_name}]",
             total=n_dyn, dynamic_ncols=True, disable=not (verbose and comm.rank == 0))
        if verbose else range(n_dyn)
    )
    for step in dyn_iter:
        t_cur += dt
        u_right_val.value = float(U_fn(t_cur))
        v_right_val.value = float(V_fn(t_cur))
        a_right_val.value = float(A_fn(t_cur))

        n_alt = alt_min_loop(solver_acc, a_new.x.petsc_vec,
                             solver_alpha_dyn, alpha,
                             alpha_old_iter, error_L2_alpha_form,
                             AltMin_parameters["max_iter"], AltMin_parameters["tol"],
                             comm_=domain.comm)

        # Newmark update
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
        dyn["K"].append(domain.comm.allreduce(fem.assemble_scalar(kinetic_energy_form), op=MPI.SUM))
        dyn["P_el"].append(domain.comm.allreduce(fem.assemble_scalar(strain_energy_form),     op=MPI.SUM))
        dyn["P_f"].append(domain.comm.allreduce(fem.assemble_scalar(foundation_energy_form), op=MPI.SUM))
        dyn["S"].append(domain.comm.allreduce(fem.assemble_scalar(fracture_energy_form),     op=MPI.SUM))
        dyn["total"].append(dyn["K"][-1] + dyn["P_el"][-1] + dyn["P_f"][-1] + dyn["S"][-1])

        if step in snap_idx_dyn and paraview:
            alpha_snap = fem.Function(V_alpha, name="Damage")
            alpha_snap.x.array[:] = alpha.x.array
            paraview_alpha.append((t_cur, alpha_snap))
            u_snap = fem.Function(V_u, name="Displacement")
            u_snap.x.array[:] = u.x.array
            paraview_u.append((t_cur, u_snap))

        if hasattr(dyn_iter, "set_postfix"):
            alpha_max = float(domain.comm.allreduce(alpha.x.array.max(), op=MPI.MAX))
            dyn_iter.set_postfix(U=f"{float(u_right_val.value):.3f}",
                                 K=f"{dyn['K'][-1]:.2e}",
                                 a_max=f"{alpha_max:.3f}", altmin=n_alt)

    alpha_dyn_final = alpha.x.array.copy()
    for k in dyn:
        dyn[k] = np.asarray(dyn[k])

    # Coordinates of the damage dofs (used by plotters)
    x_alpha = V_alpha.tabulate_dof_coordinates()[:, 0]

    # -------------------------------------------------------------------------
    # House-keeping and post-processing
    # -------------------------------------------------------------------------
    for s in (solver_u_qs, solver_alpha_qs, solver_acc, solver_alpha_dyn):
        s.destroy()

    result = {
        "physics":         physics,
        "model":           model_name,
        "qs":              qs,
        "dyn":             dyn,
        "x_alpha":         x_alpha,
        "alpha_qs_final":  alpha_qs_final,
        "alpha_dyn_final": alpha_dyn_final,
        "triang":          triangulation_from_domain(domain) if physics == "2D" else None,
    }

    if output_dir is None:
        output_dir = ROOT / "output"

    if plot and comm.rank == 0:
        png, pdf = plot_mechanical_run(result, model_parameters, mesh_parameters,
                                       loading_parameters, solver_parameters, output_dir)
        if verbose:
            print(f"  saved {png}")
            print(f"        {pdf}")

    if paraview and comm.rank == 0:
        xdmf = export_paraview(domain, paraview_alpha, paraview_u,
                               "mechanical",
                               model_parameters, mesh_parameters,
                               loading_parameters, solver_parameters,
                               output_dir)
        if verbose and xdmf:
            print(f"        {xdmf}")

    return result


# =============================================================================
# Stand-alone execution
# =============================================================================
if __name__ == "__main__":
    cfg = get_defaults("mechanical")

    # ----- edit *here* to switch model / physics --------------------------
    cfg["solver_parameters"]["model"]   = "AT2"        # "AT1" or "AT2"
    cfg["mesh_parameters"]["physics"]   = "2D"         # "1D"  or "2D"
    # 2D-only settings (ignored in 1D):
    cfg["mesh_parameters"]["nx"]        = 50
    cfg["mesh_parameters"]["ny"]        = 20
    # ----------------------------------------------------------------------

    run_problem(**cfg)
