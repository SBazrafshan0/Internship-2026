"""
problems/thermal.py
===================
**Thermal** phase-field fragmentation problem.

Geometry & loading
------------------
* Bar :math:`[0, L_x]` (1D) or *unstructured* triangulation of
  :math:`[0, L_x]\\times[0, L_y]` (2D, Gmsh-generated).  Mesh resolution
  driven by ``mesh_per_lhat``.
* Free ends in displacement; the foundation alone removes the rigid body
  motion.
* Uniform thermal eigenstrain :math:`\\theta(t)` ramped as in the
  mechanical problem.
* Damage free at every boundary.

Stopping criterion
------------------
The QS and Dyn loops run until *any* damage point reaches
``loading_parameters["alpha_break"]`` (default 0.99), with safety cap at
``t = loading_parameters["t_max"]`` (default 3.0).

Switches
--------
``solver_parameters["model"]`` -- ``"AT1"`` or ``"AT2"``.
``mesh_parameters["physics"]`` -- ``"1D"`` or ``"2D"``.
``mesh_parameters["shape"]``   -- registered shape (default ``"rectangle"``).
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
def _make_thermal_loader(loading_parameters):
    t_sp     = sp.Symbol("t", real=True)
    T0_v     = loading_parameters["T0"]
    ThMax_v  = loading_parameters["theta_max"]
    Thdot    = ThMax_v / (np.sqrt(T0_v ** 2 + 1.0 ** 2) - T0_v)
    Theta_sp = Thdot * (sp.sqrt(T0_v ** 2 + t_sp ** 2) - T0_v)
    return sp.lambdify(t_sp, Theta_sp, "numpy")


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
) -> dict:
    """One *quasi-static + dynamic* thermal fragmentation run."""
    # Geometry is a property of the *problem*, not a user-tunable knob.
    mesh_parameters["shape"] = _PROBLEM_SHAPE

    physics    = mesh_parameters["physics"]
    model_name = solver_parameters["model"]
    n_qs       = loading_parameters["N_steps_qs"]
    n_dyn      = loading_parameters["N_steps_dyn"]
    alpha_break = float(loading_parameters.get("alpha_break", 0.99))
    t_max       = float(loading_parameters.get("t_max", 3.0))
    N_snap      = loading_parameters.get("N_snapshots", 6)

    if verbose and comm.rank == 0:
        print(
            f"\n[THERMAL | {physics} | {model_name}]  "
            f"l_hat={model_parameters['l_hat']}, "
            f"Lambda={model_parameters['Lambda']}, "
            f"eta={model_parameters['eta']}  |  "
            f"mesh_per_lhat={mesh_parameters.get('mesh_per_lhat', 4)}  |  "
            f"dt_QS=1/{n_qs}, dt_Dyn=1/{n_dyn}  |  "
            f"stop at alpha={alpha_break} or t={t_max}"
        )

    # -------------------------------------------------------------------------
    # Mesh / spaces
    # -------------------------------------------------------------------------
    domain, mt, dx, ds = create_mesh_and_tags(mesh_parameters, model_parameters, comm)
    V_u, V_alpha = _setup_function_spaces(domain, physics)
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

    bcs_u, bcs_v, bcs_a, bcs_alpha = [], [], [], []

    # -------------------------------------------------------------------------
    # Constants + energies
    # -------------------------------------------------------------------------
    Lambda_c  = fem.Constant(domain, PETSc.ScalarType(model_parameters["Lambda"]))
    l_hat_c   = fem.Constant(domain, PETSc.ScalarType(model_parameters["l_hat"]))
    eta_c     = fem.Constant(domain, PETSc.ScalarType(model_parameters["eta"]))
    delta_t_c = fem.Constant(domain, PETSc.ScalarType(1.0 / n_dyn))
    theta_c   = fem.Constant(domain, PETSc.ScalarType(0.0))

    g = g_degradation(alpha)
    eps_e = elastic_strain(u, theta_c, physics)

    if physics == "1D":
        strain_e_density     = 0.5 * g * eps_e ** 2
        foundation_e_density = 0.5 * Lambda_c ** 2 * u ** 2
        kinetic_e_density    = 0.5 * eta_c ** 2 * v ** 2
        mean_stress_expr     = g * eps_e
    else:
        strain_e_density     = 0.5 * g * ufl.inner(eps_e, eps_e)
        foundation_e_density = 0.5 * Lambda_c ** 2 * ufl.dot(u, u)
        kinetic_e_density    = 0.5 * eta_c ** 2 * ufl.dot(v, v)
        sigma = g * 2.0 * eps_e
        mean_stress_expr     = sigma[0, 0]

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
    mean_stress_form       = fem.form(mean_stress_expr * dx)
    error_L2_alpha_form    = fem.form((alpha - alpha_old_iter) ** 2 * dx)

    u_test     = ufl.TestFunction(V_u)
    alpha_test = ufl.TestFunction(V_alpha)

    Res_u_qs     = ufl.derivative(potential_energy, u, u_test)
    Res_alpha_qs = ufl.derivative(potential_energy + fracture_energy, alpha, alpha_test)
    if physics == "1D":
        inertia_term = eta_c ** 2 * a_new * u_test * dx
    else:
        inertia_term = eta_c ** 2 * ufl.dot(a_new, u_test) * dx
    Res_acc = inertia_term + ufl.derivative(potential_energy, u, u_test)

    Theta_fn = _make_thermal_loader(loading_parameters)

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
    # QS loop -- while max(alpha) < alpha_break and t < t_max
    # -------------------------------------------------------------------------
    u.x.array[:] = 0.0; v.x.array[:] = 0.0; a.x.array[:] = 0.0
    alpha.x.array[:] = 0.0
    alpha_lb.x.array[:] = 0.0
    alpha_ub.x.array[:] = 1.0
    theta_c.value = 0.0

    qs = {"t": [], "theta": [], "sigma_bar": [], "P_el": [], "P_f": [], "S": [], "total": []}
    qs_snapshots = []
    dt_qs = 1.0 / n_qs
    n_qs_max = max(1, int(np.ceil(t_max / dt_qs)))
    snap_every_qs = max(1, n_qs_max // max(1, N_snap))
    pbar_qs = tqdm(total=n_qs_max,
                   desc=f"QS  [{physics}|{model_name}]",
                   dynamic_ncols=True,
                   disable=not (verbose and comm.rank == 0))

    t_cur = 0.0
    i = 0
    while t_cur + 1e-12 < t_max:
        i += 1
        t_cur += dt_qs
        theta_c.value = float(Theta_fn(t_cur))
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

        a_max = _alpha_max(alpha, domain.comm)
        pbar_qs.update(1)
        pbar_qs.set_postfix(t=f"{t_cur:.3f}", theta=f"{float(theta_c.value):.3f}",
                            a_max=f"{a_max:.3f}", altmin=n_alt)

        if i % snap_every_qs == 0 or a_max >= alpha_break:
            qs_snapshots.append({"step": i, "t": float(t_cur),
                                 "theta": float(theta_c.value),
                                 "alpha": alpha.x.array.copy()})
        if a_max >= alpha_break:
            break
    pbar_qs.close()

    alpha_qs_final = alpha.x.array.copy()
    for k in qs:
        qs[k] = np.asarray(qs[k])

    # -------------------------------------------------------------------------
    # SNES setup (Dyn)
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

    # -------------------------------------------------------------------------
    # Dynamic loop -- while max(alpha) < alpha_break and t < t_max
    # -------------------------------------------------------------------------
    for fn in (u, u_new, v, v_new, a, a_new):
        fn.x.array[:] = 0.0
    alpha.x.array[:] = 0.0
    alpha_lb.x.array[:] = 0.0
    alpha_ub.x.array[:] = 1.0
    theta_c.value = 0.0

    dyn = {"t": [], "theta": [], "sigma_bar": [], "K": [], "P_el": [], "P_f": [], "S": [], "total": []}
    dyn_snapshots = []
    paraview_alpha = []
    paraview_u     = []
    dt = 1.0 / n_dyn
    n_dyn_max = max(1, int(np.ceil(t_max / dt)))
    snap_every_dyn = max(1, n_dyn_max // max(1, N_snap))
    pbar_dyn = tqdm(total=n_dyn_max,
                    desc=f"Dyn [{physics}|{model_name}]",
                    dynamic_ncols=True,
                    disable=not (verbose and comm.rank == 0))

    t_cur = 0.0
    step = 0
    while t_cur + 1e-12 < t_max:
        step += 1
        t_cur += dt
        theta_c.value = float(Theta_fn(t_cur))

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
        alpha_lb.x.array[:] = alpha.x.array

        dyn["t"].append(t_cur)
        dyn["theta"].append(float(theta_c.value))
        dyn["sigma_bar"].append(domain.comm.allreduce(fem.assemble_scalar(mean_stress_form), op=MPI.SUM))
        dyn["K"].append(domain.comm.allreduce(fem.assemble_scalar(kinetic_energy_form),       op=MPI.SUM))
        dyn["P_el"].append(domain.comm.allreduce(fem.assemble_scalar(strain_energy_form),     op=MPI.SUM))
        dyn["P_f"].append(domain.comm.allreduce(fem.assemble_scalar(foundation_energy_form), op=MPI.SUM))
        dyn["S"].append(domain.comm.allreduce(fem.assemble_scalar(fracture_energy_form),     op=MPI.SUM))
        dyn["total"].append(dyn["K"][-1] + dyn["P_el"][-1] + dyn["P_f"][-1] + dyn["S"][-1])

        a_max = _alpha_max(alpha, domain.comm)
        pbar_dyn.update(1)
        pbar_dyn.set_postfix(t=f"{t_cur:.3f}", theta=f"{float(theta_c.value):.3f}",
                             K=f"{dyn['K'][-1]:.2e}",
                             a_max=f"{a_max:.3f}", altmin=n_alt)

        if step % snap_every_dyn == 0 or a_max >= alpha_break:
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

        if a_max >= alpha_break:
            break
    pbar_dyn.close()

    alpha_dyn_final = alpha.x.array.copy()
    for k in dyn:
        dyn[k] = np.asarray(dyn[k])

    x_alpha = V_alpha.tabulate_dof_coordinates()[:, 0]

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
        "qs_snapshots":    qs_snapshots,
        "dyn_snapshots":   dyn_snapshots,
        "triang":          triangulation_from_domain(domain) if physics == "2D" else None,
    }

    if output_dir is None:
        output_dir = ROOT / "output"


    if plot and comm.rank == 0:
        png, pdf = plot_thermal_run(result, model_parameters, mesh_parameters,
                                    loading_parameters, solver_parameters, output_dir)
        if verbose:
            print(f"  saved {png}")
            print(f"        {pdf}")

    if paraview and comm.rank == 0:
        xdmf = export_paraview(domain, paraview_alpha, paraview_u,
                               "thermal",
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
    cfg = get_defaults("thermal")

    # ----- edit *here* to switch model / physics --------------------------
    cfg["solver_parameters"]["model"]   = "AT2"        # "AT1" or "AT2"
    cfg["mesh_parameters"]["physics"]   = "1D"         # "1D"  or "2D"
    cfg["mesh_parameters"]["mesh_per_lhat"] = 5
    # ----------------------------------------------------------------------

    run_problem(**cfg)
