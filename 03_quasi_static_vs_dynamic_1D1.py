import os
import itertools
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

import dolfinx
from dolfinx import mesh as df_mesh, fem
import dolfinx.fem.petsc
import ufl

from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD

## SNESProblem helper
class SNESProblem:
    def __init__(self, F, u, bcs, J=None):
        V = u.function_space
        du = ufl.TrialFunction(V)
        self.L = fem.form(F)
        if J is None:
            self.a = fem.form(ufl.derivative(F, u, du))
        else:
            self.a = fem.form(J)
        self.bcs = bcs
        self.u = u

    def F(self, snes, x, F):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        with F.localForm() as f_local:
            f_local.set(0.0)
        fem.petsc.assemble_vector(F, self.L)
        fem.petsc.apply_lifting(F, [self.a], bcs=[self.bcs], x0=[x], alpha=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(F, self.bcs, x, -1.0)

    def J(self, snes, x, J, P):
        J.zeroEntries()
        fem.petsc.assemble_matrix(J, self.a, bcs=self.bcs)
        J.assemble()


def run_simulation(l_hat_val, lambda_val, eta_val):
    """
    Runs the 1D phase-field damage model for a given set of parameters.
    """
    ## Parameters
    model_parameters = {
        "l_hat":  l_hat_val,
        "Lambda": lambda_val,
        "eta":    eta_val,
    }

    mesh_parameters = {"nx": 200}

    loading_parameters = {
        "U_max":        1.4,
        "T0":           1.0,
        "N_steps_qs":   20,
        "N_steps_dyn":  60,
    }

    AltMin_parameters  = {"max_iter": 200, "tol": 1e-7}
    Newmark_parameters = {"beta": 0.25, "gamma": 0.5}

    if comm.rank == 0:
        print(f"\n--- Starting run with l_hat={l_hat_val}, Lambda={lambda_val}, eta={eta_val} ---")

    ## Mesh, spaces, surface tags
    domain = df_mesh.create_interval(comm, mesh_parameters["nx"], (0.0, 1.0))
    gdim = domain.topology.dim
    fdim = gdim - 1

    V_u     = fem.functionspace(domain, ("Lagrange", 1))
    V_alpha = fem.functionspace(domain, ("Lagrange", 1))

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

    # markers
    def left_marker(x):  return np.isclose(x[0], 0.0)
    def right_marker(x): return np.isclose(x[0], 1.0)

    left_facets  = df_mesh.locate_entities_boundary(domain, fdim, left_marker)
    right_facets = df_mesh.locate_entities_boundary(domain, fdim, right_marker)

    all_facets = np.concatenate([left_facets, right_facets]).astype(np.int32)
    markers    = np.concatenate([
        np.full(len(left_facets),  1, dtype=np.int32),
        np.full(len(right_facets), 2, dtype=np.int32),
    ])
    sort_ix = np.argsort(all_facets)
    mt = df_mesh.meshtags(domain, fdim, all_facets[sort_ix], markers[sort_ix])

    dx = ufl.Measure("dx", domain=domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)

    ## Boundary conditions
    u_left_val  = fem.Constant(domain, PETSc.ScalarType(0.0))
    u_right_val = fem.Constant(domain, PETSc.ScalarType(0.0))
    v_left_val  = fem.Constant(domain, PETSc.ScalarType(0.0))
    v_right_val = fem.Constant(domain, PETSc.ScalarType(0.0))
    a_left_val  = fem.Constant(domain, PETSc.ScalarType(0.0))
    a_right_val = fem.Constant(domain, PETSc.ScalarType(0.0))

    left_dofs_u      = fem.locate_dofs_topological(V_u,     fdim, left_facets)
    right_dofs_u     = fem.locate_dofs_topological(V_u,     fdim, right_facets)
    left_dofs_alpha  = fem.locate_dofs_topological(V_alpha, fdim, left_facets)
    right_dofs_alpha = fem.locate_dofs_topological(V_alpha, fdim, right_facets)

    bc_u_left  = fem.dirichletbc(u_left_val,  left_dofs_u,  V_u)
    bc_u_right = fem.dirichletbc(u_right_val, right_dofs_u, V_u)
    bc_v_left  = fem.dirichletbc(v_left_val,  left_dofs_u,  V_u)
    bc_v_right = fem.dirichletbc(v_right_val, right_dofs_u, V_u)
    bc_a_left  = fem.dirichletbc(a_left_val,  left_dofs_u,  V_u)
    bc_a_right = fem.dirichletbc(a_right_val, right_dofs_u, V_u)
    bc_alpha_left  = fem.dirichletbc(PETSc.ScalarType(0.0), left_dofs_alpha,  V_alpha)
    bc_alpha_right = fem.dirichletbc(PETSc.ScalarType(0.0), right_dofs_alpha, V_alpha)

    bcs_u     = [bc_u_left,     bc_u_right]
    bcs_v     = [bc_v_left,     bc_v_right]
    bcs_a     = [bc_a_left,     bc_a_right]
    bcs_alpha = [bc_alpha_left, bc_alpha_right]

    ## Energies and residuals
    Lambda_c  = fem.Constant(domain, PETSc.ScalarType(model_parameters["Lambda"]))
    l_hat_c   = fem.Constant(domain, PETSc.ScalarType(model_parameters["l_hat"]))
    eta_c     = fem.Constant(domain, PETSc.ScalarType(model_parameters["eta"]))
    delta_t_c = fem.Constant(domain, PETSc.ScalarType(1.0 / loading_parameters["N_steps_dyn"]))

    def strain(w):         return w.dx(0)
    def stress(w, alpha_): return (1.0 - alpha_)**2 * strain(w)

    strain_e_density     = 0.5 * (1.0 - alpha)**2 * strain(u)**2
    foundation_e_density = 0.5 * Lambda_c**2 * u**2
    fracture_e_density   = alpha + l_hat_c**2 * alpha.dx(0)**2
    kinetic_e_density    = 0.5 * eta_c**2 * v**2

    strain_energy     = strain_e_density     * dx
    foundation_energy = foundation_e_density * dx
    potential_energy  = strain_energy + foundation_energy
    fracture_energy   = fracture_e_density   * dx
    kinetic_energy    = kinetic_e_density    * dx

    strain_energy_form     = fem.form(strain_energy)
    foundation_energy_form = fem.form(foundation_energy)
    fracture_energy_form   = fem.form(fracture_energy)
    kinetic_energy_form    = fem.form(kinetic_energy)
    reaction_right_form    = fem.form(stress(u, alpha) * ds(2))
    error_L2_alpha_form    = fem.form((alpha - alpha_old_iter)**2 * dx)

    u_test     = ufl.TestFunction(V_u)
    alpha_test = ufl.TestFunction(V_alpha)

    Res_u_qs     = ufl.derivative(potential_energy, u, u_test)
    Res_alpha_qs = ufl.derivative(potential_energy + fracture_energy, alpha, alpha_test)
    Res_acc = eta_c**2 * a_new * u_test * dx + ufl.derivative(potential_energy, u, u_test)

    ## Loading
    t_sp   = sp.Symbol("t", real=True)
    T0_v   = loading_parameters["T0"]
    Umax_v = loading_parameters["U_max"]

    Udot_p = Umax_v / (np.sqrt(T0_v**2 + 1.0**2) - T0_v)
    U_imp_sp = Udot_p * (sp.sqrt(T0_v**2 + t_sp**2) - T0_v)
    V_imp_sp = sp.diff(U_imp_sp, t_sp)
    A_imp_sp = sp.diff(V_imp_sp, t_sp)

    U_fn = sp.lambdify(t_sp, U_imp_sp, "numpy")
    V_fn = sp.lambdify(t_sp, V_imp_sp, "numpy")
    A_fn = sp.lambdify(t_sp, A_imp_sp, "numpy")

    ## QS solver
    J_u_qs     = ufl.derivative(Res_u_qs,     u,     ufl.TrialFunction(V_u))
    J_alpha_qs = ufl.derivative(Res_alpha_qs, alpha, ufl.TrialFunction(V_alpha))

    elastic_problem_qs = SNESProblem(Res_u_qs,     u,     bcs_u,     J=J_u_qs)
    damage_problem_qs  = SNESProblem(Res_alpha_qs, alpha, bcs_alpha, J=J_alpha_qs)

    b_u_qs       = fem.petsc.create_vector(V_u)
    J_u_qs_m     = fem.petsc.create_matrix(elastic_problem_qs.a)
    b_alpha_qs   = fem.petsc.create_vector(V_alpha)
    J_alpha_qs_m = fem.petsc.create_matrix(damage_problem_qs.a)

    solver_u_qs = PETSc.SNES().create()
    solver_u_qs.setType("ksponly")
    solver_u_qs.setFunction(elastic_problem_qs.F, b_u_qs)
    solver_u_qs.setJacobian(elastic_problem_qs.J, J_u_qs_m)
    solver_u_qs.setTolerances(rtol=1.0e-9, max_it=50)
    solver_u_qs.getKSP().setType("preonly")
    solver_u_qs.getKSP().getPC().setType("lu")
    solver_u_qs.getKSP().getPC().setFactorSolverType("mumps")

    solver_alpha_qs = PETSc.SNES().create()
    solver_alpha_qs.setType("vinewtonrsls")
    solver_alpha_qs.setFunction(damage_problem_qs.F, b_alpha_qs)
    solver_alpha_qs.setJacobian(damage_problem_qs.J, J_alpha_qs_m)
    solver_alpha_qs.setTolerances(rtol=1.0e-9, max_it=50)
    solver_alpha_qs.getKSP().setType("preonly")
    solver_alpha_qs.getKSP().getPC().setType("lu")
    solver_alpha_qs.setVariableBounds(alpha_lb.x.petsc_vec, alpha_ub.x.petsc_vec)
    solver_alpha_qs.getKSP().getPC().setFactorSolverType("mumps")

    ## QS load loop
    u.x.array[:] = 0.0; v.x.array[:] = 0.0; a.x.array[:] = 0.0
    alpha.x.array[:]    = 0.0
    alpha_lb.x.array[:] = 0.0
    alpha_ub.x.array[:] = 1.0

    qs = {"t": [], "U": [], "F": [], "P_el": [], "P_f": [], "S": [], "total": []}

    N_qs      = loading_parameters["N_steps_qs"]
    t_grid_qs = np.linspace(0.0, 1.0, N_qs + 1)[1:]

    for i, ti in enumerate(t_grid_qs):
        u_right_val.value = float(U_fn(ti))

        n_alt = 0
        for n_alt in range(1, AltMin_parameters["max_iter"] + 1):
            solver_u_qs.solve(None, u.x.petsc_vec)
            u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            solver_alpha_qs.solve(None, alpha.x.petsc_vec)
            alpha.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            err = comm.allreduce(fem.assemble_scalar(error_L2_alpha_form), op=MPI.SUM)
            err_alpha = float(np.sqrt(max(err, 0.0)))
            alpha_old_iter.x.array[:] = alpha.x.array

            if err_alpha <= AltMin_parameters["tol"]: break

        alpha_lb.x.array[:] = alpha.x.array

        qs["t"].append(float(ti))
        qs["U"].append(float(u_right_val.value))
        qs["F"].append(comm.allreduce(fem.assemble_scalar(reaction_right_form),    op=MPI.SUM))
        qs["P_el"].append(comm.allreduce(fem.assemble_scalar(strain_energy_form),     op=MPI.SUM))
        qs["P_f"].append(comm.allreduce(fem.assemble_scalar(foundation_energy_form), op=MPI.SUM))
        qs["S"].append(comm.allreduce(fem.assemble_scalar(fracture_energy_form),     op=MPI.SUM))
        qs["total"].append(qs["P_el"][-1] + qs["P_f"][-1] + qs["S"][-1])

    u_qs_final     = u.x.array.copy()
    alpha_qs_final = alpha.x.array.copy()
    for k in qs: qs[k] = np.array(qs[k])

    ## Dynamic solver
    beta_v  = Newmark_parameters["beta"]
    gamma_v = Newmark_parameters["gamma"]

    def u_newmark(u_, v_, a_, a_new_, dt):
        return u_ + dt*v_ + 0.5*dt**2 * ((1.0 - 2.0*beta_v)*a_ + 2.0*beta_v*a_new_)

    Res_acc_newmark = ufl.replace(Res_acc, {u: u_newmark(u, v, a, a_new, delta_t_c)})
    Res_alpha_dyn   = ufl.replace(Res_alpha_qs, {u: u_newmark(u, v, a, a_new, delta_t_c)})

    J_acc_newmark = ufl.derivative(Res_acc_newmark, a_new, ufl.TrialFunction(V_u))
    J_alpha_dyn   = ufl.derivative(Res_alpha_dyn,   alpha, ufl.TrialFunction(V_alpha))

    acc_problem        = SNESProblem(Res_acc_newmark, a_new, bcs_a,     J=J_acc_newmark)
    damage_problem_dyn = SNESProblem(Res_alpha_dyn,   alpha, bcs_alpha, J=J_alpha_dyn)

    b_acc_dyn     = fem.petsc.create_vector(V_u)
    J_acc_dyn_m   = fem.petsc.create_matrix(acc_problem.a)
    b_alpha_dyn   = fem.petsc.create_vector(V_alpha)
    J_alpha_dyn_m = fem.petsc.create_matrix(damage_problem_dyn.a)

    solver_acc = PETSc.SNES().create()
    solver_acc.setType("ksponly")
    solver_acc.setFunction(acc_problem.F, b_acc_dyn)
    solver_acc.setJacobian(acc_problem.J, J_acc_dyn_m)
    solver_acc.setTolerances(rtol=1.0e-9, max_it=50)
    solver_acc.getKSP().setType("preonly")
    solver_acc.getKSP().getPC().setType("lu")
    solver_acc.getKSP().getPC().setFactorSolverType("mumps")

    solver_alpha_dyn = PETSc.SNES().create()
    solver_alpha_dyn.setType("vinewtonrsls")
    solver_alpha_dyn.setFunction(damage_problem_dyn.F, b_alpha_dyn)
    solver_alpha_dyn.setJacobian(damage_problem_dyn.J, J_alpha_dyn_m)
    solver_alpha_dyn.setTolerances(rtol=1.0e-9, max_it=50)
    solver_alpha_dyn.getKSP().setType("preonly")
    solver_alpha_dyn.getKSP().getPC().setType("lu")
    solver_alpha_dyn.setVariableBounds(alpha_lb.x.petsc_vec, alpha_ub.x.petsc_vec)
    solver_alpha_dyn.getKSP().getPC().setFactorSolverType("mumps")

    delta_t_c.value = 1.0 / loading_parameters["N_steps_dyn"]

    ## Dynamic time loop
    u.x.array[:]     = 0.0; u_new.x.array[:] = 0.0
    v.x.array[:]     = 0.0; v_new.x.array[:] = 0.0
    a.x.array[:]     = 0.0; a_new.x.array[:] = 0.0
    alpha.x.array[:]    = 0.0
    alpha_lb.x.array[:] = 0.0
    alpha_ub.x.array[:] = 1.0

    dyn = {"t": [], "U": [], "F": [], "K": [], "P_el": [], "P_f": [], "S": [], "total": []}

    N_dyn = loading_parameters["N_steps_dyn"]
    dt    = 1.0 / N_dyn
    t_cur = 0.0

    for step in range(N_dyn):
        t_cur += dt
        u_right_val.value = float(U_fn(t_cur))
        v_right_val.value = float(V_fn(t_cur))
        a_right_val.value = float(A_fn(t_cur))

        n_alt = 0
        for n_alt in range(1, AltMin_parameters["max_iter"] + 1):
            solver_acc.solve(None, a_new.x.petsc_vec)
            a_new.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            solver_alpha_dyn.solve(None, alpha.x.petsc_vec)
            alpha.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            err = comm.allreduce(fem.assemble_scalar(error_L2_alpha_form), op=MPI.SUM)
            err_alpha = float(np.sqrt(max(err, 0.0)))
            alpha_old_iter.x.array[:] = alpha.x.array

            if err_alpha <= AltMin_parameters["tol"]: break

        u_new.x.array[:] = (u.x.array + dt * v.x.array + 0.5 * dt**2 * ((1.0 - 2.0*beta_v) * a.x.array + 2.0*beta_v * a_new.x.array))
        v_new.x.array[:] = (v.x.array + dt * ((1.0 - gamma_v) * a.x.array + gamma_v * a_new.x.array))
        
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
        dyn["F"].append(comm.allreduce(fem.assemble_scalar(reaction_right_form), op=MPI.SUM))
        dyn["K"].append(comm.allreduce(fem.assemble_scalar(kinetic_energy_form), op=MPI.SUM))
        dyn["P_el"].append(comm.allreduce(fem.assemble_scalar(strain_energy_form),     op=MPI.SUM))
        dyn["P_f"].append(comm.allreduce(fem.assemble_scalar(foundation_energy_form), op=MPI.SUM))
        dyn["S"].append(comm.allreduce(fem.assemble_scalar(fracture_energy_form),     op=MPI.SUM))
        dyn["total"].append(dyn["K"][-1] + dyn["P_el"][-1] + dyn["P_f"][-1] + dyn["S"][-1])

    u_dyn_final     = u.x.array.copy()
    alpha_dyn_final = alpha.x.array.copy()
    for k in dyn: dyn[k] = np.array(dyn[k])

    ## Comparison & Plotting
    x_alpha  = V_alpha.tabulate_dof_coordinates()[:, 0]
    ix_alpha = np.argsort(x_alpha)

    P_el_dyn_at_qs = np.interp(qs["U"], dyn["U"], dyn["P_el"])
    P_f_dyn_at_qs  = np.interp(qs["U"], dyn["U"], dyn["P_f"])
    S_dyn_at_qs    = np.interp(qs["U"], dyn["U"], dyn["S"])
    tot_dyn_at_qs  = np.interp(qs["U"], dyn["U"], dyn["total"])

    err_P_el = float(np.max(np.abs(qs["P_el"]  - P_el_dyn_at_qs)))
    err_P_f  = float(np.max(np.abs(qs["P_f"]   - P_f_dyn_at_qs)))
    err_S    = float(np.max(np.abs(qs["S"]     - S_dyn_at_qs)))
    err_tot  = float(np.max(np.abs(qs["total"] - tot_dyn_at_qs)))

    if comm.rank == 0:
        fig = plt.figure(figsize=(16, 9))
        gs  = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2], hspace=0.30, wspace=0.25)
        ax_force  = fig.add_subplot(gs[0, 0])
        ax_dam    = fig.add_subplot(gs[0, 1])
        ax_energy = fig.add_subplot(gs[1, :])

        u_max_val   = float(np.max(qs["U"]))
        n_steps_qs  = len(qs["U"])
        n_steps_dyn = len(dyn["U"])
        mesh_val      = mesh_parameters["nx"]
        smoth_val     = loading_parameters["T0"]
        
        header_text = (
            f"Model: $\\hat\\ell={l_hat_val}$ | $\\Lambda={lambda_val}$ | $\\eta={eta_val}$\n"
            f"Run: $U_{{\\max}} = {u_max_val:.2f}$ | $N_{{QS}}={n_steps_qs}$ | $N_{{Dyn}}={n_steps_dyn}$ | $N_{{Mesh}}={mesh_val}$ | $T_0={smoth_val}$"
        )
        fig.suptitle(header_text, fontsize=13, fontweight="bold", y=0.97)

        # --- MODIFIED: Force Plot (QS: dots, Dyn: crosses with faint lines) ---
        ax_force.plot(qs["U"],  qs["F"],  color="black", marker=".", linestyle="-", linewidth=0.6, alpha=0.7, label="QS")
        ax_force.plot(dyn["U"], dyn["F"], color="red", marker="x", linestyle="-", linewidth=0.6, alpha=0.7, label=fr"dynamic, $\eta={eta_val}$")
        
        ax_force.set_xlabel(r"$\hat{U}(t)$")
        ax_force.set_ylabel(r"reaction $\hat{F}$")
        ax_force.set_title("Force-displacement")
        ax_force.grid(True, alpha=0.3)
        ax_force.legend()

        ax_dam.plot(x_alpha[ix_alpha], alpha_qs_final[ix_alpha],  "k-",  label="QS")
        ax_dam.plot(x_alpha[ix_alpha], alpha_dyn_final[ix_alpha], "r--", label="Dynamic")
        ax_dam.set_xlabel(r"$\hat{x}$"); ax_dam.set_ylabel(r"$\alpha$")
        ax_dam.set_title(r"Final damage at $\hat{U}=\hat{U}_{\rm max}$")
        ax_dam.grid(True, alpha=0.3); ax_dam.legend()

        # --- MODIFIED: Energy Plot (QS: dots, Dyn: crosses with faint lines) ---
        ax_energy.plot(dyn["U"], dyn["K"],     color="m", marker="x", linestyle="-", linewidth=0.6, alpha=0.7, label=r"$\hat{K}$ (dyn only)")
        
        ax_energy.plot(qs["U"],  qs["P_el"],   color="b", marker=".", linestyle="-", linewidth=0.6, alpha=0.7, label=r"$\hat{P}_{el}$ QS")
        ax_energy.plot(dyn["U"], dyn["P_el"],  color="b", marker="x", linestyle="-", linewidth=0.6, alpha=0.7, label=r"$\hat{P}_{el}$ Dyn")
        
        ax_energy.plot(qs["U"],  qs["P_f"],    color="g", marker=".", linestyle="-", linewidth=0.6, alpha=0.7, label=r"$\hat{P}_f$ QS")
        ax_energy.plot(dyn["U"], dyn["P_f"],   color="g", marker="x", linestyle="-", linewidth=0.6, alpha=0.7, label=r"$\hat{P}_f$ Dyn")
        
        ax_energy.plot(qs["U"],  qs["S"],      color="r", marker=".", linestyle="-", linewidth=0.6, alpha=0.7, label=r"$\hat{S}$ QS")
        ax_energy.plot(dyn["U"], dyn["S"],     color="r", marker="x", linestyle="-", linewidth=0.6, alpha=0.7, label=r"$\hat{S}$ Dyn")
        
        ax_energy.plot(qs["U"],  qs["total"],  color="k", marker=".", linestyle="-", linewidth=0.6, alpha=0.7, label="Total QS")
        ax_energy.plot(dyn["U"], dyn["total"], color="k", marker="x", linestyle="-", linewidth=0.6, alpha=0.7, label="Total Dyn (incl K)")
        
        ax_energy.set_xlabel(r"$\hat{U}(t)$")
        ax_energy.set_ylabel("Energy")
        ax_energy.set_title("Energy evolution")
        ax_energy.grid(True, alpha=0.3)
        ax_energy.legend(fontsize=9, ncol=2, loc="best")

        err_text = (
            f"Inf-norm gaps (max |QS - Dyn|):  "
            f"$\\Delta P_{{el}} = {err_P_el:.3e}$  |  "
            f"$\\Delta P_f = {err_P_f:.3e}$  |  "
            f"$\\Delta S = {err_S:.3e}$  |  "
            f"$\\Delta\\,\\mathrm{{Total}} = {err_tot:.3e}$"
        )
        fig.text(0.5, 0.015, err_text, ha="center", va="bottom", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="aliceblue", edgecolor="steelblue", alpha=0.8))

        plt.tight_layout(rect=[0, 0.05, 1, 0.93])

        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "Output")
        
        png_dir = os.path.join(output_dir, "png")
        pdf_dir = os.path.join(output_dir, "pdf")
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(pdf_dir, exist_ok=True)

        filename_str = (
            f"mechanical_lhat_{l_hat_val}_lam_{lambda_val}_eta_{eta_val}_"
            f"umax_{u_max_val:.2f}_nQS_{n_steps_qs}_nDyn_{n_steps_dyn}"
            f"_nMesh_{mesh_val}_T0_{smoth_val}"
        )

        png_path = os.path.join(png_dir, f"{filename_str}.png")
        pdf_path = os.path.join(pdf_dir, f"{filename_str}.pdf")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.savefig(pdf_path, bbox_inches="tight")

        # Crucial for parameter sweeping: close the figure to prevent execution halting
        plt.close(fig) 
        
        print(f"Results saved to {output_dir}/")
        print(f"  - PNG: {png_path}")
        print(f"  - PDF: {pdf_path}")

    # Clean up PETSc solvers to free memory during the sweep
    solver_u_qs.destroy()
    solver_alpha_qs.destroy()
    solver_acc.destroy()
    solver_alpha_dyn.destroy()


if __name__ == "__main__":
    # Define your sweep grids (3 values each as requested)
    l_hat_grid  = [0.01, 0.02, 0.04, 0.1]
    Lambda_grid = [1.0, 10.0, 20.0, 50.0]
    eta_grid    = [1e-3, 1e-2, 5e-2, 1e-1]

    # Generate all combinations
    parameter_combinations = list(itertools.product(l_hat_grid, Lambda_grid, eta_grid))

    if comm.rank == 0:
        print(f"Initiating parameter sweep: {len(parameter_combinations)} total combinations to run.")

    # Execute the sweep
    for l_hat_val, lambda_val, eta_val in parameter_combinations:
        run_simulation(l_hat_val, lambda_val, eta_val)

    if comm.rank == 0:
        print("\n--- Parameter Sweep Complete! ---")