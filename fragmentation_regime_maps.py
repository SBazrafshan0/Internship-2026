"""
=============================================================================
 fragmentation_regime_maps.py
=============================================================================
Gradient-damage (phase-field) fragmentation of a 1D bar on an elastic
foundation, solved both quasi-statically (Alternate Minimization) and
dynamically (Newmark-beta), for two loading types:

    * MECHANICAL  : imposed end displacement U(t)        -> tensile fracture
    * THERMAL     : imposed uniform thermal strain th(t) -> periodic fragmentation
=============================================================================
"""

import os
import json
import itertools
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch
from matplotlib.lines import Line2D

# -----------------------------------------------------------------------------
# FEM stack is optional at import time so the post-processing can run anywhere.
# -----------------------------------------------------------------------------
try:
    import sympy as sp
    import dolfinx
    from dolfinx import mesh as df_mesh, fem
    import dolfinx.fem.petsc
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc

    comm = MPI.COMM_WORLD
    HAVE_FENICS = True
except Exception as _exc:           # pragma: no cover
    HAVE_FENICS = False
    _FENICS_IMPORT_ERROR = _exc

    class _DummyComm:
        rank = 0
        size = 1
        def gather(self, *args, **kwargs): pass
    comm = _DummyComm()


# =============================================================================
#  SHARED HELPER CLASS
# =============================================================================
if HAVE_FENICS:

    class SNESProblem:
        def __init__(self, F, u, bcs, J=None):
            V = u.function_space
            du = ufl.TrialFunction(V)
            self.L = fem.form(F)
            self.a = fem.form(J) if J is not None else fem.form(ufl.derivative(F, u, du))
            self.bcs = bcs
            self.u = u

        def F(self, snes, x, F):
            x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            x.copy(self.u.x.petsc_vec)
            self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
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


# =============================================================================
#  MECHANICAL SIMULATION
# =============================================================================
def run_mechanical_simulation(model_parameters, mesh_parameters, loading_parameters,
                              AltMin_parameters, Newmark_parameters):

    l_hat_val  = model_parameters["l_hat"]
    lambda_val = model_parameters["Lambda"]
    eta_val    = model_parameters["eta"]

    n_steps_qs  = loading_parameters["N_steps_qs"]
    n_steps_dyn = loading_parameters["N_steps_dyn"]

    # EDIT: Use MPI.COMM_SELF so each core runs its own independent 1D mesh
    domain = df_mesh.create_interval(MPI.COMM_SELF, mesh_parameters["nx"], (0.0, 1.0))
    gdim, fdim = domain.topology.dim, domain.topology.dim - 1

    V_u     = fem.functionspace(domain, ("Lagrange", 1))
    V_alpha = fem.functionspace(domain, ("Lagrange", 1))

    u, v, a       = (fem.Function(V_u, name=n) for n in ("Displacement", "Velocity", "Acceleration"))
    u_new, v_new, a_new = fem.Function(V_u), fem.Function(V_u), fem.Function(V_u)

    alpha          = fem.Function(V_alpha, name="Damage")
    alpha_old_iter = fem.Function(V_alpha)
    alpha_lb       = fem.Function(V_alpha)
    alpha_ub       = fem.Function(V_alpha)

    def left_marker(x):  return np.isclose(x[0], 0.0)
    def right_marker(x): return np.isclose(x[0], 1.0)

    left_facets  = df_mesh.locate_entities_boundary(domain, fdim, left_marker)
    right_facets = df_mesh.locate_entities_boundary(domain, fdim, right_marker)

    all_facets = np.concatenate([left_facets, right_facets]).astype(np.int32)
    markers    = np.concatenate([np.full(len(left_facets), 1, dtype=np.int32),
                                 np.full(len(right_facets), 2, dtype=np.int32)])
    sort_ix = np.argsort(all_facets)
    mt = df_mesh.meshtags(domain, fdim, all_facets[sort_ix], markers[sort_ix])

    dx = ufl.Measure("dx", domain=domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)

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

    bcs_u = [fem.dirichletbc(u_left_val, left_dofs_u, V_u),
             fem.dirichletbc(u_right_val, right_dofs_u, V_u)]
    bcs_v = [fem.dirichletbc(v_left_val, left_dofs_u, V_u),
             fem.dirichletbc(v_right_val, right_dofs_u, V_u)]
    bcs_a = [fem.dirichletbc(a_left_val, left_dofs_u, V_u),
             fem.dirichletbc(a_right_val, right_dofs_u, V_u)]
    bcs_alpha = [fem.dirichletbc(PETSc.ScalarType(0.0), left_dofs_alpha,  V_alpha),
                 fem.dirichletbc(PETSc.ScalarType(0.0), right_dofs_alpha, V_alpha)]

    Lambda_c  = fem.Constant(domain, PETSc.ScalarType(lambda_val))
    l_hat_c   = fem.Constant(domain, PETSc.ScalarType(l_hat_val))
    eta_c     = fem.Constant(domain, PETSc.ScalarType(eta_val))
    delta_t_c = fem.Constant(domain, PETSc.ScalarType(1.0 / n_steps_dyn))

    def strain(w):         return w.dx(0)
    def stress(w, alpha_): return (1.0 - alpha_)**2 * strain(w)

    strain_energy     = 0.5 * (1.0 - alpha)**2 * strain(u)**2 * dx
    foundation_energy = 0.5 * Lambda_c**2 * u**2 * dx
    potential_energy  = strain_energy + foundation_energy
    fracture_energy   = (alpha + l_hat_c**2 * alpha.dx(0)**2) * dx
    kinetic_energy    = 0.5 * eta_c**2 * v**2 * dx

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
    Res_acc      = eta_c**2 * a_new * u_test * dx + ufl.derivative(potential_energy, u, u_test)

    t_sp   = sp.Symbol("t", real=True)
    T0_v   = loading_parameters["T0"]
    Umax_v = loading_parameters["U_max"]
    Udot_p   = Umax_v / (np.sqrt(T0_v**2 + 1.0) - T0_v)
    U_imp_sp = Udot_p * (sp.sqrt(T0_v**2 + t_sp**2) - T0_v)
    V_imp_sp = sp.diff(U_imp_sp, t_sp)
    A_imp_sp = sp.diff(V_imp_sp, t_sp)
    U_fn = sp.lambdify(t_sp, U_imp_sp, "numpy")
    V_fn = sp.lambdify(t_sp, V_imp_sp, "numpy")
    A_fn = sp.lambdify(t_sp, A_imp_sp, "numpy")

    J_u_qs     = ufl.derivative(Res_u_qs,     u,     ufl.TrialFunction(V_u))
    J_alpha_qs = ufl.derivative(Res_alpha_qs, alpha, ufl.TrialFunction(V_alpha))

    elastic_problem_qs = SNESProblem(Res_u_qs,     u,     bcs_u,     J=J_u_qs)
    damage_problem_qs  = SNESProblem(Res_alpha_qs, alpha, bcs_alpha, J=J_alpha_qs)

    b_u_qs     = fem.petsc.create_vector(V_u)
    J_u_qs_m   = fem.petsc.create_matrix(elastic_problem_qs.a)
    b_alpha_qs = fem.petsc.create_vector(V_alpha)
    J_alpha_qs_m = fem.petsc.create_matrix(damage_problem_qs.a)

    solver_u_qs = PETSc.SNES().create()
    solver_u_qs.setType("ksponly")
    solver_u_qs.setFunction(elastic_problem_qs.F, b_u_qs)
    solver_u_qs.setJacobian(elastic_problem_qs.J, J_u_qs_m)
    solver_u_qs.setTolerances(rtol=1.0e-9, max_it=50)
    solver_u_qs.getKSP().setType("preonly")
    solver_u_qs.getKSP().getPC().setType("lu")
    # EDIT: Removed MUMPS to use default lightweight serial LU
    # solver_u_qs.getKSP().getPC().setFactorSolverType("mumps") 

    solver_alpha_qs = PETSc.SNES().create()
    solver_alpha_qs.setType("vinewtonrsls")
    solver_alpha_qs.setFunction(damage_problem_qs.F, b_alpha_qs)
    solver_alpha_qs.setJacobian(damage_problem_qs.J, J_alpha_qs_m)
    solver_alpha_qs.setTolerances(rtol=1.0e-9, max_it=50)
    solver_alpha_qs.getKSP().setType("preonly")
    solver_alpha_qs.getKSP().getPC().setType("lu")
    solver_alpha_qs.setVariableBounds(alpha_lb.x.petsc_vec, alpha_ub.x.petsc_vec)

    # ---- quasi-static loop ----
    u.x.array[:] = 0.0; v.x.array[:] = 0.0; a.x.array[:] = 0.0
    alpha.x.array[:] = 0.0; alpha_lb.x.array[:] = 0.0; alpha_ub.x.array[:] = 1.0

    qs = {"U": [], "F": [], "P_el": [], "P_f": [], "S": [], "total": []}
    t_grid_qs = np.linspace(0.0, 1.0, n_steps_qs + 1)[1:]

    for ti in t_grid_qs:
        u_right_val.value = float(U_fn(ti))
        for _ in range(1, AltMin_parameters["max_iter"] + 1):
            solver_u_qs.solve(None, u.x.petsc_vec)
            u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            solver_alpha_qs.solve(None, alpha.x.petsc_vec)
            alpha.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            # EDIT: Changed comm to domain.comm
            err = domain.comm.allreduce(fem.assemble_scalar(error_L2_alpha_form), op=MPI.SUM)
            alpha_old_iter.x.array[:] = alpha.x.array
            if float(np.sqrt(max(err, 0.0))) <= AltMin_parameters["tol"]:
                break
        alpha_lb.x.array[:] = alpha.x.array
        qs["U"].append(float(u_right_val.value))
        # EDIT: Changed comm to domain.comm
        qs["F"].append(domain.comm.allreduce(fem.assemble_scalar(reaction_right_form),     op=MPI.SUM))
        qs["P_el"].append(domain.comm.allreduce(fem.assemble_scalar(strain_energy_form),     op=MPI.SUM))
        qs["P_f"].append(domain.comm.allreduce(fem.assemble_scalar(foundation_energy_form), op=MPI.SUM))
        qs["S"].append(domain.comm.allreduce(fem.assemble_scalar(fracture_energy_form),     op=MPI.SUM))
        qs["total"].append(qs["P_el"][-1] + qs["P_f"][-1] + qs["S"][-1])

    alpha_qs_final = alpha.x.array.copy()
    for k in qs:
        qs[k] = np.asarray(qs[k])

    # ---- dynamic (Newmark) loop ----
    beta_v, gamma_v = Newmark_parameters["beta"], Newmark_parameters["gamma"]

    def u_newmark(u_, v_, a_, a_new_, dt):
        return u_ + dt*v_ + 0.5*dt**2 * ((1.0 - 2.0*beta_v)*a_ + 2.0*beta_v*a_new_)

    Res_acc_newmark = ufl.replace(Res_acc, {u: u_newmark(u, v, a, a_new, delta_t_c)})
    Res_alpha_dyn   = ufl.replace(Res_alpha_qs, {u: u_newmark(u, v, a, a_new, delta_t_c)})

    J_acc_newmark = ufl.derivative(Res_acc_newmark, a_new, ufl.TrialFunction(V_u))
    J_alpha_dyn   = ufl.derivative(Res_alpha_dyn,   alpha, ufl.TrialFunction(V_alpha))

    acc_problem        = SNESProblem(Res_acc_newmark, a_new, bcs_a,     J=J_acc_newmark)
    damage_problem_dyn = SNESProblem(Res_alpha_dyn,   alpha, bcs_alpha, J=J_alpha_dyn)

    b_acc_dyn   = fem.petsc.create_vector(V_u)
    J_acc_dyn_m = fem.petsc.create_matrix(acc_problem.a)
    b_alpha_dyn = fem.petsc.create_vector(V_alpha)
    J_alpha_dyn_m = fem.petsc.create_matrix(damage_problem_dyn.a)

    solver_acc = PETSc.SNES().create()
    solver_acc.setType("ksponly")
    solver_acc.setFunction(acc_problem.F, b_acc_dyn)
    solver_acc.setJacobian(acc_problem.J, J_acc_dyn_m)
    solver_acc.setTolerances(rtol=1.0e-9, max_it=50)
    solver_acc.getKSP().setType("preonly")
    solver_acc.getKSP().getPC().setType("lu")

    solver_alpha_dyn = PETSc.SNES().create()
    solver_alpha_dyn.setType("vinewtonrsls")
    solver_alpha_dyn.setFunction(damage_problem_dyn.F, b_alpha_dyn)
    solver_alpha_dyn.setJacobian(damage_problem_dyn.J, J_alpha_dyn_m)
    solver_alpha_dyn.setTolerances(rtol=1.0e-9, max_it=50)
    solver_alpha_dyn.getKSP().setType("preonly")
    solver_alpha_dyn.getKSP().getPC().setType("lu")
    solver_alpha_dyn.setVariableBounds(alpha_lb.x.petsc_vec, alpha_ub.x.petsc_vec)

    delta_t_c.value = 1.0 / n_steps_dyn

    u.x.array[:] = 0.0; u_new.x.array[:] = 0.0
    v.x.array[:] = 0.0; v_new.x.array[:] = 0.0
    a.x.array[:] = 0.0; a_new.x.array[:] = 0.0
    alpha.x.array[:] = 0.0; alpha_lb.x.array[:] = 0.0; alpha_ub.x.array[:] = 1.0

    dyn = {"U": [], "F": [], "K": [], "P_el": [], "P_f": [], "S": [], "total": []}
    dt = 1.0 / n_steps_dyn
    t_cur = 0.0
    for _ in range(n_steps_dyn):
        t_cur += dt
        u_right_val.value = float(U_fn(t_cur))
        v_right_val.value = float(V_fn(t_cur))
        a_right_val.value = float(A_fn(t_cur))
        for _ in range(1, AltMin_parameters["max_iter"] + 1):
            solver_acc.solve(None, a_new.x.petsc_vec)
            a_new.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            solver_alpha_dyn.solve(None, alpha.x.petsc_vec)
            alpha.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            # EDIT: Changed comm to domain.comm
            err = domain.comm.allreduce(fem.assemble_scalar(error_L2_alpha_form), op=MPI.SUM)
            alpha_old_iter.x.array[:] = alpha.x.array
            if float(np.sqrt(max(err, 0.0))) <= AltMin_parameters["tol"]:
                break
        u_new.x.array[:] = (u.x.array + dt*v.x.array
                            + 0.5*dt**2 * ((1.0 - 2.0*beta_v)*a.x.array + 2.0*beta_v*a_new.x.array))
        v_new.x.array[:] = v.x.array + dt * ((1.0 - gamma_v)*a.x.array + gamma_v*a_new.x.array)
        fem.set_bc(u_new.x.petsc_vec, bcs_u)
        fem.set_bc(v_new.x.petsc_vec, bcs_v)
        u_new.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        v_new.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        u.x.array[:] = u_new.x.array
        v.x.array[:] = v_new.x.array
        a.x.array[:] = a_new.x.array
        alpha_lb.x.array[:] = alpha.x.array

        dyn["U"].append(float(u_right_val.value))
        # EDIT: Changed comm to domain.comm
        dyn["F"].append(domain.comm.allreduce(fem.assemble_scalar(reaction_right_form), op=MPI.SUM))
        dyn["K"].append(domain.comm.allreduce(fem.assemble_scalar(kinetic_energy_form), op=MPI.SUM))
        dyn["P_el"].append(domain.comm.allreduce(fem.assemble_scalar(strain_energy_form),     op=MPI.SUM))
        dyn["P_f"].append(domain.comm.allreduce(fem.assemble_scalar(foundation_energy_form), op=MPI.SUM))
        dyn["S"].append(domain.comm.allreduce(fem.assemble_scalar(fracture_energy_form),     op=MPI.SUM))
        dyn["total"].append(dyn["K"][-1] + dyn["P_el"][-1] + dyn["P_f"][-1] + dyn["S"][-1])

    alpha_dyn_final = alpha.x.array.copy()
    for k in dyn:
        dyn[k] = np.asarray(dyn[k])

    x_alpha = V_alpha.tabulate_dof_coordinates()[:, 0]
    ix = np.argsort(x_alpha)

    # QS-vs-Dyn inf-norm gaps (interp dyn onto qs abscissa)
    P_el_d = np.interp(qs["U"], dyn["U"], dyn["P_el"])
    P_f_d  = np.interp(qs["U"], dyn["U"], dyn["P_f"])
    S_d    = np.interp(qs["U"], dyn["U"], dyn["S"])
    tot_d  = np.interp(qs["U"], dyn["U"], dyn["total"])
    gaps = dict(P_el=float(np.max(np.abs(qs["P_el"]-P_el_d))),
                P_f=float(np.max(np.abs(qs["P_f"]-P_f_d))),
                S=float(np.max(np.abs(qs["S"]-S_d))),
                total=float(np.max(np.abs(qs["total"]-tot_d))))

    for s in (solver_u_qs, solver_alpha_qs, solver_acc, solver_alpha_dyn):
        s.destroy()

    return {
        "physics": "mechanical",
        "model": dict(model_parameters),
        "x": x_alpha[ix],
        "alpha_qs": alpha_qs_final[ix],
        "alpha_dyn": alpha_dyn_final[ix],
        "load_qs": qs["U"], "load_dyn": dyn["U"],
        "resp_qs": qs["F"], "resp_dyn": dyn["F"],
        "S_qs": qs["S"], "S_dyn": dyn["S"],
        "total_qs": qs["total"], "total_dyn": dyn["total"],
        "K_dyn": dyn["K"],
        "gaps": gaps,
    }


# =============================================================================
#  THERMAL SIMULATION
# =============================================================================
def run_thermal_simulation(model_parameters, mesh_parameters, loading_parameters,
                           AltMin_parameters, Newmark_parameters):

    l_hat_val  = model_parameters["l_hat"]
    lambda_val = model_parameters["Lambda"]
    eta_val    = model_parameters["eta"]
    n_steps_qs  = loading_parameters["N_steps_qs"]
    n_steps_dyn = loading_parameters["N_steps_dyn"]

    # EDIT: Use MPI.COMM_SELF
    domain = df_mesh.create_interval(MPI.COMM_SELF, mesh_parameters["nx"], (0.0, 1.0))

    V_u     = fem.functionspace(domain, ("Lagrange", 1))
    V_alpha = fem.functionspace(domain, ("Lagrange", 1))

    u, v, a       = (fem.Function(V_u, name=n) for n in ("Displacement", "Velocity", "Acceleration"))
    u_new, v_new, a_new = fem.Function(V_u), fem.Function(V_u), fem.Function(V_u)

    alpha          = fem.Function(V_alpha, name="Damage")
    alpha_old_iter = fem.Function(V_alpha)
    alpha_lb       = fem.Function(V_alpha)
    alpha_ub       = fem.Function(V_alpha)

    dx = ufl.Measure("dx", domain=domain)
    bcs_u, bcs_v, bcs_a, bcs_alpha = [], [], [], []

    Lambda_c  = fem.Constant(domain, PETSc.ScalarType(lambda_val))
    l_hat_c   = fem.Constant(domain, PETSc.ScalarType(l_hat_val))
    eta_c     = fem.Constant(domain, PETSc.ScalarType(eta_val))
    delta_t_c = fem.Constant(domain, PETSc.ScalarType(1.0 / n_steps_dyn))
    theta_c   = fem.Constant(domain, PETSc.ScalarType(0.0))

    def elastic_strain(w, th): return w.dx(0) - th
    def stress(w, alpha_, th): return (1.0 - alpha_)**2 * elastic_strain(w, th)

    strain_energy     = 0.5 * (1.0 - alpha)**2 * elastic_strain(u, theta_c)**2 * dx
    foundation_energy = 0.5 * Lambda_c**2 * u**2 * dx
    potential_energy  = strain_energy + foundation_energy
    fracture_energy   = (alpha + l_hat_c**2 * alpha.dx(0)**2) * dx
    kinetic_energy    = 0.5 * eta_c**2 * v**2 * dx

    strain_energy_form     = fem.form(strain_energy)
    foundation_energy_form = fem.form(foundation_energy)
    fracture_energy_form   = fem.form(fracture_energy)
    kinetic_energy_form    = fem.form(kinetic_energy)
    mean_stress_form       = fem.form(stress(u, alpha, theta_c) * dx)
    error_L2_alpha_form    = fem.form((alpha - alpha_old_iter)**2 * dx)

    u_test     = ufl.TestFunction(V_u)
    alpha_test = ufl.TestFunction(V_alpha)

    Res_u_qs     = ufl.derivative(potential_energy, u, u_test)
    Res_alpha_qs = ufl.derivative(potential_energy + fracture_energy, alpha, alpha_test)
    Res_acc      = eta_c**2 * a_new * u_test * dx + ufl.derivative(potential_energy, u, u_test)

    t_sp    = sp.Symbol("t", real=True)
    T0_v    = loading_parameters["T0"]
    ThMax_v = loading_parameters["theta_max"]
    Thdot_p  = ThMax_v / (np.sqrt(T0_v**2 + 1.0) - T0_v)
    Theta_sp = Thdot_p * (sp.sqrt(T0_v**2 + t_sp**2) - T0_v)
    Theta_fn = sp.lambdify(t_sp, Theta_sp, "numpy")

    J_u_qs     = ufl.derivative(Res_u_qs,     u,     ufl.TrialFunction(V_u))
    J_alpha_qs = ufl.derivative(Res_alpha_qs, alpha, ufl.TrialFunction(V_alpha))

    elastic_problem_qs = SNESProblem(Res_u_qs,     u,     bcs_u,     J=J_u_qs)
    damage_problem_qs  = SNESProblem(Res_alpha_qs, alpha, bcs_alpha, J=J_alpha_qs)

    b_u_qs     = fem.petsc.create_vector(V_u)
    J_u_qs_m   = fem.petsc.create_matrix(elastic_problem_qs.a)
    b_alpha_qs = fem.petsc.create_vector(V_alpha)
    J_alpha_qs_m = fem.petsc.create_matrix(damage_problem_qs.a)

    solver_u_qs = PETSc.SNES().create()
    solver_u_qs.setType("ksponly")
    solver_u_qs.setFunction(elastic_problem_qs.F, b_u_qs)
    solver_u_qs.setJacobian(elastic_problem_qs.J, J_u_qs_m)
    solver_u_qs.setTolerances(rtol=1.0e-9, max_it=50)
    solver_u_qs.getKSP().setType("preonly")
    solver_u_qs.getKSP().getPC().setType("lu")

    solver_alpha_qs = PETSc.SNES().create()
    solver_alpha_qs.setType("vinewtonrsls")
    solver_alpha_qs.setFunction(damage_problem_qs.F, b_alpha_qs)
    solver_alpha_qs.setJacobian(damage_problem_qs.J, J_alpha_qs_m)
    solver_alpha_qs.setTolerances(rtol=1.0e-9, max_it=50)
    solver_alpha_qs.getKSP().setType("preonly")
    solver_alpha_qs.getKSP().getPC().setType("lu")
    solver_alpha_qs.setVariableBounds(alpha_lb.x.petsc_vec, alpha_ub.x.petsc_vec)

    beta_v, gamma_v = Newmark_parameters["beta"], Newmark_parameters["gamma"]

    def u_newmark(u_, v_, a_, a_new_, dt):
        return u_ + dt*v_ + 0.5*dt**2 * ((1.0 - 2.0*beta_v)*a_ + 2.0*beta_v*a_new_)

    Res_acc_newmark = ufl.replace(Res_acc, {u: u_newmark(u, v, a, a_new, delta_t_c)})
    Res_alpha_dyn   = ufl.replace(Res_alpha_qs, {u: u_newmark(u, v, a, a_new, delta_t_c)})

    J_acc_newmark = ufl.derivative(Res_acc_newmark, a_new, ufl.TrialFunction(V_u))
    J_alpha_dyn   = ufl.derivative(Res_alpha_dyn,   alpha, ufl.TrialFunction(V_alpha))

    acc_problem        = SNESProblem(Res_acc_newmark, a_new, bcs_a,     J=J_acc_newmark)
    damage_problem_dyn = SNESProblem(Res_alpha_dyn,   alpha, bcs_alpha, J=J_alpha_dyn)

    b_acc_dyn   = fem.petsc.create_vector(V_u)
    J_acc_dyn_m = fem.petsc.create_matrix(acc_problem.a)
    b_alpha_dyn = fem.petsc.create_vector(V_alpha)
    J_alpha_dyn_m = fem.petsc.create_matrix(damage_problem_dyn.a)

    solver_acc = PETSc.SNES().create()
    solver_acc.setType("ksponly")
    solver_acc.setFunction(acc_problem.F, b_acc_dyn)
    solver_acc.setJacobian(acc_problem.J, J_acc_dyn_m)
    solver_acc.setTolerances(rtol=1.0e-9, max_it=50)
    solver_acc.getKSP().setType("preonly")
    solver_acc.getKSP().getPC().setType("lu")

    solver_alpha_dyn = PETSc.SNES().create()
    solver_alpha_dyn.setType("vinewtonrsls")
    solver_alpha_dyn.setFunction(damage_problem_dyn.F, b_alpha_dyn)
    solver_alpha_dyn.setJacobian(damage_problem_dyn.J, J_alpha_dyn_m)
    solver_alpha_dyn.setTolerances(rtol=1.0e-9, max_it=50)
    solver_alpha_dyn.getKSP().setType("preonly")
    solver_alpha_dyn.getKSP().getPC().setType("lu")
    solver_alpha_dyn.setVariableBounds(alpha_lb.x.petsc_vec, alpha_ub.x.petsc_vec)

    delta_t_c.value = 1.0 / n_steps_dyn

    # ---- quasi-static loop ----
    u.x.array[:] = 0.0; v.x.array[:] = 0.0; a.x.array[:] = 0.0
    alpha.x.array[:] = 0.0; alpha_lb.x.array[:] = 0.0; alpha_ub.x.array[:] = 1.0
    theta_c.value = 0.0

    qs = {"theta": [], "sigma": [], "S": [], "total": []}
    t_grid_qs = np.linspace(0.0, 1.0, n_steps_qs + 1)[1:]
    for ti in t_grid_qs:
        theta_c.value = float(Theta_fn(ti))
        for _ in range(1, AltMin_parameters["max_iter"] + 1):
            solver_u_qs.solve(None, u.x.petsc_vec)
            u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            solver_alpha_qs.solve(None, alpha.x.petsc_vec)
            alpha.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            # EDIT: Changed comm to domain.comm
            err = domain.comm.allreduce(fem.assemble_scalar(error_L2_alpha_form), op=MPI.SUM)
            alpha_old_iter.x.array[:] = alpha.x.array
            if float(np.sqrt(max(err, 0.0))) <= AltMin_parameters["tol"]:
                break
        alpha_lb.x.array[:] = alpha.x.array
        qs["theta"].append(float(theta_c.value))
        # EDIT: Changed comm to domain.comm
        qs["sigma"].append(domain.comm.allreduce(fem.assemble_scalar(mean_stress_form),    op=MPI.SUM))
        P_el = domain.comm.allreduce(fem.assemble_scalar(strain_energy_form),     op=MPI.SUM)
        P_f  = domain.comm.allreduce(fem.assemble_scalar(foundation_energy_form), op=MPI.SUM)
        S    = domain.comm.allreduce(fem.assemble_scalar(fracture_energy_form),   op=MPI.SUM)
        qs["S"].append(S); qs["total"].append(P_el + P_f + S)

    alpha_qs_final = alpha.x.array.copy()
    for k in qs:
        qs[k] = np.asarray(qs[k])

    # ---- dynamic loop ----
    u.x.array[:] = 0.0; u_new.x.array[:] = 0.0
    v.x.array[:] = 0.0; v_new.x.array[:] = 0.0
    a.x.array[:] = 0.0; a_new.x.array[:] = 0.0
    alpha.x.array[:] = 0.0; alpha_lb.x.array[:] = 0.0; alpha_ub.x.array[:] = 1.0
    theta_c.value = 0.0

    dyn = {"theta": [], "sigma": [], "K": [], "S": [], "total": []}
    dt = 1.0 / n_steps_dyn
    t_cur = 0.0
    for _ in range(n_steps_dyn):
        t_cur += dt
        theta_c.value = float(Theta_fn(t_cur))
        for _ in range(1, AltMin_parameters["max_iter"] + 1):
            solver_acc.solve(None, a_new.x.petsc_vec)
            a_new.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            solver_alpha_dyn.solve(None, alpha.x.petsc_vec)
            alpha.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            # EDIT: Changed comm to domain.comm
            err = domain.comm.allreduce(fem.assemble_scalar(error_L2_alpha_form), op=MPI.SUM)
            alpha_old_iter.x.array[:] = alpha.x.array
            if float(np.sqrt(max(err, 0.0))) <= AltMin_parameters["tol"]:
                break
        u_new.x.array[:] = (u.x.array + dt*v.x.array
                            + 0.5*dt**2 * ((1.0 - 2.0*beta_v)*a.x.array + 2.0*beta_v*a_new.x.array))
        v_new.x.array[:] = v.x.array + dt * ((1.0 - gamma_v)*a.x.array + gamma_v*a_new.x.array)
        u_new.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        v_new.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        u.x.array[:] = u_new.x.array
        v.x.array[:] = v_new.x.array
        a.x.array[:] = a_new.x.array
        alpha_lb.x.array[:] = alpha.x.array

        dyn["theta"].append(float(theta_c.value))
        # EDIT: Changed comm to domain.comm
        dyn["sigma"].append(domain.comm.allreduce(fem.assemble_scalar(mean_stress_form), op=MPI.SUM))
        K    = domain.comm.allreduce(fem.assemble_scalar(kinetic_energy_form),    op=MPI.SUM)
        P_el = domain.comm.allreduce(fem.assemble_scalar(strain_energy_form),     op=MPI.SUM)
        P_f  = domain.comm.allreduce(fem.assemble_scalar(foundation_energy_form), op=MPI.SUM)
        S    = domain.comm.allreduce(fem.assemble_scalar(fracture_energy_form),   op=MPI.SUM)
        dyn["K"].append(K); dyn["S"].append(S); dyn["total"].append(K + P_el + P_f + S)

    alpha_dyn_final = alpha.x.array.copy()
    for k in dyn:
        dyn[k] = np.asarray(dyn[k])

    x_alpha = V_alpha.tabulate_dof_coordinates()[:, 0]
    ix = np.argsort(x_alpha)

    S_d   = np.interp(qs["theta"], dyn["theta"], dyn["S"])
    tot_d = np.interp(qs["theta"], dyn["theta"], dyn["total"])
    gaps = dict(S=float(np.max(np.abs(qs["S"]-S_d))),
                total=float(np.max(np.abs(qs["total"]-tot_d))))

    for s in (solver_u_qs, solver_alpha_qs, solver_acc, solver_alpha_dyn):
        s.destroy()

    return {
        "physics": "thermal",
        "model": dict(model_parameters),
        "x": x_alpha[ix],
        "alpha_qs": alpha_qs_final[ix],
        "alpha_dyn": alpha_dyn_final[ix],
        "load_qs": qs["theta"], "load_dyn": dyn["theta"],
        "resp_qs": qs["sigma"], "resp_dyn": dyn["sigma"],
        "S_qs": qs["S"], "S_dyn": dyn["S"],
        "total_qs": qs["total"], "total_dyn": dyn["total"],
        "K_dyn": dyn["K"],
        "gaps": gaps,
    }


# =============================================================================
#  DAMAGE-PROFILE ANALYSIS  &  REGIME CLASSIFICATION
# =============================================================================
R_NONE  = 0
R_ONE   = 1
R_MULTI = 2
R_DIFF  = 3

REGIME_NAMES = {
    R_NONE:  "No fracture\n(sub-critical)",
    R_ONE:   "Single crack",
    R_MULTI: "Fragmentation\n(multiple cracks)",
    R_DIFF:  "Diffuse / homogeneous\ndamage",
}
REGIME_COLORS = {
    R_NONE:  "#dfe7ef",
    R_ONE:   "#9ecae1",
    R_MULTI: "#fb8d3c",
    R_DIFF:  "#7d3c98",
}

ALPHA_CRACK = 0.50
ALPHA_CORE  = 0.90
DIFFUSE_FRAC = 0.55

def _count_bands(mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    return int(np.sum((mask[1:].astype(int) - mask[:-1].astype(int)) == 1) + (1 if mask[0] else 0))

def analyze_profile(x, alpha):
    x = np.asarray(x, float)
    alpha = np.asarray(alpha, float)
    L = x[-1] - x[0]
    amax = float(alpha.max())
    broken = alpha > ALPHA_CRACK
    core   = alpha > ALPHA_CORE
    damaged_fraction = float(np.trapz(broken.astype(float), x) / L)
    n_core = _count_bands(core)
    n_broken_bands = _count_bands(broken)
    n_cracks = n_core if n_core > 0 else n_broken_bands
    mean_band_width = (damaged_fraction * L / n_broken_bands) if n_broken_bands else 0.0
    return dict(alpha_max=amax, damaged_fraction=damaged_fraction,
                n_core=n_core, n_broken=n_broken_bands,
                n_cracks=n_cracks, mean_band_width=mean_band_width)

def classify_regime(prof):
    if prof["alpha_max"] < ALPHA_CRACK:
        return R_NONE
    if prof["damaged_fraction"] >= DIFFUSE_FRAC:
        return R_DIFF
    if prof["n_cracks"] >= 2:
        return R_MULTI
    return R_ONE

def energy_is_monotone(total, rel_tol=1e-3):
    total = np.asarray(total, float)
    if total.size < 2:
        return True
    drops = np.diff(total)
    scale = max(np.max(np.abs(total)), 1e-30)
    return bool(np.all(drops > -rel_tol * scale))

def analyze_run(res):
    pq = analyze_profile(res["x"], res["alpha_qs"])
    pd = analyze_profile(res["x"], res["alpha_dyn"])
    K = np.asarray(res["K_dyn"], float)
    tot_d = np.asarray(res["total_dyn"], float)
    kinetic_frac = float(np.max(K) / max(np.max(tot_d), 1e-30))
    regime = classify_regime(pd)
    return dict(
        regime=regime,
        n_cracks_dyn=pd["n_cracks"], n_cracks_qs=pq["n_cracks"],
        alpha_max_dyn=pd["alpha_max"], alpha_max_qs=pq["alpha_max"],
        damaged_fraction_dyn=pd["damaged_fraction"],
        mean_band_width_dyn=pd["mean_band_width"],
        kinetic_frac=kinetic_frac,
        qs_dyn_gap_total=float(res["gaps"]["total"]),
        energy_monotone_qs=energy_is_monotone(res["total_qs"]),
        second_crack_qs=bool(pq["n_cracks"] > pd["n_cracks"]),
    )

# =============================================================================
#  PROBLEM SCHEMATICS
# =============================================================================
def _spring(ax, x0, x1, y, n=6, amp=0.018, color="0.35", lw=1.2):
    lead = 0.12 * (x1 - x0)
    xs = [x0, x0 + lead]
    ys = [y, y]
    coil_x = np.linspace(x0 + lead, x1 - lead, 2 * n + 1)
    for i, cx in enumerate(coil_x):
        xs.append(cx)
        ys.append(y + amp * (1 if i % 2 else -1))
    xs += [x1 - lead, x1]
    ys += [y, y]
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round")

def _ground_hatch(ax, x0, x1, y, color="0.4"):
    ax.plot([x0, x1], [y, y], color=color, lw=1.5)
    n = 14
    xs = np.linspace(x0, x1, n)
    d = (x1 - x0) / n
    for xi in xs:
        ax.plot([xi, xi - d * 0.6], [y, y - 0.03], color=color, lw=0.9)

def _crack(ax, xc, y0, y1, color="#c0392b", lw=1.8, amp=0.012):
    ys = np.linspace(y0, y1, 9)
    xs = xc + amp * np.array([0, 1, -1, 1, -1, 1, -1, 1, 0])
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round")

def draw_mechanical_schematic(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Problem setup — mechanical", fontsize=11, fontweight="bold", pad=6)
    bx0, bx1 = 0.18, 0.82
    by, bh = 0.62, 0.12
    ax.add_patch(Rectangle((bx0, by), bx1 - bx0, bh, facecolor="#d6dbe0", edgecolor="black", lw=1.4, zorder=3))
    ax.add_patch(Rectangle((bx0 - 0.05, by - 0.04), 0.05, bh + 0.08, facecolor="none", edgecolor="black", lw=1.4, hatch="////", zorder=4))
    ax.plot([bx0 - 0.05, bx0 - 0.05], [by - 0.10, by + bh + 0.10], color="black", lw=2)
    ax.add_patch(FancyArrowPatch((bx1, by + bh / 2), (bx1 + 0.11, by + bh / 2), arrowstyle="-|>", mutation_scale=16, color="#1f4e79", lw=2, zorder=5))
    ax.text(bx1 + 0.12, by + bh / 2, r"$\hat U(t)$", color="#1f4e79", fontsize=11, va="center", ha="left")
    ax.text((bx0 + bx1) / 2, by + bh + 0.05, r"damage field $\alpha(\hat x),\;\hat x\in[0,1]$", ha="center", fontsize=9)
    ax.text(bx0 - 0.055, by + bh + 0.12, r"$u(0)=0$", fontsize=9, ha="left")
    _crack(ax, 0.70, by, by + bh)
    ax.text(0.70, by + bh + 0.02, "crack", color="#c0392b", fontsize=8, ha="center")
    gy = 0.30
    for xs0 in np.linspace(bx0 + 0.05, bx1 - 0.05, 6):
        ys = np.linspace(by, gy, 11)
        xx = xs0 + 0.012 * np.array([0, 1, -1, 1, -1, 1, -1, 1, -1, 1, 0])
        ax.plot(xx, ys, color="0.4", lw=1.0)
    _ground_hatch(ax, bx0 - 0.02, bx1 + 0.02, gy)
    ax.text((bx0 + bx1) / 2, gy - 0.07, r"elastic foundation, stiffness $\Lambda$", ha="center", fontsize=9, color="0.25")

def draw_thermal_schematic(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Problem setup — thermal", fontsize=11, fontweight="bold", pad=6)
    bx0, bx1 = 0.16, 0.84
    by, bh = 0.60, 0.12
    ax.add_patch(Rectangle((bx0, by), bx1 - bx0, bh, facecolor="#fbe3cf", edgecolor="black", lw=1.4, zorder=3))
    for xc in (bx0, bx1):
        ax.add_patch(plt.Circle((xc, by - 0.03), 0.012, color="0.3", zorder=5))
    ax.text(bx0, by + bh + 0.10, "free ends", fontsize=9, ha="center", color="0.25")
    ax.text(bx1, by + bh + 0.10, "free ends", fontsize=9, ha="center", color="0.25")
    ax.text((bx0 + bx1) / 2, by + bh + 0.16, r"mean stress $\bar\sigma$ measured", fontsize=9, ha="center")
    ax.add_patch(FancyArrowPatch((0.40, by + bh / 2), (0.30, by + bh / 2), arrowstyle="-|>", mutation_scale=12, color="#c0392b", lw=1.6, zorder=6))
    ax.add_patch(FancyArrowPatch((0.60, by + bh / 2), (0.70, by + bh / 2), arrowstyle="-|>", mutation_scale=12, color="#c0392b", lw=1.6, zorder=6))
    ax.text((bx0 + bx1) / 2, by + bh / 2, r"$\theta(t)$", color="#c0392b", fontsize=12, va="center", ha="center", fontweight="bold")
    for xc in np.linspace(bx0 + 0.10, bx1 - 0.10, 5):
        _crack(ax, xc, by, by + bh, lw=1.4)
    ax.text((bx0 + bx1) / 2, by - 0.10, "periodic fragmentation", fontsize=8, ha="center", color="#c0392b")
    gy = 0.28
    for xs0 in np.linspace(bx0 + 0.05, bx1 - 0.05, 7):
        ys = np.linspace(by, gy, 11)
        xx = xs0 + 0.012 * np.array([0, 1, -1, 1, -1, 1, -1, 1, -1, 1, 0])
        ax.plot(xx, ys, color="0.4", lw=1.0)
    _ground_hatch(ax, bx0 - 0.02, bx1 + 0.02, gy)
    ax.text((bx0 + bx1) / 2, gy - 0.07, r"elastic foundation, stiffness $\Lambda$", ha="center", fontsize=9, color="0.25")

# =============================================================================
#  REGIME-MAP FIGURE
# =============================================================================
def _log_edges(centers):
    c = np.asarray(centers, float)
    le = np.empty(c.size + 1)
    le[1:-1] = np.sqrt(c[:-1] * c[1:])
    le[0]  = c[0]**2 / le[1]
    le[-1] = c[-1]**2 / le[-2]
    return le

def _panel_regime_map(ax, lhat_vals, lambda_vals, regime, n_cracks,
                      inertia_mask=None, lhat_res=None, eta=None,
                      show_counts=False):
    Xe = _log_edges(lhat_vals)
    Ye = _log_edges(lambda_vals)
    cmap = ListedColormap([REGIME_COLORS[c] for c in (R_NONE, R_ONE, R_MULTI, R_DIFF)])
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], cmap.N)
    ax.pcolormesh(Xe, Ye, regime, cmap=cmap, norm=norm, edgecolors="white", linewidth=0.4, shading="flat")

    if show_counts:
        for i, lam in enumerate(lambda_vals):
            for j, lh in enumerate(lhat_vals):
                if regime[i, j] in (R_MULTI, R_DIFF) and n_cracks[i, j] > 0:
                    ax.text(lh, lam, f"{int(n_cracks[i, j])}", ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")

    if inertia_mask is not None and inertia_mask.any():
        ax.pcolor(Xe, Ye, np.ma.masked_where(~inertia_mask, inertia_mask), hatch="xxx", alpha=0.0, edgecolor="0.15", linewidth=0.0)

    if lhat_res is not None:
        ax.axvline(lhat_res, color="0.15", ls=":", lw=1.3)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(Xe[0], Xe[-1]); ax.set_ylim(Ye[0], Ye[-1])
    ax.set_xlabel(r"internal length  $\hat\ell$")
    ax.set_ylabel(r"foundation stiffness  $\Lambda$")
    if eta is not None:
        ax.set_title(rf"$\eta = {eta:g}$", fontsize=11)
    ax.tick_params(labelsize=8)

def make_regime_figure(grid, physics, out_path, with_schematic=True):
    lhat   = grid["lhat"]
    lam    = grid["lambda"]
    etas   = grid["eta"]
    lhat_res = grid.get("lhat_res", None)

    if with_schematic:
        fig = plt.figure(figsize=(15.5, 9.0))
        outer = fig.add_gridspec(1, 2, width_ratios=[2.25, 1.0], wspace=0.16)
        gs = outer[0, 0].subgridspec(2, 2, hspace=0.34, wspace=0.30)
        ax_sch = fig.add_subplot(outer[0, 1])
        if physics == "mechanical":
            draw_mechanical_schematic(ax_sch)
        else:
            draw_thermal_schematic(ax_sch)
    else:
        fig = plt.figure(figsize=(12.5, 9.0))
        gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    for k, eta in enumerate(etas[:4]):
        _panel_regime_map(axes[k], lhat, lam, grid["regime"][eta], grid["ncracks"][eta], inertia_mask=grid.get("inertia", {}).get(eta), lhat_res=lhat_res, eta=eta)

    title = ("MECHANICAL test" if physics == "mechanical" else "THERMAL test")
    fig.suptitle(rf"$\hat\ell$–$\Lambda$ regime map — {title}" "\n(regime read from the final dynamic damage profile; numbers = crack/fragment count)", fontsize=14, fontweight="bold", y=0.985)

    handles = [Patch(facecolor=REGIME_COLORS[c], edgecolor="white", label=REGIME_NAMES[c].replace("\n", " ")) for c in (R_NONE, R_ONE, R_MULTI, R_DIFF)]
    handles.append(Line2D([0], [0], color="0.15", ls=":", lw=1.3, label=r"under-resolved  $\hat\ell\lesssim 2h$"))
    handles.append(Patch(facecolor="white", edgecolor="0.15", hatch="xxx", label="inertia-dominated (large $K$/$E$ & QS–Dyn gap)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.005))

    #cap = (r"Reading the map:  small $\hat\ell$ $\rightarrow$ thin, numerous cracks (severe fragmentation);  "
    #       r"large $\hat\ell$ & large $\Lambda$ $\rightarrow$ damage bands merge into a diffuse/homogeneous field;  "
    #       r"small $\Lambda$ $\rightarrow$ no crack nucleates in the loading window (monotone energy).  "
    #       r"Larger $\eta$ amplifies inertial overshoot, pushing the fragmentation/diffuse boundary.")
    # fig.text(0.5, 0.052, cap, ha="center", va="bottom", fontsize=9, bbox=dict(boxstyle="round,pad=0.5", facecolor="aliceblue", edgecolor="steelblue", alpha=0.85))

    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.13)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")

# =============================================================================
#  SWEEP DRIVER
# =============================================================================
def build_grid_from_runs(run_metrics, lhat_vals, lambda_vals, eta_vals,
                         inertia_kfrac=0.15, lhat_res=None):
    NL, Nl = len(lambda_vals), len(lhat_vals)
    grid = {"lhat": np.asarray(lhat_vals), "lambda": np.asarray(lambda_vals),
            "eta": list(eta_vals), "lhat_res": lhat_res,
            "regime": {}, "ncracks": {}, "inertia": {}}
    for eta in eta_vals:
        reg = np.full((NL, Nl), R_NONE, dtype=int)
        ncr = np.zeros((NL, Nl), dtype=int)
        inr = np.zeros((NL, Nl), dtype=bool)
        for i, lam in enumerate(lambda_vals):
            for j, lh in enumerate(lhat_vals):
                m = run_metrics[(eta, lam, lh)]
                reg[i, j] = m["regime"]
                ncr[i, j] = m["n_cracks_dyn"]
                inr[i, j] = (m["kinetic_frac"] >= inertia_kfrac)
        grid["regime"][eta] = reg
        grid["ncracks"][eta] = ncr
        grid["inertia"][eta] = inr
    return grid


def run_sweep(physics, lhat_vals, lambda_vals, eta_vals,
              mesh_parameters, AltMin_parameters, Newmark_parameters,
              loading_parameters):
    runner = run_mechanical_simulation if physics == "mechanical" else run_thermal_simulation
    out = {}
    combos = list(itertools.product(eta_vals, lambda_vals, lhat_vals))
    
    # EDIT: Split the work across available MPI ranks
    my_combos = combos[comm.rank :: comm.size]

    for n, (eta, lam, lh) in enumerate(my_combos, 1):
        # Using a simple print now so all cores can output their progress locally
        print(f"[Rank {comm.rank} | {physics} {n}/{len(my_combos)}] eta={eta:.4g}, lam={lam:.4g}, lh={lh:.4g}")
        model = {"l_hat": float(lh), "Lambda": float(lam), "eta": float(eta)}
        res = runner(model, mesh_parameters, loading_parameters,
                     AltMin_parameters, Newmark_parameters)
        out[(eta, lam, lh)] = analyze_run(res)
        
    # EDIT: Gather all the sub-dictionaries to rank 0
    gathered_out = comm.gather(out, root=0)
    
    if comm.rank == 0:
        combined_out = {}
        for d in gathered_out:
            combined_out.update(d)
        return combined_out
    
    return None

def _key(eta, lam, lh):
    return f"{eta:.6g}|{lam:.6g}|{lh:.6g}"

def save_cache(path, physics_metrics, lhat_vals, lambda_vals, eta_vals):
    flat = {}
    for physics, metrics in physics_metrics.items():
        for (eta, lam, lh), m in metrics.items():
            flat[f"{physics}::{_key(eta, lam, lh)}"] = m
    np.savez(path,
             meta=json.dumps(dict(lhat=list(map(float, lhat_vals)),
                                  lam=list(map(float, lambda_vals)),
                                  eta=list(map(float, eta_vals)))),
             data=json.dumps(flat))
    print(f"cache saved -> {path}")

def load_cache(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    flat = json.loads(str(z["data"]))
    out = {"mechanical": {}, "thermal": {}}
    for key, m in flat.items():
        physics, rest = key.split("::")
        eta, lam, lh = (float(s) for s in rest.split("|"))
        out[physics][(eta, lam, lh)] = m
    return meta, out

# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":

    # ---------------- configuration ----------------
    RECOMPUTE = True            

    lhat_vals   = np.geomspace(0.001, 0.2, 50)
    lambda_vals = np.geomspace(0.1,   50.0, 50)
    eta_vals    = [0.001, 0.01, 0.05, 0.1]

    mesh_parameters    = {"nx": 200}
    # EDIT: Relaxed tolerance for speed
    AltMin_parameters  = {"max_iter": 500, "tol": 1e-5}
    Newmark_parameters = {"beta": 0.25, "gamma": 0.5}

    N_QS, N_DYN = 50, 300        
    mech_loading  = {"U_max": 1.7,    "T0": 1.0, "N_steps_qs": N_QS, "N_steps_dyn": N_DYN}
    therm_loading = {"theta_max": 6.0, "T0": 1.0, "N_steps_qs": N_QS, "N_steps_dyn": N_DYN}

    lhat_res = 2.0 / mesh_parameters["nx"]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "Output")
    
    if comm.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "sweep_cache.npz")

    # ---------------- compute or load ----------------
    if RECOMPUTE or not os.path.exists(cache_path):
        if not HAVE_FENICS:
            raise RuntimeError(
                "dolfinx/FEniCSx is not importable in this environment."
            )
        if comm.rank == 0:
            print(f"Parallel Sweep: Distributing {len(lhat_vals)*len(lambda_vals)*len(eta_vals)*2} runs across {comm.size} cores.")
        
        # All ranks execute run_sweep now!
        mech_metrics  = run_sweep("mechanical", lhat_vals, lambda_vals, eta_vals,
                                  mesh_parameters, AltMin_parameters, Newmark_parameters, mech_loading)
        therm_metrics = run_sweep("thermal", lhat_vals, lambda_vals, eta_vals,
                                  mesh_parameters, AltMin_parameters, Newmark_parameters, therm_loading)
        
        if comm.rank == 0:
            save_cache(cache_path, {"mechanical": mech_metrics, "thermal": therm_metrics},
                       lhat_vals, lambda_vals, eta_vals)
    else:
        # If not recomputing, only rank 0 needs to load and plot
        if comm.rank == 0:
            meta, cached = load_cache(cache_path)
            lhat_vals   = np.asarray(meta["lhat"])
            lambda_vals = np.asarray(meta["lam"])
            eta_vals    = list(meta["eta"])
            mech_metrics, therm_metrics = cached["mechanical"], cached["thermal"]

    # ---------------- figures (rank 0 only) ----------------
    if comm.rank == 0:
        mech_grid  = build_grid_from_runs(mech_metrics,  lhat_vals, lambda_vals, eta_vals, lhat_res=lhat_res)
        therm_grid = build_grid_from_runs(therm_metrics, lhat_vals, lambda_vals, eta_vals, lhat_res=lhat_res)

        make_regime_figure(mech_grid,  "mechanical",
                           os.path.join(out_dir, "regime_map_mechanical_with_schematic.png"), True)
        make_regime_figure(mech_grid,  "mechanical",
                           os.path.join(out_dir, "regime_map_mechanical_plots_only.png"), False)
        make_regime_figure(therm_grid, "thermal",
                           os.path.join(out_dir, "regime_map_thermal_with_schematic.png"), True)
        make_regime_figure(therm_grid, "thermal",
                           os.path.join(out_dir, "regime_map_thermal_plots_only.png"), False)
        print("All figures written to", out_dir)