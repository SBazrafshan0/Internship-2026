"""
tools/helpers.py
================
Low-level helpers shared by every problem.

* :class:`SNESProblem` -- thin adapter exposing a UFL residual ``F`` and its
  Jacobian to PETSc's *Scalable Nonlinear Equations Solver* (SNES).  Used both
  for the displacement (linear, ``ksponly``) and the damage (bound-constrained,
  ``vinewtonrsls``) sub-problems.
* :func:`make_linear_snes` / :func:`make_damage_snes` -- factory functions
  that build the two SNES objects with a sensible default configuration.
* :func:`alt_min_loop` -- one outer alternate-minimisation iteration over the
  pair ``(u, alpha)`` (or ``(a_new, alpha)`` in dynamics).  Returns the L2
  increment of alpha so that the caller can stop on a tolerance.
"""

from .imports import (
    fem, ufl, PETSc, MPI, np, comm,
)


# =============================================================================
# Mesh / DoF reporting
# =============================================================================
def print_mesh_info(domain, V_u, V_alpha, label: str = "", comm_=None):
    """
    Print a one-line summary of the mesh / function-space sizes so the user
    can see, at the start of every run, how big the problem is.

    Output looks like::

        [mesh|MECH 2D]  cells=12000  dofs_u=12462 (vector)  dofs_alpha=6231
    """
    comm_ = comm_ if comm_ is not None else comm
    tdim    = domain.topology.dim
    n_cells = domain.topology.index_map(tdim).size_global
    n_dofs_u     = V_u.dofmap.index_map.size_global * V_u.dofmap.index_map_bs
    n_dofs_alpha = V_alpha.dofmap.index_map.size_global * V_alpha.dofmap.index_map_bs
    vec_tag = " (vector)" if V_u.dofmap.index_map_bs > 1 else ""
    if comm_.rank == 0:
        print(
            f"  [mesh|{label}]  cells={n_cells}"
            f"  dofs_u={n_dofs_u}{vec_tag}"
            f"  dofs_alpha={n_dofs_alpha}"
        )


# =============================================================================
# SNES wrapper
# =============================================================================
class SNESProblem:
    """
    Adapter: ``(F, u, bcs)`` --> PETSc SNES call-backs ``F(snes,x,F)`` and
    ``J(snes,x,J,P)``.

    Parameters
    ----------
    F : ufl.Form
        Residual form (= derivative of an energy if you started from one).
    u : dolfinx.fem.Function
        The unknown field this residual is taken with respect to.
    bcs : list of dolfinx.fem.DirichletBC
        Dirichlet boundary conditions on ``u``.
    J : ufl.Form, optional
        Jacobian.  If ``None`` it is built automatically via
        ``ufl.derivative(F, u, du)``.
    """

    def __init__(self, F, u, bcs, J=None):
        V = u.function_space
        du = ufl.TrialFunction(V)
        self.L = fem.form(F)
        self.a = fem.form(J) if J is not None else fem.form(ufl.derivative(F, u, du))
        self.bcs = bcs
        self.u = u

    # PETSc SNES call-backs ---------------------------------------------------
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


# =============================================================================
# Standard PETSc solvers
# =============================================================================
def make_linear_snes(problem, V, factor: str | None = "mumps") -> PETSc.SNES:
    """
    ``ksponly`` SNES with an LU preconditioner -- used for *linear* sub-problems
    (elastic equilibrium, Newmark acceleration update).
    """
    b  = fem.petsc.create_vector(V)
    Jm = fem.petsc.create_matrix(problem.a)
    snes = PETSc.SNES().create()
    snes.setType("ksponly")
    snes.setFunction(problem.F, b)
    snes.setJacobian(problem.J, Jm)
    snes.setTolerances(rtol=1.0e-9, max_it=50)
    snes.getKSP().setType("preonly")
    snes.getKSP().getPC().setType("lu")
    if factor:
        try:
            snes.getKSP().getPC().setFactorSolverType(factor)
        except Exception:
            pass                                # fall back to default LU
    return snes


def make_damage_snes(problem, lb, ub, factor: str | None = "mumps") -> PETSc.SNES:
    """Bound-constrained SNES (``vinewtonrsls``) used for the damage update."""
    V = problem.u.function_space
    b  = fem.petsc.create_vector(V)
    Jm = fem.petsc.create_matrix(problem.a)
    snes = PETSc.SNES().create()
    snes.setType("vinewtonrsls")
    snes.setFunction(problem.F, b)
    snes.setJacobian(problem.J, Jm)
    snes.setTolerances(rtol=1.0e-9, max_it=50)
    snes.getKSP().setType("preonly")
    snes.getKSP().getPC().setType("lu")
    snes.setVariableBounds(lb.x.petsc_vec, ub.x.petsc_vec)
    if factor:
        try:
            snes.getKSP().getPC().setFactorSolverType(factor)
        except Exception:
            pass
    return snes


# =============================================================================
# Alternate-minimisation outer loop
# =============================================================================
def alt_min_loop(
    primal_solver, primal_x,
    damage_solver, alpha,
    alpha_old, err_form,
    max_iter: int, tol: float,
    comm_=None,
):
    """
    Run alternate-minimisation until ``||alpha - alpha_old||_L2 <= tol``.

    Parameters
    ----------
    primal_solver  : PETSc.SNES        -- solver for u (or a_new in dynamics)
    damage_solver  : PETSc.SNES        -- solver for alpha
    alpha          : fem.Function      -- the damage field
    alpha_old      : fem.Function      -- previous iterate (used as scratch)
    err_form       : fem.Form          -- compiled form for ``(alpha - alpha_old)**2 * dx``
    max_iter, tol  : convergence settings
    comm_          : MPI communicator (defaults to the world communicator)

    Returns
    -------
    iterations : int   number of alternations actually used
    """
    comm_ = comm_ if comm_ is not None else comm
    for k in range(1, max_iter + 1):
        primal_solver.solve(None, primal_x)
        primal_x.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        damage_solver.solve(None, alpha.x.petsc_vec)
        alpha.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        err = comm_.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM)
        alpha_old.x.array[:] = alpha.x.array
        if float(np.sqrt(max(err, 0.0))) <= tol:
            return k
    return max_iter


# =============================================================================
# Crack counting and crack-nucleation events
# =============================================================================
def make_crack_counter(domain, V_alpha, thr: float = 0.5):
    """Return ``count(alpha_array) -> int``: the number of connected damaged
    regions (``alpha > thr``), i.e. the current crack count.

    Adjacency is built once from the cell -> dof map, so the *same* routine
    counts isolated damage bands in 1D (interval cells, 2 dofs) and connected
    crack clusters in 2D (triangle cells, 3 dofs).  Counting is serial (one MPI
    rank); in parallel it returns the rank-local count.
    """
    tdim    = domain.topology.dim
    n_cells = domain.topology.index_map(tdim).size_local
    dofmap  = V_alpha.dofmap

    # All unique dof-dof edges within each cell (undirected).
    edge_set = set()
    for c in range(n_cells):
        d = dofmap.cell_dofs(c)
        for j in range(len(d)):
            for k in range(j + 1, len(d)):
                a, b = int(d[j]), int(d[k])
                edge_set.add((a, b) if a < b else (b, a))
    if edge_set:
        edges = np.asarray(sorted(edge_set), dtype=np.int64)
        e_a, e_b = edges[:, 0], edges[:, 1]
    else:
        e_a = e_b = np.empty(0, dtype=np.int64)

    def count(alpha):
        dmg = np.asarray(alpha) > thr
        if not dmg.any():
            return 0
        active = dmg[e_a] & dmg[e_b]
        parent = np.arange(dmg.size)

        def find(x):
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        for a, b in zip(e_a[active], e_b[active]):
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[rb] = ra
        return len({find(int(i)) for i in np.nonzero(dmg)[0]})

    return count


def detect_crack_events(load, surface_energy, n_cracks, min_dS: float = 1e-9):
    """Locate crack-nucleation *generations* from a run history.

    Hybrid rule: an event is registered when the connected-crack count reaches a
    new running maximum (a genuinely new crack appeared, since irreversibility
    makes the damaged set only grow) *and* the surface energy increased over the
    step (``dS > min_dS``).  The ``dS`` guard anchors each mark to a real
    fracture-energy release and filters numerical flicker.

    Returns a list of dicts ``{gen, load, n_cracks, new, S}`` -- one per
    generation, where ``new`` is how many cracks that generation added.
    """
    load = np.asarray(load, dtype=float)
    S    = np.asarray(surface_energy, dtype=float)
    nc   = np.asarray(n_cracks, dtype=int)
    if nc.size == 0:
        return []
    dS = np.diff(S, prepend=S[0])
    events = []
    running_max = 0
    for i in range(nc.size):
        if nc[i] > running_max and dS[i] > min_dS:
            events.append({
                "gen":      len(events) + 1,
                "load":     float(load[i]),
                "n_cracks": int(nc[i]),
                "new":      int(nc[i] - running_max),
                "S":        float(S[i]),
            })
            running_max = nc[i]
    return events
