"""
tools/parameters.py
===================
Default parameter dictionaries.

Everything is **non-dimensional** and dimension-agnostic, i.e. exactly the same
dictionaries are reused for 1D and 2D problems.  The geometric / dimensional
information lives in ``mesh_parameters``; ``physics`` is the switch that
selects between a 1D segment and a 2D unstructured (Gmsh) triangulation.

Physical / model parameters
---------------------------
``l_hat``  -- regularisation length :math:`\\hat\\ell` (ratio of the
internal length to the characteristic length of the domain).
``Lambda`` -- foundation stiffness :math:`\\Lambda` (non-dimensional).
``eta``    -- inverse wave-speed :math:`\\eta`.

Mesh parameters
---------------
``physics``       -- ``"1D"`` or ``"2D"``.
``shape``         -- ``"rectangle"`` (default).  Anything you register in
                      :data:`tools.meshing.GEOMETRY_BUILDERS` is accepted.
``mesh_per_lhat`` -- *single* knob controlling mesh resolution.  The cell
                      size is set to ``h = l_hat / mesh_per_lhat``.  Use
                      ``>= 4`` to resolve diffuse crack bands cleanly.
``Lx``, ``Ly``    -- domain dimensions (1D uses only ``Lx``).

Loading parameters
------------------
``U_max`` / ``theta_max`` -- *amplitude* of the load at the canonical
pseudo-time ``t = 1`` (the load keeps growing past ``t = 1`` if the
simulation is not yet stopped by the damage criterion).
``T0``                   -- smoothing length of the ramp.
``N_steps_qs`` / ``N_steps_dyn`` -- *resolution* of the pseudo-time grid:
the step size is ``dt = 1 / N_steps``.  The total number of steps is
*not* fixed -- the simulation stops when the damage threshold is reached
(see below).
``N_snapshots`` -- number of intermediate snapshots kept for plotting.
``alpha_break`` -- stop when :math:`\\max_\\Omega\\alpha \\ge`
``alpha_break`` (default 0.99 -- "complete failure at some point").
``t_max``       -- safety upper bound on the pseudo-time.  If the damage
threshold is never reached, the run bails out at ``t = t_max`` (default
3.0, i.e. up to triple the canonical loading amplitude).

Solver parameters
-----------------
``model`` -- ``"AT1"`` or ``"AT2"``.

Alternate-minimisation
----------------------
``max_iter``, ``tol`` -- outer-loop convergence settings.

Newmark
-------
``beta``, ``gamma`` -- standard Newmark-:math:`\\beta` coefficients.
"""

from copy import deepcopy


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_MODEL_PARAMETERS = {
    "l_hat":  0.02,
    "Lambda": 10.0,
    "eta":    1.0e-2,
}

DEFAULT_MESH_PARAMETERS = {
    "physics":       "1D",          # "1D" or "2D"
    "shape":         "rectangle",   # any key registered in GEOMETRY_BUILDERS
    "mesh_per_lhat": 4,             # cells per regularisation length
    "Lx":            1.0,
    "Ly":            1.0,
}

DEFAULT_MECH_LOADING = {
    "U_max":       1.4,
    "T0":          1.0,
    "N_steps_qs":   60,             # step *resolution*  (dt = 1/N)
    "N_steps_dyn": 180,
    "N_snapshots":   6,
    "alpha_break": 0.99,            # stop when max(alpha) >= this
    "t_max":         3.0,           # safety cap on pseudo-time
}

DEFAULT_THERM_LOADING = {
    "theta_max":   4.0,
    "T0":          1.0,
    "N_steps_qs":   60,
    "N_steps_dyn": 180,
    "N_snapshots":   6,
    "alpha_break": 0.99,
    "t_max":         3.0,
}

DEFAULT_SOLVER_PARAMETERS = {
    "model": "AT2",                 # "AT1" or "AT2"
}

DEFAULT_ALTMIN_PARAMETERS = {
    "max_iter": 500,
    "tol":      1.0e-7,
}

DEFAULT_NEWMARK_PARAMETERS = {
    "beta":  0.25,
    "gamma": 0.5,
}


# -----------------------------------------------------------------------------
# Convenience
# -----------------------------------------------------------------------------
def get_defaults(physics_type: str) -> dict:
    """
    Return a *fresh* nested dictionary of defaults for ``physics_type`` in
    ``{"mechanical", "thermal"}``.  Modifying the returned dictionary leaves
    the originals untouched.
    """
    if physics_type == "mechanical":
        loading = deepcopy(DEFAULT_MECH_LOADING)
    elif physics_type == "thermal":
        loading = deepcopy(DEFAULT_THERM_LOADING)
    else:
        raise ValueError(f"Unknown physics_type {physics_type!r}")

    return dict(
        model_parameters   = deepcopy(DEFAULT_MODEL_PARAMETERS),
        mesh_parameters    = deepcopy(DEFAULT_MESH_PARAMETERS),
        loading_parameters = loading,
        solver_parameters  = deepcopy(DEFAULT_SOLVER_PARAMETERS),
        AltMin_parameters  = deepcopy(DEFAULT_ALTMIN_PARAMETERS),
        Newmark_parameters = deepcopy(DEFAULT_NEWMARK_PARAMETERS),
    )


def filename_stub(physics_type: str, model: dict, mesh: dict,
                  loading: dict, solver: dict) -> str:
    """
    Build a *unique, human-readable* filename stub from a parameter set.
    Used by :mod:`tools.plotting` so that PNG/PDF/XDMF outputs of different
    runs cannot be silently overwritten.
    """
    ph    = mesh["physics"]
    shape = mesh.get("shape", "rectangle")
    mdl   = solver["model"]
    mpl   = mesh.get("mesh_per_lhat", 4)
    if physics_type == "mechanical":
        amp_key, amp_val = "umax", loading["U_max"]
    else:
        amp_key, amp_val = "thmax", loading["theta_max"]
    return (
        f"{physics_type}_{ph}_{shape}_{mdl}"
        f"_lhat{model['l_hat']}_lam{model['Lambda']}_eta{model['eta']}"
        f"_{amp_key}{amp_val:.2f}"
        f"_nQS{loading['N_steps_qs']}_nDyn{loading['N_steps_dyn']}"
        f"_mpl{mpl}_T0{loading['T0']}"
    )
