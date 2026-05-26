"""
tools/parameters.py
===================
Default parameter dictionaries.

Everything is **non-dimensional** and dimension-agnostic, i.e. exactly the same
dictionaries are reused for 1D and 2D problems.  The geometric / dimensional
information lives in ``mesh_parameters``; ``physics`` is the switch that
selects between a 1D segment and a 2D rectangle (triangular mesh).

Physical / model parameters
---------------------------
``l_hat``  -- regularisation length :math:`\\hat\\ell` (ratio of the
internal length to the characteristic length of the domain).
``Lambda`` -- foundation stiffness :math:`\\Lambda` (non-dimensional);
controls how strongly the bar / plate is pulled back towards the reference
configuration.
``eta``    -- inverse wave-speed :math:`\\eta = \\sqrt{\\rho/E}\\,L/\\tau`,
i.e. the dynamical parameter that toggles between the quasi-static and the
truly inertial regimes.

Mesh parameters
---------------
``physics`` -- ``"1D"`` or ``"2D"``.
``nx``      -- number of cells along x (for 2D, ``ny`` defaults to ``nx``,
the rectangle is meshed with **triangles** and the option
``diagonal="crossed"`` to keep the mesh symmetric).
``Lx``, ``Ly`` -- domain dimensions (1D uses only ``Lx``).

Loading parameters
------------------
``U_max`` / ``theta_max`` -- imposed amplitude of the displacement /
thermal load at the end of the pseudo-time interval ``t in [0,1]``.
``T0``                   -- smoothing length of the load ramp
``U(t) = U_dot (sqrt(T0^2 + t^2) - T0)`` (a ``C^{\\infty}`` ramp that starts
with zero velocity).
``N_steps_qs`` / ``N_steps_dyn`` -- number of pseudo-time steps for the
quasi-static and dynamic loops.
``N_snapshots`` -- number of intermediate damage profiles stored for
visualisation.

Solver parameters
-----------------
``model`` -- ``"AT1"`` or ``"AT2"`` (see :mod:`tools.solvers`).

Alternate-minimisation parameters
---------------------------------
``max_iter``, ``tol`` -- outer-loop convergence settings.

Newmark parameters
------------------
``beta``, ``gamma`` -- standard Newmark-:math:`\\beta` coefficients; defaults
to the unconditionally stable :math:`\\beta = 1/4`, :math:`\\gamma = 1/2`
(implicit average acceleration).
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
    "physics": "1D",          # "1D" or "2D"
    "nx": 200,
    "ny": None,               # 2D only; falls back to nx when None
    "Lx": 1.0,
    "Ly": 0.2,
}

DEFAULT_MECH_LOADING = {
    "U_max":      1.4,
    "T0":         1.0,
    "N_steps_qs":  60,
    "N_steps_dyn": 180,
    "N_snapshots": 6,
}

DEFAULT_THERM_LOADING = {
    "theta_max":  1.75,
    "T0":         1.0,
    "N_steps_qs":  60,
    "N_steps_dyn": 180,
    "N_snapshots": 6,
}

DEFAULT_SOLVER_PARAMETERS = {
    "model": "AT2",           # "AT1" or "AT2"
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
    ph = mesh["physics"]
    mdl = solver["model"]
    if physics_type == "mechanical":
        amp_key, amp_val = "umax", loading["U_max"]
    else:
        amp_key, amp_val = "thmax", loading["theta_max"]
    nx, ny = mesh["nx"], (mesh.get("ny") or mesh["nx"])
    mesh_tag = f"nx{nx}" if ph == "1D" else f"nx{nx}_ny{ny}"
    return (
        f"{physics_type}_{ph}_{mdl}"
        f"_lhat{model['l_hat']}_lam{model['Lambda']}_eta{model['eta']}"
        f"_{amp_key}{amp_val:.2f}"
        f"_nQS{loading['N_steps_qs']}_nDyn{loading['N_steps_dyn']}"
        f"_{mesh_tag}_T0{loading['T0']}"
    )
