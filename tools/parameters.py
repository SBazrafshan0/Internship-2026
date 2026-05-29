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
``Lambda`` -- foundation stiffness :math:`\\Lambda` (non-dimensional).  Note
this is *not* the elastic Lame parameter :math:`\\lambda_{\\rm Lame}` (the
latter is computed from ``E_ref`` and ``nu``).
``eta``    -- dynamic *loading time-scale* :math:`\\eta`, used only through
:math:`\\tau = \\eta t` in the imposed dynamic load.  It is no longer a mass
or kinetic-energy multiplier.  Smaller ``eta`` slows the loading down, so the
dynamic run is extended to ``t_final_dyn = 1 / eta``.

Mesh parameters
---------------
``physics``       -- ``"1D"`` or ``"2D"``.
(The geometry ``shape`` is *not* a user parameter: each problem file hard-codes
its own shape via ``_PROBLEM_SHAPE``.  For a different geometry, write a new
problem file rather than changing a parameter here.)
``mesh_per_lhat`` -- *single* knob controlling mesh resolution.  The cell
                      size is set to ``h = l_hat / mesh_per_lhat``.  Use
                      ``>= 4`` to resolve diffuse crack bands cleanly.
``Lx``, ``Ly``    -- domain dimensions (1D uses only ``Lx``).

Loading parameters
------------------
``U_max`` / ``theta_max`` -- *amplitude* of the load, reached at the canonical
pseudo-time ``t = 1``: the QS ramp is ``U_max * t`` and the dynamic ramp
``U_max * (tau/2)(1 + tanh(tau/T0))`` with ``tau = eta * t``.  Both reach the
amplitude at ``t = 1`` (QS) / ``tau = 1`` (dynamic), so the final times are
fixed: ``t_final_qs = 1`` and ``t_final_dyn = 1 / eta``.
``T0``                   -- smoothing length of the dynamic ramp.
``N_steps_qs`` / ``N_steps_dyn`` -- *number of steps* over each run.  The QS
step is ``dt = 1 / N_steps_qs``; the dynamic step is stretched to
``dt = (1/eta) / N_steps_dyn`` so the run covers the extended dynamic time.
``N_snapshots`` -- number of intermediate snapshots kept for plotting.

Crack-nucleation *generations* are detected from the run history (jumps in the
surface energy, labelled by the connected damaged-region count) and marked on
the energy plot -- there is no stopping/threshold parameter.

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
    "Lambda": 1.0,
    "eta":    1.0e-2,
    "E_ref":  1.0,    # reference Young's modulus (for non-dimensionalisation)
    "nu":     0.3,    # Poisson's ratio (only used for 2D plane strain elasticity)
    # Viscous dissipation potential  Q = 0.5 * int( c1|u'|^2 + c2 eps(u'):C:eps(u') + c3|alpha'|^2 ):
    "c1":     0.0e-3,    # local-velocity damping
    "c2":     0.0e-3,    # strain-rate damping (Kelvin-Voigt, also a Cauchy stress component)
    "c3":     1.0e-3,    # damage-rate damping (viscous regularisation of damage evolution)
}

DEFAULT_MESH_PARAMETERS = {
    "physics":       "1D",          # "1D" or "2D"
    "mesh_per_lhat": 4,             # cells per regularisation length
    "Lx":            1.0,
    "Ly":            0.3,
}

# Loading settings shared by both problems (step counts, snapshots, fracture
# marker).  Each problem dict below adds only its own amplitude (U_max /
# theta_max) and ramp smoothing T0.
_COMMON_LOADING = {
    "N_steps_qs":   30,             # number of quasi-static steps (dt = 1/N)
    "N_steps_dyn": 180,             # number of dynamic steps
    "N_snapshots":   20,             # intermediate snapshots kept for plotting
}

DEFAULT_MECH_LOADING = {
    "U_max":       0.7,
    "T0":          0.7,
    **_COMMON_LOADING,
}

DEFAULT_THERM_LOADING = {
    "theta_max":   20.0,
    "T0":          0.7,
    **_COMMON_LOADING,
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
    E     = model.get("E_ref", 1.0)
    nu    = model.get("nu", 0.0)
    c1    = model.get("c1", 0.0)
    c2    = model.get("c2", 0.0)
    c3    = model.get("c3", 0.0)
    if physics_type == "mechanical":
        amp_key, amp_val = "umax", loading["U_max"]
    else:
        amp_key, amp_val = "thmax", loading["theta_max"]
    return (
        f"{physics_type}_{ph}_{shape}_{mdl}"
        f"_lhat{model['l_hat']}_lam{model['Lambda']}_eta{model['eta']}"
        f"_E{E:g}_nu{nu:g}_c1{c1:g}_c2{c2:g}_c3{c3:g}"
        f"_{amp_key}{amp_val:.2f}"
        f"_nQS{loading['N_steps_qs']}_nDyn{loading['N_steps_dyn']}"
        f"_mpl{mpl}_T0{loading['T0']}"
    )
