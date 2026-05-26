"""
tools/meshing.py
================
**Unstructured** mesh factory.

The cell size is driven by a *single* knob tied to the regularisation length:

    h = l_hat / mesh_per_lhat

so a coarser-or-finer mesh follows the physics automatically.  No more
``nx`` / ``ny`` to set by hand.

* 1D --  ``dolfinx.mesh.create_interval`` with
  ``nx = ceil(Lx / h)``.  (Intervals are inherently structured but the
  cell count tracks ``l_hat``.)
* 2D --  Gmsh-generated *unstructured* triangulation with
  ``characteristic length = h``.  Falls back to a crossed-diagonal
  structured mesh if Gmsh is not importable.

Geometry templates
------------------
The factory dispatches on ``mesh_parameters["shape"]`` (default
``"rectangle"``).  To add a new shape, just append a builder to
:data:`GEOMETRY_BUILDERS`; the rest of the code (problem files, sweep
driver, plotters) keeps working without change.

Every builder produces the same canonical boundary tags so that the
problem files never have to know which shape they are running on:

    1 -- "left"     (x-min side, or first wall in any geometry)
    2 -- "right"    (x-max side, or last wall)
    3 -- "bottom"   (y-min, optional)
    4 -- "top"      (y-max, optional)
    100 -- interior surface tag

If a future shape has no natural "top/bottom", the corresponding tag is
simply absent and ``mt.find(...)`` returns an empty array.
"""

from __future__ import annotations
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as df_mesh
from dolfinx.mesh import CellType, DiagonalType

try:
    import gmsh
    from dolfinx.io import gmshio
    HAVE_GMSH = True
except Exception:                                       # pragma: no cover
    HAVE_GMSH = False


# =============================================================================
# Target cell size
# =============================================================================
def target_h(mesh_parameters: dict, model_parameters: dict) -> float:
    """
    Compute the target cell size ``h = l_hat / mesh_per_lhat``.

    ``mesh_per_lhat`` defaults to 4 (i.e. four cells across one internal
    length, which is the empirical lower bound to resolve a diffuse crack
    band cleanly).
    """
    mpl = float(mesh_parameters.get("mesh_per_lhat", 4))
    if mpl <= 0:
        raise ValueError("mesh_per_lhat must be > 0")
    return float(model_parameters["l_hat"]) / mpl


# =============================================================================
# 1D
# =============================================================================
def _create_1d(mesh_parameters, model_parameters, comm):
    h  = target_h(mesh_parameters, model_parameters)
    Lx = float(mesh_parameters.get("Lx", 1.0))
    nx = max(2, int(np.ceil(Lx / h)))
    domain = df_mesh.create_interval(comm, nx, (0.0, Lx))
    markers = {
        1: lambda x: np.isclose(x[0], 0.0),
        2: lambda x: np.isclose(x[0], Lx),
    }
    return domain, markers


# =============================================================================
# 2D Gmsh builders
# =============================================================================
def _gmsh_rectangle(Lx: float, Ly: float, lc: float):
    """Add a rectangle to the current gmsh model.  Boundary tags 1/2/3/4
    are bound to left/right/bottom/top."""
    gmsh.model.add("domain")
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(Lx,  0.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(Lx,  Ly,  0.0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, Ly,  0.0, lc)
    l_bot   = gmsh.model.geo.addLine(p1, p2)
    l_right = gmsh.model.geo.addLine(p2, p3)
    l_top   = gmsh.model.geo.addLine(p3, p4)
    l_left  = gmsh.model.geo.addLine(p4, p1)
    loop = gmsh.model.geo.addCurveLoop([l_bot, l_right, l_top, l_left])
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [l_left],  1)
    gmsh.model.addPhysicalGroup(1, [l_right], 2)
    gmsh.model.addPhysicalGroup(1, [l_bot],   3)
    gmsh.model.addPhysicalGroup(1, [l_top],   4)
    gmsh.model.addPhysicalGroup(2, [surf],    100)


#: Registry of 2D shape builders.  Add a new one with::
#:
#:     GEOMETRY_BUILDERS["my_shape"] = lambda mesh_params, lc: ...
#:
#: The builder must populate the current gmsh model with physical groups
#: 1..N (line tags) and 100 (surface tag).
GEOMETRY_BUILDERS = {
    "rectangle": lambda mp, lc: _gmsh_rectangle(
        float(mp.get("Lx", 1.0)),
        float(mp.get("Ly", 1.0)),
        lc,
    ),
}


# =============================================================================
# 2D dispatch
# =============================================================================
def _create_2d_gmsh(mesh_parameters, model_parameters, comm, shape: str):
    if shape not in GEOMETRY_BUILDERS:
        raise ValueError(
            f"Unknown 2D shape {shape!r}.  Known: {list(GEOMETRY_BUILDERS)}"
        )
    lc = target_h(mesh_parameters, model_parameters)
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        GEOMETRY_BUILDERS[shape](mesh_parameters, lc)
        gmsh.option.setNumber("Mesh.Algorithm", 5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5 * lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1.5 * lc)
        gmsh.model.mesh.generate(2)
        domain, _cell_tags, facet_tags = gmshio.model_to_mesh(
            gmsh.model, comm, 0, gdim=2
        )
    finally:
        gmsh.finalize()
    return domain, facet_tags


def _create_2d_fallback(mesh_parameters, model_parameters, comm, shape: str):
    """Structured fallback if Gmsh is unavailable -- *rectangle only*."""
    if shape != "rectangle":
        raise RuntimeError(
            f"Shape {shape!r} requires Gmsh.  Install with "
            "`pip install gmsh` (or use the conda-forge package)."
        )
    Lx = float(mesh_parameters.get("Lx", 1.0))
    Ly = float(mesh_parameters.get("Ly", 1.0))
    h  = target_h(mesh_parameters, model_parameters)
    nx = max(2, int(np.ceil(Lx / h)))
    ny = max(2, int(np.ceil(Ly / h)))
    domain = df_mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([Lx, Ly])],
        [nx, ny],
        cell_type=CellType.triangle,
        diagonal=DiagonalType.crossed,
    )
    markers = {
        1: lambda x: np.isclose(x[0], 0.0),
        2: lambda x: np.isclose(x[0], Lx),
        3: lambda x: np.isclose(x[1], 0.0),
        4: lambda x: np.isclose(x[1], Ly),
    }
    return domain, markers


# =============================================================================
# Public entry point
# =============================================================================
def create_mesh_and_tags(mesh_parameters: dict, model_parameters: dict,
                          comm=MPI.COMM_WORLD):
    """Build mesh, facet tags, and the dx/ds measures."""
    physics = mesh_parameters.get("physics", "1D")
    shape   = mesh_parameters.get("shape", "rectangle")

    if physics == "1D":
        domain, markers = _create_1d(mesh_parameters, model_parameters, comm)
        mt = _build_meshtags(domain, markers)
    elif physics == "2D":
        if HAVE_GMSH:
            domain, mt = _create_2d_gmsh(mesh_parameters, model_parameters,
                                         comm, shape)
        else:
            domain, markers = _create_2d_fallback(mesh_parameters,
                                                  model_parameters, comm, shape)
            mt = _build_meshtags(domain, markers)
    else:
        raise ValueError(f"Unknown physics {physics!r} -- expected '1D' or '2D'.")

    dx = ufl.Measure("dx", domain=domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)
    return domain, mt, dx, ds


def _build_meshtags(domain, markers: dict):
    fdim = domain.topology.dim - 1
    facets, vals = [], []
    for tag, fn in markers.items():
        f = df_mesh.locate_entities_boundary(domain, fdim, fn)
        facets.append(f)
        vals.append(np.full(len(f), tag, dtype=np.int32))
    if facets:
        all_f = np.concatenate(facets).astype(np.int32)
        all_v = np.concatenate(vals).astype(np.int32)
    else:
        all_f = np.array([], dtype=np.int32)
        all_v = np.array([], dtype=np.int32)
    sort_ix = np.argsort(all_f)
    return df_mesh.meshtags(domain, fdim, all_f[sort_ix], all_v[sort_ix])


# =============================================================================
# UFL helpers
# =============================================================================
def grad_sq(alpha, physics: str):
    """|grad alpha|^2."""
    if physics == "1D":
        return alpha.dx(0) ** 2
    return ufl.inner(ufl.grad(alpha), ufl.grad(alpha))


def strain(u, physics: str):
    """Strain (scalar 1D, sym-grad 2D)."""
    if physics == "1D":
        return u.dx(0)
    return ufl.sym(ufl.grad(u))


def elastic_strain(u, theta, physics: str):
    """eps - theta I (isotropic eigenstrain)."""
    eps = strain(u, physics)
    if physics == "1D":
        return eps - theta
    return eps - theta * ufl.Identity(2)


def stress(eps_e, alpha, physics: str,
           lame_mu: float = 1.0, lame_lambda: float = 0.0):
    """Degraded elastic stress."""
    g = (1.0 - alpha) ** 2
    if physics == "1D":
        return g * eps_e
    return g * (2.0 * lame_mu * eps_e
                + lame_lambda * ufl.tr(eps_e) * ufl.Identity(2))
