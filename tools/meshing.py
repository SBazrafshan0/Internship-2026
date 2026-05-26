"""
tools/meshing.py
================
Mesh factory.  Returns the FEniCSx mesh, the boundary marker tags and the
quadrature measures used by the problem files.

* 1D: ``create_interval`` on ``(0, Lx)``.
* 2D: ``create_rectangle`` on ``(0, Lx) x (0, Ly)`` with triangular cells.
  We use ``diagonal=DiagonalType.crossed`` so that the mesh is symmetric
  (each rectangular cell is split into four triangles), which avoids the
  spurious anisotropy that the standard ``right``/``left`` diagonals would
  introduce in fragmentation patterns.

The boundary tags are:

* ``1`` -- left edge   (x = 0)
* ``2`` -- right edge  (x = Lx)
* ``3`` -- bottom edge (2D only, y = 0)
* ``4`` -- top edge    (2D only, y = Ly)
"""

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh as df_mesh
from dolfinx.mesh import CellType, DiagonalType


def create_mesh_and_tags(mesh_parameters: dict, comm=MPI.COMM_WORLD):
    """
    Build the mesh and the boundary tag object.

    Returns
    -------
    domain : dolfinx.mesh.Mesh
    mt     : dolfinx.mesh.MeshTags     -- facet tags (1..4, see module doc).
    dx, ds : ufl.Measure              -- volume / surface measures bound to
                                          ``domain`` (and to ``mt`` for ``ds``).
    """
    physics = mesh_parameters.get("physics", "1D")
    Lx      = mesh_parameters.get("Lx", 1.0)
    Ly      = mesh_parameters.get("Ly", 1.0)
    nx      = mesh_parameters["nx"]
    ny      = mesh_parameters.get("ny") or nx

    if physics == "1D":
        domain = df_mesh.create_interval(comm, nx, (0.0, Lx))
        markers_def = {
            1: lambda x: np.isclose(x[0], 0.0),
            2: lambda x: np.isclose(x[0], Lx),
        }
    elif physics == "2D":
        domain = df_mesh.create_rectangle(
            comm,
            [np.array([0.0, 0.0]), np.array([Lx, Ly])],
            [nx, ny],
            cell_type=CellType.triangle,
            diagonal=DiagonalType.crossed,
        )
        markers_def = {
            1: lambda x: np.isclose(x[0], 0.0),
            2: lambda x: np.isclose(x[0], Lx),
            3: lambda x: np.isclose(x[1], 0.0),
            4: lambda x: np.isclose(x[1], Ly),
        }
    else:
        raise ValueError(f"Unknown physics {physics!r} -- expected '1D' or '2D'.")

    fdim = domain.topology.dim - 1

    facets, markers = [], []
    for tag, fn in markers_def.items():
        f = df_mesh.locate_entities_boundary(domain, fdim, fn)
        facets.append(f)
        markers.append(np.full(len(f), tag, dtype=np.int32))
    all_facets = np.concatenate(facets).astype(np.int32) if facets else np.array([], dtype=np.int32)
    all_markers = np.concatenate(markers).astype(np.int32) if markers else np.array([], dtype=np.int32)
    sort_ix = np.argsort(all_facets)

    mt = df_mesh.meshtags(domain, fdim, all_facets[sort_ix], all_markers[sort_ix])

    dx = ufl.Measure("dx", domain=domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)
    return domain, mt, dx, ds


def grad_sq(alpha, physics: str):
    """
    UFL helper -- returns :math:`|\\nabla\\alpha|^2`, which is

    * ``alpha.dx(0)**2`` in 1D, and
    * ``ufl.inner(ufl.grad(alpha), ufl.grad(alpha))`` in 2D.

    Centralising this here means the problem files use the same scalar damage
    field in both dimensions without conditionals.
    """
    if physics == "1D":
        return alpha.dx(0) ** 2
    return ufl.inner(ufl.grad(alpha), ufl.grad(alpha))


def strain(u, physics: str):
    """
    UFL strain tensor / strain scalar.

    * 1D: scalar :math:`\\partial u/\\partial x`.
    * 2D: symmetric gradient :math:`\\tfrac12(\\nabla u + \\nabla u^T)`.
    """
    if physics == "1D":
        return u.dx(0)
    return ufl.sym(ufl.grad(u))


def elastic_strain(u, theta, physics: str):
    """
    Total strain minus an isotropic eigenstrain ``theta * I``.

    Used by the thermal problem so that the elastic part is
    ``eps_e = eps - theta * I``.  In 1D this collapses to ``u.dx(0) - theta``.
    """
    eps = strain(u, physics)
    if physics == "1D":
        return eps - theta
    return eps - theta * ufl.Identity(2)


def stress(eps_e, alpha, physics: str, lame_mu: float = 1.0, lame_lambda: float = 0.0):
    """
    Stress = degraded elastic stress.

    * 1D: ``(1-alpha)**2 * eps_e``.
    * 2D: ``(1-alpha)**2 * (2 mu eps_e + lambda tr(eps_e) I)``.

    The default Lame parameters give a unit shear modulus and no volumetric
    contribution.  Override them in the problem file if a different elastic
    behaviour is wanted.
    """
    g = (1.0 - alpha) ** 2
    if physics == "1D":
        return g * eps_e
    return g * (2.0 * lame_mu * eps_e + lame_lambda * ufl.tr(eps_e) * ufl.Identity(2))
