"""
tools/
======
Shared utilities for the phase-field fragmentation repository.

Sub-modules
-----------
* :mod:`tools.imports`    -- centralised third-party imports (FEniCSx, PETSc,
  matplotlib, parallel back-ends ...).  Every problem / sweep script should do
  ``from tools.imports import *`` so that the same versions are picked up
  everywhere.
* :mod:`tools.helpers`    -- the ``SNESProblem`` adapter that wraps a UFL form
  into the call-backs PETSc's SNES expects, plus the alternate-minimisation
  inner loop.
* :mod:`tools.parameters` -- default parameter dictionaries.  Each dictionary
  is dimension-agnostic; a single ``physics`` switch (``"1D"`` or ``"2D"``)
  selects between an interval or a rectangle made of triangles.
* :mod:`tools.solvers`    -- definitions of the AT1 and AT2 fracture
  surface-energy densities (the only place where the model variant lives).
* :mod:`tools.meshing`    -- factory that returns the FEniCSx mesh (1D
  ``create_interval`` or 2D triangular ``create_rectangle``) together with the
  boundary tags used by the problem files.
* :mod:`tools.plotting`   -- matplotlib figures (saved both as PNG and PDF
  with a problem-aware filename) plus Paraview-ready XDMF export.
"""
