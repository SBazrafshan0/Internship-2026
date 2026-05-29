"""
tools/imports.py
================
Centralised imports.  Every problem / sweep file does::

    from tools.imports import *

That way:

* third-party versions are pinned at one place,
* MPI / PETSc / FEniCSx are imported in the right order,
* a few parallel back-ends used by the *sweep* driver are exposed too.

Parallel back-ends
------------------
Two complementary mechanisms are available:

* **mpi4py** -- the native parallelism of FEniCSx.  A single problem can be
  split across processes (mesh partitioning) by calling ``mpirun -n NP
  python problems/dynamic.py``.
* **joblib** -- used by the sweep driver to launch *several independent FEM
  problems in parallel*, one per CPU core.  This is the recommended way to
  saturate a multi-core machine for parameter studies because each individual
  problem is small.

A helper :func:`n_workers` returns a sensible default number of workers based
on the machine's CPU count and on the ``COWORK_N_WORKERS`` environment
variable (so the user can override it without touching the source code).
"""

from __future__ import annotations

# ---- standard library -------------------------------------------------------
import os
import json
import time
import itertools
from pathlib import Path
from typing import Any, Callable

# ---- scientific stack -------------------------------------------------------
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")          # headless / cluster-safe
import matplotlib.pyplot as plt

# ---- FEniCSx / PETSc --------------------------------------------------------
import ufl
import dolfinx
from dolfinx import mesh as df_mesh, fem, io
import dolfinx.fem.petsc
from dolfinx.mesh import CellType

from mpi4py import MPI
from petsc4py import PETSc

# Single, repository-wide MPI communicator.
comm = MPI.COMM_WORLD

# ---- parallel back-end for parameter sweeps --------------------------------
try:
    from joblib import Parallel, delayed
    HAVE_JOBLIB = True
except Exception:                                       # pragma: no cover
    HAVE_JOBLIB = False
    Parallel = None
    delayed = None

import multiprocessing as mp

# ---- progress bar (optional) ------------------------------------------------
try:
    from tqdm import tqdm
    HAVE_TQDM = True
except Exception:                                       # pragma: no cover
    HAVE_TQDM = False

    def tqdm(iterable=None, **kwargs):                  # type: ignore
        """Minimal stand-in -- prints a periodic progress line every ~10%.

        If ``tqdm`` is not installed the problem files still get readable
        progress output (no pretty bar though).  Install ``tqdm`` for the
        full experience: ``pip install tqdm``.
        """
        if iterable is None:
            return _DummyTQDM(**kwargs)
        return _wrap_periodic(iterable, **kwargs)


    class _DummyTQDM:
        def __init__(self, total=None, desc="", disable=False, **kw):
            self.total   = total or 0
            self.desc    = desc
            self.disable = disable
            self.n       = 0
        def update(self, k=1):
            self.n += k
            if self.disable:
                return
            if self.total and (self.n == 1 or self.n == self.total
                               or self.n % max(1, self.total // 10) == 0):
                pct = 100.0 * self.n / self.total
                print(f"  [{self.desc}] {self.n}/{self.total} ({pct:5.1f}%)", flush=True)
        def close(self):
            pass
        def set_postfix(self, **kw):
            pass
        def __enter__(self):  return self
        def __exit__(self, *a): self.close()


    def _wrap_periodic(iterable, desc="", total=None, **_kw):
        try:
            total = total if total is not None else len(iterable)
        except TypeError:
            total = None
        step = max(1, (total or 10) // 10)
        for i, x in enumerate(iterable, 1):
            yield x
            if total is None or i == total or i % step == 0:
                pct = 100.0 * i / total if total else 0.0
                print(f"  [{desc}] {i}/{total or '?'} ({pct:5.1f}%)", flush=True)


def n_workers(default: int | None = None) -> int:
    """
    Decide how many workers the *sweep* driver should launch.

    Resolution order
    ----------------
    1. The ``COWORK_N_WORKERS`` environment variable, if set.
    2. The ``default`` argument, if not ``None``.
    3. ``os.cpu_count() - 1`` (leave one core for the OS), with a floor of 1.
    """
    env = os.environ.get("COWORK_N_WORKERS")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    if default is not None:
        return max(1, int(default))
    cpu = os.cpu_count() or 1
    return max(1, cpu - 1)


__all__ = [
    # std-lib
    "os", "json", "time", "itertools", "Path", "Any", "Callable",
    # numerics
    "np", "sp", "matplotlib", "plt",
    # FEniCSx
    "ufl", "dolfinx", "df_mesh", "fem", "io", "CellType",
    # PETSc / MPI
    "MPI", "PETSc", "comm",
    # parallel back-end
    "HAVE_JOBLIB", "Parallel", "delayed", "mp", "n_workers",
    # progress bar
    "tqdm", "HAVE_TQDM",
]
