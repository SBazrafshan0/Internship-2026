"""
problems/
=========
One file per *physical problem*.  Each file:

* defines the geometry / boundary conditions,
* assembles the energy density (delegating the choice of AT1 vs AT2 to
  :mod:`tools.solvers`),
* exposes a single :func:`run_problem` entry point so the sweep driver can
  call it generically,
* and is also runnable as a stand-alone script
  (``python -m problems.dynamic`` or ``python problems/dynamic.py``).

To switch the *model variant* (AT1/AT2) or the *physics* (1D/2D), edit the
``solver_parameters["model"]`` and ``mesh_parameters["physics"]`` entries at
the bottom of the file -- no other change is needed.

If you ever need a problem whose geometry breaks the symmetry between 1D and
2D (e.g. an L-shape, a specimen with a notch, a different loading device),
create a *new* file in this folder rather than overloading an existing one.
"""

from .dynamic import run_problem as run_dynamic       # noqa: F401
from .thermal import run_problem as run_thermal       # noqa: F401

#: Dispatcher used by the sweep driver:  ``PROBLEMS[name](...)``
PROBLEMS = {
    "dynamic": run_dynamic,
    "thermal": run_thermal,
}
