"""
sweep.py
========
Parameter-sweep driver.

Usage
-----
::

    # run a sweep over the dynamic problem:
    python sweep.py dynamic

    # ... or the thermal one:
    python sweep.py thermal

    # ... or both:
    python sweep.py both

    # specify the number of joblib workers:
    COWORK_N_WORKERS=8 python sweep.py dynamic

Each combination of the sweep values defined in :data:`SWEEP` becomes a single
problem call.  Because each problem is *embarrassingly parallel* with respect
to the others, we farm them out to ``joblib`` workers when available
(falling back to a serial loop otherwise).

To switch the *base* set of parameters, edit :data:`BASE_OVERRIDES` below
(everything that is not in :data:`SWEEP` is inherited from
``tools.parameters.get_defaults``).

Adding a new problem to the sweep
---------------------------------
1. Drop a new file in ``problems/`` that exposes a ``run_problem(...)``
   function with the same signature as ``problems.dynamic.run_problem``.
2. Register it in ``problems/__init__.py`` (``PROBLEMS[name] = ...``).
3. Call ``python sweep.py <new_name>``.
"""
from __future__ import annotations
import sys
import itertools
from copy import deepcopy
from pathlib import Path

# Make ``tools`` / ``problems`` importable when run as a script.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.imports import (
    HAVE_JOBLIB, Parallel, delayed, n_workers, comm,
)
from tools.parameters import get_defaults
from problems import PROBLEMS


# =============================================================================
# Sweep configuration -- *EDIT HERE*
# =============================================================================
#: Values to sweep.  Lists with a single element disable that dimension.
SWEEP = {
    # phase-field internal length
    "l_hat":   [0.01, 0.02, 0.04, 0.08],
    # foundation stiffness
    "Lambda":  [1.0, 10.0, 20.0, 50.0],
    # inverse wave speed (set very small for quasi-static-like behaviour)
    "eta":     [1e-3, 1e-2, 5e-2, 1e-1],
    # number of quasi-static load steps (dynamic steps default to 3x this)
    "N_qs":    [60],
    # which model variant to test
    "model":   ["AT2"],            # add "AT1" to sweep both
    # physics (1D or 2D)
    "physics": ["1D"],             # add "2D" to sweep both
}

#: Overrides that apply to *every* run on top of the dimension-agnostic
#: defaults from ``tools.parameters.get_defaults``.
BASE_OVERRIDES = {
    "mesh_parameters": {
        "nx": 200,                 # 1D resolution / x-resolution in 2D
        "ny": 30,                  # 2D only
        "Lx": 1.0,
        "Ly": 1.0,
    },
    "AltMin_parameters": {
        "max_iter": 500,
        "tol":      1e-7,
    },
    "Newmark_parameters": {
        "beta":  0.25,
        "gamma": 0.5,
    },
    # ratio "N_dyn / N_qs" used when N_qs is in the sweep:
    "_dyn_to_qs_ratio": 3,
}


# =============================================================================
# Helpers
# =============================================================================
def _enumerate_runs():
    """Cartesian product of :data:`SWEEP`.  Yields one config dict per run."""
    keys = list(SWEEP.keys())
    for combo in itertools.product(*[SWEEP[k] for k in keys]):
        yield dict(zip(keys, combo))


def _build_config(problem_name: str, run: dict) -> dict:
    """Merge a single sweep point with the defaults / overrides into the
    nested dictionary expected by ``run_problem``."""
    physics_kind = "mechanical" if problem_name == "dynamic" else "thermal"
    cfg = get_defaults(physics_kind)

    # overrides
    for section, ovr in BASE_OVERRIDES.items():
        if section.startswith("_"):
            continue
        cfg[section].update(ovr)

    # sweep entries
    cfg["model_parameters"]["l_hat"]  = run["l_hat"]
    cfg["model_parameters"]["Lambda"] = run["Lambda"]
    cfg["model_parameters"]["eta"]    = run["eta"]
    cfg["solver_parameters"]["model"] = run["model"]
    cfg["mesh_parameters"]["physics"] = run["physics"]

    n_qs = run["N_qs"]
    cfg["loading_parameters"]["N_steps_qs"]  = n_qs
    cfg["loading_parameters"]["N_steps_dyn"] = BASE_OVERRIDES["_dyn_to_qs_ratio"] * n_qs

    return cfg


def _run_one(problem_name: str, run: dict) -> str:
    """Pure-function worker -- safe to ship to a joblib process."""
    cfg = _build_config(problem_name, run)
    runner = PROBLEMS[problem_name]
    try:
        runner(**cfg, verbose=False)
        return f"OK  {problem_name} {run}"
    except Exception as exc:                # pragma: no cover
        return f"ERR {problem_name} {run} -- {exc}"


# =============================================================================
# Entry point
# =============================================================================
def run_sweep(problem_name: str):
    if problem_name not in PROBLEMS:
        raise SystemExit(
            f"Unknown problem {problem_name!r}.  Available: {sorted(PROBLEMS)}"
        )

    runs = list(_enumerate_runs())
    n_runs = len(runs)
    n_par  = n_workers()

    if comm.rank == 0:
        print("=" * 64)
        print(f" SWEEP  problem={problem_name}  runs={n_runs}  workers={n_par}")
        print(f" SWEEP  joblib={'yes' if HAVE_JOBLIB else 'no (serial fallback)'}")
        print("=" * 64)

    if HAVE_JOBLIB and n_par > 1 and comm.size == 1:
        # joblib parallelism over independent FEM problems
        results = Parallel(n_jobs=n_par, verbose=10)(
            delayed(_run_one)(problem_name, run) for run in runs
        )
    else:
        # serial fallback (also when running under mpirun -- in that case
        # MPI handles intra-problem parallelism, joblib would oversubscribe)
        results = []
        for i, run in enumerate(runs, start=1):
            if comm.rank == 0:
                print(f"[{i}/{n_runs}] {run}")
            results.append(_run_one(problem_name, run))

    if comm.rank == 0:
        print("\nSUMMARY")
        for line in results:
            print(" ", line)
        print(f"\nDone -- {n_runs} runs.")


def main():
    target = sys.argv[1] if len(sys.argv) >= 2 else "dynamic"
    if target == "both":
        run_sweep("dynamic")
        run_sweep("thermal")
    else:
        run_sweep(target)


if __name__ == "__main__":
    main()
