"""
sweep.py
========
Parameter-sweep driver -- **one problem at a time**.

Usage
-----
::

    python sweep.py dynamic        # sweep over the mechanical/dynamic problem
    python sweep.py thermal        # sweep over the thermal problem

The driver requires *exactly one* problem name; there is intentionally no
"both" shortcut, so you cannot accidentally launch two sweeps in one
command.

Configuration
-------------
* :data:`SWEEP` -- the cartesian product of values that defines the runs.
  Lists with a single element disable that dimension.
* :data:`BASE_OVERRIDES` -- entries applied on top of the dimension-agnostic
  defaults from :func:`tools.parameters.get_defaults`.  Every key in
  ``mesh_parameters``, ``AltMin_parameters``, etc. can be set here.

Parallelism
-----------
Each ``(eta, Lambda, l_hat, model, physics, N_qs)`` combination is a
self-contained FEM problem -- joblib runs them in parallel, one per CPU
core.  Override the worker count with the environment variable
``COWORK_N_WORKERS=<N>``.  When the driver is launched under ``mpirun``,
joblib is bypassed (MPI handles intra-problem parallelism) and the runs
are processed serially on every rank.
"""

from __future__ import annotations
import sys
import itertools
from pathlib import Path

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
SWEEP = {
    "l_hat":   [0.01, 0.02, 0.04, 0.08],
    "Lambda":  [1.0, 10.0, 20.0, 50.0],
    "eta":     [1e-3, 1e-2, 5e-2, 1e-1],
    "N_qs":    [60],
    "model":   ["AT2"],            # add "AT1" to sweep both
    "physics": ["1D"],             # add "2D" to sweep both
}

BASE_OVERRIDES = {
    "mesh_parameters": {
        "mesh_per_lhat": 5,        # cells per regularisation length
        "shape":         "rectangle",
        "Lx":            1.0,
        "Ly":            1.0,
    },
    "AltMin_parameters": {
        "max_iter": 500,
        "tol":      1e-7,
    },
    "Newmark_parameters": {
        "beta":  0.25,
        "gamma": 0.5,
    },
    "_dyn_to_qs_ratio": 3,
}


# =============================================================================
# Helpers
# =============================================================================
def _enumerate_runs():
    keys = list(SWEEP.keys())
    for combo in itertools.product(*[SWEEP[k] for k in keys]):
        yield dict(zip(keys, combo))


def _build_config(problem_name: str, run: dict) -> dict:
    physics_kind = "mechanical" if problem_name == "dynamic" else "thermal"
    cfg = get_defaults(physics_kind)

    for section, ovr in BASE_OVERRIDES.items():
        if section.startswith("_"):
            continue
        cfg[section].update(ovr)

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
        results = Parallel(n_jobs=n_par, verbose=10)(
            delayed(_run_one)(problem_name, run) for run in runs
        )
    else:
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


# =============================================================================
# CLI
# =============================================================================
USAGE = (
    "Usage: python sweep.py <problem>\n"
    f"  <problem> is one of: {sorted(PROBLEMS)}\n"
    "Only one problem at a time -- there is no 'both' option."
)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(USAGE)
    target = sys.argv[1].strip().lower()
    if target in {"both", "all"}:
        raise SystemExit(
            "sweep.py runs one problem at a time -- please pick exactly one of: "
            f"{sorted(PROBLEMS)}"
        )
    if target not in PROBLEMS:
        raise SystemExit(f"Unknown problem {target!r}.\n{USAGE}")
    run_sweep(target)


if __name__ == "__main__":
    main()
