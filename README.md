# Phase-field fragmentation -- academic repository layout

```
fragmentation_repo/
├── README.md                     <- this file
├── sweep.py                      <- parameter-sweep driver (joblib parallel)
│
├── problems/                     <- ONE FILE PER PHYSICAL PROBLEM
│   ├── __init__.py               <- PROBLEMS dispatcher
│   ├── dynamic.py                <- mechanical / dynamic test  (run as script)
│   ├── thermal.py                <- thermal fragmentation test (run as script)
│   ├── dynamic_theory.ipynb      <- theory + numerical scheme
│   └── thermal_theory.ipynb      <- theory + numerical scheme
│
├── tools/                        <- LIBRARY CODE -- NEVER edit during a run
│   ├── __init__.py
│   ├── imports.py                <- centralised third-party imports (FEniCSx,
│   │                                PETSc, matplotlib, joblib ...)
│   ├── helpers.py                <- SNESProblem + AltMin loop
│   ├── parameters.py             <- default parameter dictionaries (1D & 2D)
│   ├── solvers.py                <- AT1, AT2 fracture-energy densities
│   ├── meshing.py                <- 1D interval / 2D triangular rectangle
│   └── plotting.py               <- matplotlib + Paraview XDMF export
│
└── output/
    ├── png/                      <- 300 dpi PNGs
    ├── pdf/                      <- vector PDFs
    └── paraview/                 <- XDMF + H5 time series for 2D inspection
```

## What goes where

* **`problems/`** -- one file per physical problem.  Each file defines:
  geometry, boundary conditions, the run loops (quasi-static + dynamic) and
  is *runnable as a script* (`python problems/thermal.py`).  Two switches
  live at the very bottom of every file:

      cfg["solver_parameters"]["model"]   = "AT1" | "AT2"
      cfg["mesh_parameters"]["physics"]   = "1D"  | "2D"

  If a problem becomes too asymmetric to stay under those switches, **create
  a new file** rather than overloading an existing one.

* **`tools/`** -- the library.  Everything the problem files need is
  centralised here.  Most importantly:

  * **`solvers.py`** -- where the AT1 / AT2 dissipation lives.  Add a new
    variant by appending an entry to the `MODELS` dict.
  * **`parameters.py`** -- where the default parameter dictionaries live;
    `filename_stub(...)` builds the filename used for the saved figures so
    that no two runs can silently overwrite each other.
  * **`meshing.py`** -- 1D `create_interval`, 2D triangular `create_rectangle`
    (with `DiagonalType.crossed` to remove spurious anisotropy).

* **`sweep.py`** -- sweeps across `(l_hat, Lambda, eta, model, physics, N_qs)`.
  Pass the problem name on the command line: `python sweep.py dynamic`.
  Joblib drives the parallelism (one CPU per FEM problem).

## How to run

```bash
# single run
python problems/dynamic.py             # 1D, AT2, default parameters
python problems/thermal.py

# sweep
python sweep.py dynamic
python sweep.py thermal
python sweep.py both

# parallel sweep (joblib, defaults to ncpu-1 workers)
COWORK_N_WORKERS=8 python sweep.py both

# native FEniCSx parallelism inside a single problem
mpirun -n 4 python problems/dynamic.py
```

## Where things land

For a single run of `problems/dynamic.py` with `l_hat=0.02, Lambda=10, eta=1e-2,
U_max=1.4, N_qs=60, N_dyn=180, nx=200, T0=1, AT2, 1D`:

* `output/png/mechanical_1D_AT2_lhat0.02_lam10.0_eta0.01_umax1.40_nQS60_nDyn180_nx200_T01.0.png`
* `output/pdf/...`
* `output/paraview/...xdmf` + `.h5`

The filename always encodes the *problem*, *physics*, *model*, and the *full
parameter set* so that two runs cannot collide.

## Adding a new physical problem

1. Create `problems/my_problem.py` with a `run_problem(...)` entry point.
2. Register it in `problems/__init__.py`:
   ```python
   from .my_problem import run_problem as run_my_problem
   PROBLEMS["my_problem"] = run_my_problem
   ```
3. Add a `problems/my_problem_theory.ipynb` next to it.
4. `python sweep.py my_problem` Just Works(tm).

## Adding a new model variant

Edit `tools/solvers.py`:
```python
MODELS["my_model"] = {
    "w":   lambda a: a**3,
    "c_w": 4.0,
    "description": "...",
}
```
All problem files will pick it up automatically.

## Parallelism

Two complementary mechanisms are exposed:

* **MPI / FEniCSx** -- intra-problem domain decomposition.  Useful for big 2D
  meshes; just prepend `mpirun -n NP` to the command.
* **joblib** -- inter-problem parallelism in the sweep driver.  Each worker
  takes one independent FEM problem from the queue.  Override the number of
  workers with `COWORK_N_WORKERS=<N>`.

The two should *not* be combined: when MPI is on, joblib falls back to a
serial loop to avoid oversubscription.
