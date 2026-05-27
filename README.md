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
│   ├── imports.py                <- centralised third-party imports
│   ├── helpers.py                <- SNES wrapper + AltMin loop + mesh info
│   ├── parameters.py             <- default parameter dictionaries
│   ├── solvers.py                <- AT1, AT2 fracture-energy densities
│   ├── meshing.py                <- 1D interval / 2D Gmsh unstructured mesh
│   └── plotting.py               <- matplotlib + Paraview XDMF export
│
└── output/{png,pdf,paraview}/
```


## How to run

### A. one isolated run

The two problem files are *self-contained scripts*.  Open the file you want
to run, edit the three switches at the bottom (`model`, `physics`,
`mesh_per_lhat`) and launch it:

```bash
python problems/dynamic.py          # mechanical test, one set of parameters
python problems/thermal.py          # thermal test
```

The figure header and the file name encode the full parameter set, so you
cannot silently overwrite a previous run.

### B. parameter sweep

`sweep.py` farms many independent runs out to joblib (one CPU per FEM
problem).  It runs **one problem at a time**, controlled by a single
constant at the top of the file:

```python
PROBLEM = "dynamic"   # "dynamic" or "thermal" or any key of PROBLEMS
```

Then either:

```bash
python sweep.py                     # uses whatever PROBLEM is set to
python sweep.py thermal             # one-off override of the default
```

To add a new problem `myproblem`:

1. Create `problems/myproblem.py` (copy `dynamic.py` as a template, change
   `_PROBLEM_SHAPE` if you need a different geometry).
2. Register it in `problems/__init__.py`:
   ```python
   from .myproblem import run_problem as run_myproblem
   PROBLEMS["myproblem"] = run_myproblem
   ```
3. Set `PROBLEM = "myproblem"` in `sweep.py`.
4. `python sweep.py`.

### C. parallelism

```bash
# inter-problem (sweeps): one joblib worker per FEM problem
COWORK_N_WORKERS=8 python sweep.py

# intra-problem (one big FEM problem on multiple cores)
mpirun -n 4 python problems/dynamic.py
```

The two should *not* be combined.  When the driver is launched under
`mpirun`, joblib is bypassed automatically.


## Key conventions

### Mesh resolution -- single knob

The cell size is set from the regularisation length:

    h = l_hat / mesh_per_lhat

so `mesh_per_lhat = 5` gives 5 cells across one internal length.  There is
**no** `nx`/`ny` to set by hand.  In 2D, Gmsh generates an unstructured
triangulation with that characteristic length.

### Stopping criterion -- damage hits 1

Both the QS and dynamic loops are `while` loops that step until any damage
point reaches `alpha_break` (default 0.99), with a safety cap `t_max`
(default 3.0) to prevent infinite loops in the sub-critical regime.

### Geometry

The default geometry is a rectangle with `Lx = 1.0`, `Ly = 0.2` (a thin
strip), defined in `tools/parameters.py` and overridable from
`sweep.BASE_OVERRIDES`.  The *shape name* (e.g. `"rectangle"`,
`"plate_with_hole"`, ...) is **not** a user-facing knob: each problem file
pins its own shape via the module-level constant `_PROBLEM_SHAPE`.  Need a
different geometry?  Write a new problem file rather than tweaking an
existing one, and register a new builder in
`tools.meshing.GEOMETRY_BUILDERS`.

### Switches at the bottom of each problem file

```python
cfg["solver_parameters"]["model"]      = "AT1" | "AT2"
cfg["mesh_parameters"]["physics"]      = "1D"  | "2D"
cfg["mesh_parameters"]["mesh_per_lhat"] = 5
```

Anything else (`U_max`, `Lambda`, `eta`, ...) is read from the defaults in
`tools/parameters.py` -- override it inside `__main__` or via
`BASE_OVERRIDES` in `sweep.py`.

### Paraview output

`tools.plotting.export_paraview` writes an XDMF time series of `alpha` and
`u`.  In 2D this is *the* way to look at the crack pattern.  In **1D** the
output is redundant (matplotlib already shows the damage profile), so
the Paraview export is **skipped automatically** when `physics == "1D"`.
You don't need to do anything -- 1D runs produce only PNG/PDF.

### Output file names

A run with `l_hat=0.02, Lambda=10, eta=0.01, AT2, 2D, mesh_per_lhat=5,
U_max=1.4, T0=1, N_qs=60, N_dyn=180` writes:

```
output/png/mechanical_2D_rectangle_AT2_lhat0.02_lam10.0_eta0.01_umax1.40_nQS60_nDyn180_mpl5_T01.0.png
output/pdf/...
output/paraview/...xdmf   (only when physics == 2D)
```


## Adding a new fracture model

Append to `tools.solvers.MODELS`:

```python
MODELS["my_model"] = {
    "w":   lambda a: a**3,
    "c_w": 4.0,
    "description": "...",
}
```

All problem and sweep files pick it up automatically.
