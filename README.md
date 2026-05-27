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



## Installation

The code relies on **FEniCSx 0.7+** (`dolfinx`, `ufl`, `petsc4py`, `mpi4py`),
plus a handful of Python packages.  Pick the install path that matches your
operating system; *Linux / macOS via conda-forge is the recommended route*
and the one we test against.  Native Windows is not supported by FEniCSx --
use WSL2.

### Required packages (summary)

| Package        | Purpose                              |
|----------------|--------------------------------------|
| `fenics-dolfinx` | FEM (mesh, function spaces, forms) |
| `mpi4py`         | MPI bindings                       |
| `petsc4py`       | linear/nonlinear solvers (SNES)    |
| `python-gmsh`    | unstructured 2D triangulation      |
| `numpy`, `sympy`, `matplotlib` | numerics + plotting  |
| `joblib`         | parallel sweeps over CPU cores     |
| `tqdm`           | progress bars during long runs     |
| `jupyter`        | only for the theory notebooks      |

### A. Linux (Ubuntu / Debian / WSL2) -- conda-forge

```bash
# 1. miniconda / mambaforge if you don't have it yet:
#    https://github.com/conda-forge/miniforge

conda create -n fenicsx-env -c conda-forge \
    fenics-dolfinx mpich python-gmsh \
    numpy sympy matplotlib joblib tqdm jupyter
conda activate fenicsx-env

# 2. clone & run
git clone <this-repo> && cd fragmentation_repo
python problems/dynamic.py
```

### B. macOS -- conda-forge

Same command as A.  Use `openmpi` instead of `mpich` if Apple-Silicon gives
you trouble:

```bash
conda create -n fenicsx-env -c conda-forge \
    fenics-dolfinx openmpi python-gmsh \
    numpy sympy matplotlib joblib tqdm jupyter
conda activate fenicsx-env
```

### C. Windows -- WSL2 + conda-forge

FEniCSx does **not** run natively on Windows.  The recommended path is
WSL2 (Windows Subsystem for Linux):

1. Open PowerShell as administrator and install WSL Ubuntu:
   ```powershell
   wsl --install -d Ubuntu
   ```
2. Reboot, finish the Ubuntu setup, then **inside the Ubuntu shell** follow
   recipe A above.
3. Edit the repo from VS Code with the *"Remote -- WSL"* extension so that
   you keep a native-Windows editor while the code runs in Linux.

### D. Docker (any OS) -- one-shot

```bash
docker run -it --rm -v "$PWD":/work -w /work dolfinx/dolfinx:stable bash
# now inside the container:
pip install --break-system-packages joblib tqdm gmsh
python problems/dynamic.py
```

### E. HPC clusters

Most centres already provide a FEniCSx module.  Load it, then in your
user-site `pip install --user joblib tqdm gmsh`.  Launch single runs with
`srun python problems/dynamic.py`, sweeps with `python sweep.py` (joblib
will use the cores you allocated).

### Verifying the install

```bash
python -c "import dolfinx, ufl, mpi4py, petsc4py, gmsh, joblib, tqdm; \
           print('dolfinx', dolfinx.__version__)"
```

A line ending with `dolfinx 0.7.x` (or newer) means you're ready to run.


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
