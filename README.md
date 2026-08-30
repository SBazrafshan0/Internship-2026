# Variational gradient-damage fragmentation of a thin film on an elastic foundation

A brittle film bonded to a Winkler foundation, driven by a mechanical or a
thermal (eigenstrain) load, solved with FEniCSx/DOLFINx.

The film is modelled with a variational gradient-damage functional (AT1 or
AT2). What the foundation buys is the whole point: an unbonded film either
damages uniformly or breaks once, whereas a bonded one meets the damage
criterion everywhere at the same instant and so fragments into a pattern.
Both a quasi-static and a dynamic branch are solved for every problem, from
the same energy, so the two can be compared on the same configuration.

```
Internship-2026/
├── README.md
│
├── problems/                     <- ONE FILE PER PHYSICAL PROBLEM (each a
│   │                                 self-contained, runnable script)
│   ├── __init__.py                <- PROBLEMS dispatcher, used by sweep.py
│   ├── dynamic.py                 <- mechanical / dynamic test
│   ├── thermal.py                 <- thermal fragmentation test
│   ├── thermal_clamped.py         <- thermal bar, both ends held: the damage plateau
│   ├── fragmentation.py           <- parametric studies: spacing law, verification,
│   │                                  AT1/AT2 bar benchmark, crack-site non-uniqueness
│   ├── eta_convergence.py         <- dyn -> quasi-static convergence vs eta (H1 norms)
│   ├── secondary.py               <- semi-analytic secondary-cracking model (pure NumPy)
│   ├── secondary_wave.py          <- direct wave solver: real cracks, splitting
│   ├── secondary_fem.py           <- the wave model's initial-value problem in FEM
│   │                                  (pre-cracked, constant load); the FE regime map
│   ├── sweep.py                   <- parameter-sweep driver (joblib parallel)
│   └── *_theory.ipynb             <- theory + numerical scheme, one per problem
│
├── tools/                         <- LIBRARY CODE -- never edit during a run
│   ├── imports.py                  <- centralised third-party imports
│   ├── helpers.py                  <- SNES wrapper + AltMin loop + irreversibility
│   ├── parameters.py               <- default parameter dicts; also runnable
│   │                                  directly, see "How to run" below
│   ├── solvers.py                  <- AT1 / AT2 fracture-energy densities
│   ├── norms.py                    <- exact L2 / H1 norms of P1 fields
│   ├── meshing.py                  <- 1D interval / 2D Gmsh unstructured mesh
│   ├── plotting.py                 <- diagnostic run figures + Paraview XDMF export
│   ├── animations.py               <- three-panel run animations (schematic,
│   │                                  energy budget, damage field) as MP4
│   └── assets/                     <- the two 2D specimen panels animations.py draws
│
└── output/                        <- everything generated (git-ignored)
    ├── cache/                      <- raw sweep results (npz/pkl); a rerun reuses them
    ├── png/, pdf/, paraview/       <- diagnostic figures from problems/*.py runs
    │                                  (filename encodes the full parameter set)
    └── anim/                       <- the rendered MP4 animations
```

## How to run

### A. Straight from `tools/parameters.py`

The simplest way to launch a run: `tools/parameters.py` holds every default
parameter *and* a stand-alone entry point that reads them.

```bash
python tools/parameters.py
```

Edit the top of that file to choose what runs and with what:

```python
PROBLEM = "dynamic"   # "dynamic", "thermal", or "thermal_clamped"
```

and edit `DEFAULT_MODEL_PARAMETERS` / `DEFAULT_MESH_PARAMETERS` /
`DEFAULT_SOLVER_PARAMETERS` / `DEFAULT_MECH_LOADING` / `DEFAULT_THERM_LOADING`
for everything else (`l_hat`, `Lambda`, `eta`, `model`, `physics`,
`mesh_per_lhat`, `U_max`/`theta_max`, ...). Nothing needs to be edited inside
`problems/*.py` for a normal run.

**Stopping the load early without rescaling it — `U_target` / `theta_target`.**
The QS ramp is `U_max * t`; the dynamic ramp is the smoothed
`U_max * (tau/2)(1 + tanh(tau/T0))`, `tau = eta*t`. Both are usually run out
to their canonical end (`t=1` / `tau=1`, i.e. load = `U_max` / `~94.6% of
U_max`). Reducing `U_max` to stop a run earlier rescales the *whole* ramp
equation, not just where it stops. `U_target`/`theta_target` (default
`None`) solves that: set it to any value with `0 < target < U_max`, and both
branches independently invert their own ramp to find the pseudo-time at
which they reach exactly that load — so QS and dynamic stop at the *same*
physical load as each other, and `U_max`'s own shape/asymptote is untouched.
See the `_final_times()` docstring in `problems/dynamic.py` /
`problems/thermal.py` / `problems/thermal_clamped.py` for the derivation
(numerical inversion of the tanh ramp via `scipy.optimize.brentq`).

### B. A problem file directly

Each `problems/*.py` is also runnable on its own; with no `__main__`
overrides left in the files, this is equivalent to (A) above:

```bash
python problems/dynamic.py
python problems/thermal.py
python problems/thermal_clamped.py
```

The figure header and the output filename encode the full parameter set, so
you cannot silently overwrite a previous run.

### C. Parameter sweeps

`problems/sweep.py` farms many independent runs out to joblib (one CPU per
FEM problem). It runs **one problem at a time**, controlled by a single
constant at the top of the file:

```python
PROBLEM = "thermal"   # "dynamic" or "thermal", or any key of problems.PROBLEMS
```

```bash
python problems/sweep.py              # uses whatever PROBLEM is set to
python problems/sweep.py thermal      # one-off override of the default
```

To add a new problem `myproblem`:

1. Create `problems/myproblem.py` (copy `dynamic.py` as a template, change
   `_PROBLEM_SHAPE` for a different geometry).
2. Register it in `problems/__init__.py`:
   ```python
   from .myproblem import run_problem as run_myproblem
   PROBLEMS["myproblem"] = run_myproblem
   ```
3. Point `sweep.py`'s `PROBLEM` at it, or pass it on the command line.

### D. Parallelism

```bash
# inter-problem (sweeps): one joblib worker per FEM problem
COWORK_N_WORKERS=8 python problems/sweep.py

# intra-problem (one big FEM problem on multiple cores)
mpirun -n 4 python problems/dynamic.py
```

The two should *not* be combined. When the driver is launched under
`mpirun`, joblib is bypassed automatically and runs go serially per rank.

## ⚠ Set the thread limits

These are *small* problems (a few hundred degrees of freedom) solved *tens
of thousands* of times in a sweep. PETSc/MUMPS and the BLAS each default to
one thread per core and then spend all their time spin-waiting on barriers —
a sweep that takes minutes with one thread can take an hour with the
defaults on a many-core machine. Before any sweep:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## Installation

The code relies on **FEniCSx** (`dolfinx`, `ufl`, `petsc4py`, `mpi4py`;
developed and tested against `dolfinx 0.10`), plus a handful of Python
packages. *Linux / macOS via conda-forge is the recommended route.* Native
Windows is not supported by FEniCSx -- use WSL2.

### Required packages (summary)

| Package        | Purpose                              |
|----------------|---------------------------------------|
| `fenics-dolfinx` | FEM (mesh, function spaces, forms) |
| `mpi4py`         | MPI bindings                       |
| `petsc4py`       | linear/nonlinear solvers (SNES)    |
| `python-gmsh`    | unstructured 2D triangulation      |
| `numpy`, `sympy`, `scipy`, `matplotlib` | numerics + plotting |
| `joblib`         | parallel sweeps over CPU cores     |
| `tqdm`           | progress bars during long runs     |
| `jupyter`        | only for the theory notebooks      |

### A. Linux (Ubuntu / Debian / WSL2) -- conda-forge

```bash
# 1. miniconda / mambaforge if you don't have it yet:
#    https://github.com/conda-forge/miniforge

conda create -n fenicsx-env -c conda-forge \
    fenics-dolfinx mpich python-gmsh \
    numpy sympy scipy matplotlib joblib tqdm jupyter
conda activate fenicsx-env

# 2. clone & run
git clone <this-repo> && cd Internship-2026
python tools/parameters.py
```

### B. macOS -- conda-forge

Same command as A. Use `openmpi` instead of `mpich` if Apple-Silicon gives
you trouble:

```bash
conda create -n fenicsx-env -c conda-forge \
    fenics-dolfinx openmpi python-gmsh \
    numpy sympy scipy matplotlib joblib tqdm jupyter
conda activate fenicsx-env
```

### C. Windows -- WSL2 + conda-forge

FEniCSx does **not** run natively on Windows. The recommended path is WSL2
(Windows Subsystem for Linux):

1. Open PowerShell as administrator and install WSL Ubuntu:
   ```powershell
   wsl --install -d Ubuntu
   ```
2. Reboot, finish the Ubuntu setup, then **inside the Ubuntu shell** follow
   recipe A above.
3. Edit the repo from VS Code with the *"Remote -- WSL"* extension so you
   keep a native-Windows editor while the code runs in Linux.

### D. Docker (any OS) -- one-shot

```bash
docker run -it --rm -v "$PWD":/work -w /work dolfinx/dolfinx:stable bash
# now inside the container:
pip install --break-system-packages joblib tqdm gmsh scipy
python tools/parameters.py
```

### E. HPC clusters

Most centres already provide a FEniCSx module. Load it, then in your
user-site `pip install --user joblib tqdm gmsh scipy`. Launch single runs
with `srun python problems/dynamic.py`, sweeps with
`python problems/sweep.py` (joblib will use the cores you allocated).

### Verifying the install

```bash
python -c "import dolfinx, ufl, mpi4py, petsc4py, gmsh, joblib, tqdm, scipy; \
           print('dolfinx', dolfinx.__version__)"
```

A line printing your dolfinx version with no import error means you're ready
to run.

## Key conventions

### Mesh resolution -- single knob

The cell size is set from the regularisation length:

    h = l_hat / mesh_per_lhat

so `mesh_per_lhat = 5` gives 5 cells across one internal length. There is
**no** `nx`/`ny` to set by hand. In 2D, Gmsh generates an unstructured
triangulation with that characteristic length; when Gmsh is unavailable the
mesh factory falls back to a crossed-diagonal rectangle.

### Stopping criterion -- a fixed number of load steps

Both the QS and dynamic loops run a *fixed* number of steps
(`N_steps_qs`/`N_steps_dyn`) from `t=0` to the pseudo-time computed by
`_final_times()` (§ "How to run", part A) -- there is no `alpha`-based early
stop. Refining the load step (raising `N_steps_qs`/`N_steps_dyn`) is itself
one of the convergence checks, not a safety net.

### Geometry

The default geometry is a rectangle (`Lx`, `Ly` in `DEFAULT_MESH_PARAMETERS`,
`tools/parameters.py`). The *shape name* is **not** a user-facing knob: each
problem file pins its own shape via the module-level constant
`_PROBLEM_SHAPE`. Need a different geometry? Write a new problem file rather
than tweaking an existing one, and register a new builder in
`tools.meshing.GEOMETRY_BUILDERS`.

### Paraview output

`tools.plotting.export_paraview` writes an XDMF time series of `alpha` and
`u`. In 2D this is *the* way to look at the crack pattern. In **1D** the
output is redundant (matplotlib already shows the damage profile), so the
Paraview export is skipped automatically when `physics == "1D"`.

### Output file names

A run with `l_hat=0.02, Lambda=10, eta=0.01, AT1, 2D, mesh_per_lhat=5,
U_max=1.4, T0=0.7, N_qs=60, N_dyn=180` writes:

```
output/png/mechanical_2D_rectangle_AT1_lhat0.02_lam10.0_eta0.01_E1_nu0.3_c1..._umax1.40_nQS60_nDyn180_mpl5_T00.7.png
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

All problem and sweep files pick it up automatically through
`tools.solvers.get_model`.
