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
│   │                                PETSc, matplotlib, joblib, tqdm ...)
│   ├── helpers.py                <- SNESProblem + AltMin loop + mesh info
│   ├── parameters.py             <- default parameter dictionaries
│   ├── solvers.py                <- AT1, AT2 fracture-energy densities
│   ├── meshing.py                <- 1D interval / 2D Gmsh unstructured mesh
│   └── plotting.py               <- matplotlib + Paraview XDMF export
│
└── output/{png,pdf,paraview}/
```

## Single mesh-resolution knob: `mesh_per_lhat`

The mesh cell size is set automatically from the regularisation length:

    h = l_hat / mesh_per_lhat

There is **no** `nx`/`ny` to tweak by hand.  Pick `mesh_per_lhat = 4` to be
safe (four cells across a diffuse crack band), increase to 6-8 for
publication-quality patterns.

* 1D -- `dolfinx.mesh.create_interval` with `nx = ceil(Lx / h)`.
* 2D -- **Gmsh** generates an unstructured triangulation with characteristic
  length `h`.  If Gmsh is not importable, the factory falls back to a
  crossed-diagonal structured triangular mesh.

## Stopping criterion: damage at `alpha=1`

Both the QS loop and the dynamic loop are *while* loops that step until any
damage point reaches `alpha_break` (default 0.99) -- no fixed step count.
A safety cap `t_max` (default 3.0) prevents infinite loops in the
sub-critical regime where no crack ever nucleates.

The progress bar shows live `t`, `a_max`, `K` and AltMin iterations so you
can follow the run.

## Switches

Every problem file has, at the bottom, exactly the same block:

    cfg["solver_parameters"]["model"]      = "AT1" | "AT2"
    cfg["mesh_parameters"]["physics"]      = "1D"  | "2D"
    cfg["mesh_parameters"]["shape"]        = "rectangle"
    cfg["mesh_parameters"]["mesh_per_lhat"] = 5

Edit, save, run.  If a new problem cannot live under these switches, *write
a new file* in `problems/` rather than overloading an existing one.

## Adding a new shape

Append a Gmsh builder to `tools.meshing.GEOMETRY_BUILDERS`:

```python
GEOMETRY_BUILDERS["L_shape"] = lambda mp, lc: _gmsh_l_shape(mp, lc)
```

Use the same boundary-tag convention: 1=left, 2=right, 3=bottom, 4=top.
All problem files keep working as long as `mt.find(1)` and `mt.find(2)`
still mean the loaded/clamped edges.

## Adding a new model

Append to `tools.solvers.MODELS`:

```python
MODELS["my_model"] = {"w": lambda a: a**3, "c_w": 4.0, "description": "..."}
```

## How to run

```bash
# single run
python problems/dynamic.py
python problems/thermal.py

# sweep -- ONE problem at a time
python sweep.py dynamic
python sweep.py thermal

# parallel sweep
COWORK_N_WORKERS=8 python sweep.py thermal

# intra-problem MPI parallelism
mpirun -n 4 python problems/dynamic.py
```

## File-name convention

Outputs land in `output/{png,pdf,paraview}/` with a fully qualified stem:

```
mechanical_2D_rectangle_AT2_lhat0.02_lam10.0_eta0.01_umax1.40_nQS60_nDyn180_mpl5_T01.0.{png,pdf,xdmf}
```

so two runs cannot silently overwrite each other.

## Parallelism

* **MPI / FEniCSx** -- intra-problem domain decomposition.  Prepend
  `mpirun -n N` to the command.  Useful for big 2D meshes.
* **joblib** -- inter-problem parallelism in the sweep driver.  Each
  worker takes one independent FEM problem from the queue.  Override the
  worker count with `COWORK_N_WORKERS=<N>`.

The two should *not* be combined.  When `comm.size > 1`, joblib falls back
to a serial loop on each rank to avoid oversubscription.
