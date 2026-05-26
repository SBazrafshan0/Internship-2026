"""
tools/plotting.py
=================
Post-processing helpers.

* :func:`plot_mechanical_run` -- 3-panel matplotlib figure (reaction, final
  damage profile, energy decomposition).  Saves both PNG and PDF; the file
  stem encodes the full parameter set (see :func:`tools.parameters.filename_stub`).
* :func:`plot_thermal_run`    -- analogous figure for the thermal test
  (mean stress, fragmentation profiles for QS and dynamic, energies).
* :func:`export_paraview`     -- writes ``alpha`` (and ``u`` when available)
  to an XDMF file that Paraview can open as a time series.  In 2D this is
  the recommended way to look at the crack pattern.
* :func:`output_paths`        -- builds the canonical
  ``output/png``, ``output/pdf``, ``output/paraview`` paths.

The 1D problem does not need Paraview, but the function still produces a
valid XDMF (Paraview will show the field as a line plot), so the same
post-processing call works in both dimensions.

The *fragmentation regime map* plot is **deliberately not included** in this
file -- as requested, it lives outside the core repository.
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .parameters import filename_stub

# Optional FEniCSx import -- only needed for ``export_paraview``.
try:
    from dolfinx import io
    from mpi4py import MPI
    HAVE_FENICS = True
except Exception:                                       # pragma: no cover
    HAVE_FENICS = False


# =============================================================================
# Filesystem
# =============================================================================
def output_paths(base_dir: str | Path) -> dict:
    """
    Return a dict of output sub-directories, creating them if needed.
    """
    base = Path(base_dir)
    paths = {
        "base":     base,
        "png":      base / "png",
        "pdf":      base / "pdf",
        "paraview": base / "paraview",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _save_fig(fig, png_dir, pdf_dir, stem: str):
    """Save the figure to both PNG (300 dpi) and PDF."""
    png_path = Path(png_dir) / f"{stem}.png"
    pdf_path = Path(pdf_dir) / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path,           bbox_inches="tight")
    plt.close(fig)
    return str(png_path), str(pdf_path)


# =============================================================================
# Mechanical
# =============================================================================
def plot_mechanical_run(
    result: dict,
    model_parameters: dict,
    mesh_parameters: dict,
    loading_parameters: dict,
    solver_parameters: dict,
    output_dir: str | Path,
) -> tuple[str, str]:
    """
    Three-panel figure for one mechanical run.

    ``result`` is the dictionary returned by ``problems.dynamic.run_problem``.
    """
    qs, dyn = result["qs"], result["dyn"]
    x_alpha = result["x_alpha"]
    alpha_qs_final  = result["alpha_qs_final"]
    alpha_dyn_final = result["alpha_dyn_final"]

    ph    = mesh_parameters["physics"]
    mdl   = solver_parameters["model"]
    l_hat = model_parameters["l_hat"]
    lam   = model_parameters["Lambda"]
    eta   = model_parameters["eta"]

    fig = plt.figure(figsize=(16, 9))
    gs  = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2], hspace=0.30, wspace=0.25)
    ax_force  = fig.add_subplot(gs[0, 0])
    ax_dam    = fig.add_subplot(gs[0, 1])
    ax_energy = fig.add_subplot(gs[1, :])

    header = (
        f"Mechanical ({ph}, {mdl}):  "
        rf"$\hat\ell={l_hat}$ | $\Lambda={lam}$ | $\eta={eta}$" "\n"
        rf"$U_{{\max}}={loading_parameters['U_max']:.2f}$ | "
        rf"$N_{{QS}}={loading_parameters['N_steps_qs']}$ | "
        rf"$N_{{Dyn}}={loading_parameters['N_steps_dyn']}$ | "
        rf"$N_x={mesh_parameters['nx']}$ | $T_0={loading_parameters['T0']}$"
    )
    fig.suptitle(header, fontsize=13, fontweight="bold", y=0.97)

    ax_force.plot(qs["U"],  qs["F"],  "k.-", lw=0.6, alpha=0.7, label="QS")
    ax_force.plot(dyn["U"], dyn["F"], "rx-", lw=0.6, alpha=0.7, label=fr"dyn, $\eta={eta}$")
    ax_force.set_xlabel(r"$\hat U(t)$"); ax_force.set_ylabel(r"reaction $\hat F$")
    ax_force.set_title("Force-displacement"); ax_force.grid(True, alpha=0.3); ax_force.legend()

    if ph == "1D":
        ix = np.argsort(x_alpha)
        ax_dam.plot(x_alpha[ix], alpha_qs_final[ix],  "k-",  label="QS")
        ax_dam.plot(x_alpha[ix], alpha_dyn_final[ix], "r--", label="Dynamic")
        ax_dam.set_xlabel(r"$\hat x$")
    else:
        # 2D: show a colour map of the final dynamic damage.
        tri = result.get("triang")
        if tri is not None:
            tpc = ax_dam.tripcolor(tri, alpha_dyn_final, cmap="inferno", shading="gouraud", vmin=0, vmax=1)
            fig.colorbar(tpc, ax=ax_dam, fraction=0.046, pad=0.04)
        ax_dam.set_xlabel(r"$\hat x$"); ax_dam.set_ylabel(r"$\hat y$")
        ax_dam.set_aspect("equal")
    ax_dam.set_title(r"Final damage at $\hat U=\hat U_{\max}$")
    ax_dam.grid(True, alpha=0.3)
    if ph == "1D":
        ax_dam.legend()

    ax_energy.plot(dyn["U"], dyn["K"],     "mx-", lw=0.6, alpha=0.7, label=r"$\hat K$ (dyn only)")
    ax_energy.plot(qs["U"],  qs["P_el"],   "b.-", lw=0.6, alpha=0.7, label=r"$\hat P_{el}$ QS")
    ax_energy.plot(dyn["U"], dyn["P_el"],  "bx-", lw=0.6, alpha=0.7, label=r"$\hat P_{el}$ Dyn")
    ax_energy.plot(qs["U"],  qs["P_f"],    "g.-", lw=0.6, alpha=0.7, label=r"$\hat P_f$ QS")
    ax_energy.plot(dyn["U"], dyn["P_f"],   "gx-", lw=0.6, alpha=0.7, label=r"$\hat P_f$ Dyn")
    ax_energy.plot(qs["U"],  qs["S"],      "r.-", lw=0.6, alpha=0.7, label=r"$\hat S$ QS")
    ax_energy.plot(dyn["U"], dyn["S"],     "rx-", lw=0.6, alpha=0.7, label=r"$\hat S$ Dyn")
    ax_energy.plot(qs["U"],  qs["total"],  "k.-", lw=0.6, alpha=0.7, label="Total QS")
    ax_energy.plot(dyn["U"], dyn["total"], "kx-", lw=0.6, alpha=0.7, label="Total Dyn (incl K)")
    ax_energy.set_xlabel(r"$\hat U(t)$"); ax_energy.set_ylabel("Energy")
    ax_energy.set_title("Energy evolution"); ax_energy.grid(True, alpha=0.3)
    ax_energy.legend(fontsize=9, ncol=2, loc="best")

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    paths = output_paths(output_dir)
    stem  = filename_stub("mechanical", model_parameters, mesh_parameters,
                          loading_parameters, solver_parameters)
    return _save_fig(fig, paths["png"], paths["pdf"], stem)


# =============================================================================
# Thermal
# =============================================================================
def plot_thermal_run(
    result: dict,
    model_parameters: dict,
    mesh_parameters: dict,
    loading_parameters: dict,
    solver_parameters: dict,
    output_dir: str | Path,
) -> tuple[str, str]:
    """
    Four-panel figure for one thermal run (mean stress, QS profiles,
    Dyn profiles, energies).  Saves PNG and PDF.
    """
    qs, dyn = result["qs"], result["dyn"]
    qs_snaps  = result["qs_snapshots"]
    dyn_snaps = result["dyn_snapshots"]
    x_alpha   = result["x_alpha"]

    ph    = mesh_parameters["physics"]
    mdl   = solver_parameters["model"]
    l_hat = model_parameters["l_hat"]
    lam   = model_parameters["Lambda"]
    eta   = model_parameters["eta"]

    fig = plt.figure(figsize=(16, 9))
    gs  = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.2], hspace=0.30, wspace=0.25)
    ax_force    = fig.add_subplot(gs[0, 0])
    ax_frag_qs  = fig.add_subplot(gs[0, 1])
    ax_frag_dyn = fig.add_subplot(gs[0, 2])
    ax_energy   = fig.add_subplot(gs[1, :])

    header = (
        f"Thermal ({ph}, {mdl}):  "
        rf"$\hat\ell={l_hat}$ | $\Lambda={lam}$ | $\eta={eta}$" "\n"
        rf"$\theta_{{\max}}={loading_parameters['theta_max']:.2f}$ | "
        rf"$N_{{QS}}={loading_parameters['N_steps_qs']}$ | "
        rf"$N_{{Dyn}}={loading_parameters['N_steps_dyn']}$ | "
        rf"$N_x={mesh_parameters['nx']}$ | $T_0={loading_parameters['T0']}$"
    )
    fig.suptitle(header, fontsize=13, fontweight="bold", y=0.97)

    ax_force.plot(qs["theta"],  qs["sigma_bar"],  "k.-", lw=0.6, alpha=0.7, label="QS")
    ax_force.plot(dyn["theta"], dyn["sigma_bar"], "rx-", lw=0.6, alpha=0.7, label="dynamic")
    ax_force.set_xlabel(r"$\theta(t)$"); ax_force.set_ylabel(r"mean stress $\bar\sigma$")
    ax_force.set_title("Mean stress vs thermal strain"); ax_force.grid(True, alpha=0.3); ax_force.legend()

    ax_energy.plot(dyn["theta"], dyn["K"],     "mx-", lw=0.6, alpha=0.7, label=r"$\hat K$ (dyn)")
    ax_energy.plot(qs["theta"],  qs["P_el"],   "b.-", lw=0.6, alpha=0.7, label=r"$\hat P_{el}$ QS")
    ax_energy.plot(dyn["theta"], dyn["P_el"],  "bx-", lw=0.6, alpha=0.7, label=r"$\hat P_{el}$ Dyn")
    ax_energy.plot(qs["theta"],  qs["P_f"],    "g.-", lw=0.6, alpha=0.7, label=r"$\hat P_f$ QS")
    ax_energy.plot(dyn["theta"], dyn["P_f"],   "gx-", lw=0.6, alpha=0.7, label=r"$\hat P_f$ Dyn")
    ax_energy.plot(qs["theta"],  qs["S"],      "r.-", lw=0.6, alpha=0.7, label=r"$\hat S$ QS")
    ax_energy.plot(dyn["theta"], dyn["S"],     "rx-", lw=0.6, alpha=0.7, label=r"$\hat S$ Dyn")
    ax_energy.plot(qs["theta"],  qs["total"],  "k.-", lw=0.6, alpha=0.7, label="Total QS")
    ax_energy.plot(dyn["theta"], dyn["total"], "kx-", lw=0.6, alpha=0.7, label="Total Dyn")
    ax_energy.set_xlabel(r"$\theta(t)$"); ax_energy.set_ylabel("Energy")
    ax_energy.set_title("Energy evolution"); ax_energy.grid(True, alpha=0.3)
    ax_energy.legend(fontsize=9, ncol=2, loc="best")

    if ph == "1D":
        ix = np.argsort(x_alpha)
        cmap_qs  = plt.cm.viridis
        cmap_dyn = plt.cm.plasma
        for k, snap in enumerate(qs_snaps):
            c = cmap_qs(k / max(1, len(qs_snaps) - 1))
            ax_frag_qs.plot(x_alpha[ix], snap["alpha"][ix], color=c, lw=1.6,
                            label=fr"$\theta={snap['theta']:.2f}$")
        for k, snap in enumerate(dyn_snaps):
            c = cmap_dyn(k / max(1, len(dyn_snaps) - 1))
            ax_frag_dyn.plot(x_alpha[ix], snap["alpha"][ix], color=c, lw=1.6,
                             label=fr"$\theta={snap['theta']:.2f}$")
        for ax in (ax_frag_qs, ax_frag_dyn):
            ax.set_ylim(-0.05, 1.05); ax.set_xlabel(r"$\hat x$"); ax.set_ylabel(r"$\alpha$")
            ax.grid(True, alpha=0.3); ax.legend(ncol=2, fontsize=6, loc="upper right")
        ax_frag_qs.set_title("QS: fragmentation generations")
        ax_frag_dyn.set_title("Dynamic: fragmentation generations")
    else:
        tri = result.get("triang")
        last_qs  = qs_snaps[-1]["alpha"] if qs_snaps  else result["alpha_qs_final"]
        last_dyn = dyn_snaps[-1]["alpha"] if dyn_snaps else result["alpha_dyn_final"]
        if tri is not None:
            ax_frag_qs.tripcolor(tri,  last_qs,  cmap="inferno", shading="gouraud", vmin=0, vmax=1)
            ax_frag_dyn.tripcolor(tri, last_dyn, cmap="inferno", shading="gouraud", vmin=0, vmax=1)
        for ax in (ax_frag_qs, ax_frag_dyn):
            ax.set_aspect("equal"); ax.set_xlabel(r"$\hat x$"); ax.set_ylabel(r"$\hat y$")
            ax.grid(True, alpha=0.3)
        ax_frag_qs.set_title("QS: final damage")
        ax_frag_dyn.set_title("Dynamic: final damage")

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    paths = output_paths(output_dir)
    stem  = filename_stub("thermal", model_parameters, mesh_parameters,
                          loading_parameters, solver_parameters)
    return _save_fig(fig, paths["png"], paths["pdf"], stem)


# =============================================================================
# Paraview export
# =============================================================================
def export_paraview(
    domain, alpha_history, u_history,
    physics_type: str,
    model_parameters: dict,
    mesh_parameters: dict,
    loading_parameters: dict,
    solver_parameters: dict,
    output_dir: str | Path,
):
    """
    Write a time series of ``alpha`` (and ``u``) to an XDMF file.

    Parameters
    ----------
    domain        : dolfinx.mesh.Mesh
    alpha_history : list of tuples ``(t, alpha_Function)``
    u_history     : list of tuples ``(t, u_Function)`` -- can be empty
    """
    if not HAVE_FENICS:
        return None
    paths = output_paths(output_dir)
    stem  = filename_stub(physics_type, model_parameters, mesh_parameters,
                          loading_parameters, solver_parameters)
    xdmf_path = Path(paths["paraview"]) / f"{stem}.xdmf"
    with io.XDMFFile(domain.comm, str(xdmf_path), "w") as xdmf:
        xdmf.write_mesh(domain)
        for t, alpha in alpha_history:
            xdmf.write_function(alpha, float(t))
        for t, u in u_history:
            xdmf.write_function(u, float(t))
    return str(xdmf_path)


def triangulation_from_domain(domain):
    """
    Build a ``matplotlib.tri.Triangulation`` from a 2D dolfinx mesh.  Used by
    the 2D plotters so that we can ``tripcolor`` the damage field directly.
    Returns ``None`` for non-2D meshes.
    """
    if domain.topology.dim != 2:
        return None
    from matplotlib.tri import Triangulation
    x = domain.geometry.x[:, 0]
    y = domain.geometry.x[:, 1]
    domain.topology.create_connectivity(2, 0)
    cells = domain.topology.connectivity(2, 0).array.reshape(-1, 3)
    return Triangulation(x, y, cells)
