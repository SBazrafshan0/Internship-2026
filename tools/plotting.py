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


def _run_header(kind, ph, mdl, model_parameters, mesh_parameters,
                loading_parameters, amp):
    """Three-line figure header with model, loading and viscosity parameters."""
    m = model_parameters
    if amp == "U":
        amp_str = rf"$U_{{\max}}={loading_parameters['U_max']:.2f}$"
    else:
        amp_str = rf"$\theta_{{\max}}={loading_parameters['theta_max']:.2f}$"
    return (
        f"{kind} ({ph}, {mdl}):  "
        rf"$\hat\ell={m['l_hat']}$ | $\Lambda={m['Lambda']}$ | $\eta={m['eta']}$ | "
        rf"$E_{{ref}}={m.get('E_ref', 1.0):g}$ | $\nu={m.get('nu', 0.0):g}$" "\n"
        + amp_str + " | "
        rf"$N_{{QS}}={loading_parameters['N_steps_qs']}$ | "
        rf"$N_{{Dyn}}={loading_parameters['N_steps_dyn']}$ | "
        rf"$h/\hat\ell=1/{mesh_parameters['mesh_per_lhat']}$ | "
        rf"$T_0={loading_parameters['T0']}$" "\n"
        rf"viscosity:  $c_1={m.get('c1', 0.0):g}$ | "
        rf"$c_2={m.get('c2', 0.0):g}$ | $c_3={m.get('c3', 0.0):g}$"
    )


def _mark_crack_events(ax, events, color, tag):
    """Draw a dashed vertical line at each crack-nucleation generation, labelled
    with its generation index and connected-crack count."""
    if not events:
        return
    _, y1 = ax.get_ylim()
    for ev in events:
        ax.axvline(ev["load"], color=color, ls="--", lw=1.0, alpha=0.6)
        ax.annotate(f"{tag} G{ev['gen']}: {ev['n_cracks']} cr",
                    xy=(ev["load"], y1), xytext=(-3, -4),
                    textcoords="offset points", rotation=90,
                    ha="right", va="top", fontsize=7, color=color)


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
    gs  = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.2], hspace=0.30, wspace=0.30)
    ax_force  = fig.add_subplot(gs[0, 0])
    ax_energy = fig.add_subplot(gs[1, :])

    fig.suptitle(_run_header("Mechanical", ph, mdl, model_parameters,
                             mesh_parameters, loading_parameters, "U"),
                 fontsize=11, fontweight="bold", y=0.985)

    ax_force.plot(qs["U"],  qs["F"],  "k.-", lw=0.6, alpha=0.7, label="QS")
    ax_force.plot(dyn["U"], dyn["F"], "rx-", lw=0.6, alpha=0.7, label=fr"dyn, $\eta={eta}$")
    ax_force.set_xlabel(r"$\hat U(t)$"); ax_force.set_ylabel(r"reaction $\hat F$")
    ax_force.set_title("Force-displacement"); ax_force.grid(True, alpha=0.3); ax_force.legend()

    if ph == "1D":
        ax_dam = fig.add_subplot(gs[0, 1:])
        ix = np.argsort(x_alpha)
        ax_dam.plot(x_alpha[ix], alpha_qs_final[ix],  "k-",  label="QS")
        ax_dam.plot(x_alpha[ix], alpha_dyn_final[ix], "r--", label="Dynamic")
        ax_dam.set_xlabel(r"$\hat x$"); ax_dam.set_ylabel(r"$\alpha$")
        ax_dam.set_ylim(-0.05, 1.05); ax_dam.grid(True, alpha=0.3); ax_dam.legend()
        ax_dam.set_title(r"Final damage profile at $\hat U=\hat U_{\max}$")
    else:
        # 2D: colour maps of the final QS and dynamic damage, with scale bars.
        ax_dam_qs  = fig.add_subplot(gs[0, 1])
        ax_dam_dyn = fig.add_subplot(gs[0, 2])
        tri = result.get("triang")
        if tri is not None:
            tpc_qs  = ax_dam_qs.tripcolor(tri,  alpha_qs_final,  cmap="inferno",
                                          shading="gouraud", vmin=0, vmax=1)
            tpc_dyn = ax_dam_dyn.tripcolor(tri, alpha_dyn_final, cmap="inferno",
                                           shading="gouraud", vmin=0, vmax=1)
            fig.colorbar(tpc_qs,  ax=ax_dam_qs,  fraction=0.046, pad=0.04, label=r"$\alpha$")
            fig.colorbar(tpc_dyn, ax=ax_dam_dyn, fraction=0.046, pad=0.04, label=r"$\alpha$")
        for ax in (ax_dam_qs, ax_dam_dyn):
            ax.set_xlabel(r"$\hat x$"); ax.set_ylabel(r"$\hat y$")
            ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        ax_dam_qs.set_title(r"QS: final damage")
        ax_dam_dyn.set_title(r"Dynamic: final damage")

    ax_energy.plot(dyn["U"], dyn["K"],     "mx-", lw=0.6, alpha=0.7, label=r"$\hat K$ (dyn)")
    ax_energy.plot(qs["U"],  qs["P_el"],   "b.-", lw=0.6, alpha=0.7, label=r"$\hat P_{el}$ QS")
    ax_energy.plot(dyn["U"], dyn["P_el"],  "bx-", lw=0.6, alpha=0.7, label=r"$\hat P_{el}$ Dyn")
    ax_energy.plot(qs["U"],  qs["P_f"],    "g.-", lw=0.6, alpha=0.7, label=r"$\hat P_f$ QS")
    ax_energy.plot(dyn["U"], dyn["P_f"],   "gx-", lw=0.6, alpha=0.7, label=r"$\hat P_f$ Dyn")
    ax_energy.plot(qs["U"],  qs["S"],      "r.-", lw=0.6, alpha=0.7, label=r"$\hat S$ QS")
    ax_energy.plot(dyn["U"], dyn["S"],     "rx-", lw=0.6, alpha=0.7, label=r"$\hat S$ Dyn")
    if "D" in dyn and len(dyn["D"]) == len(dyn["U"]):
        ax_energy.plot(dyn["U"], dyn["D"], "yx-", lw=0.6, alpha=0.7, label=r"$\hat D$ (dissipated)")
    ax_energy.plot(qs["U"],  qs["total"],  "k.-", lw=0.6, alpha=0.7, label="Total QS")
    ax_energy.plot(dyn["U"], dyn["total"], "kx-", lw=0.6, alpha=0.7, label=r"Total Dyn ($K{+}P_{el}{+}P_f{+}S$)")
    ax_energy.set_xlabel(r"$\hat U(t)$"); ax_energy.set_ylabel("Energy")
    ax_energy.set_title("Energy evolution"); ax_energy.grid(True, alpha=0.3)
    ax_energy.legend(fontsize=9, ncol=2, loc="best")

    # Mark crack-nucleation generations (QS and dynamic).
    _mark_crack_events(ax_energy, result.get("qs_events", []),  "0.25",    "QS")
    _mark_crack_events(ax_energy, result.get("dyn_events", []), "tab:red", "dyn")
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])

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

    fig.suptitle(_run_header("Thermal", ph, mdl, model_parameters,
                             mesh_parameters, loading_parameters, "theta"),
                 fontsize=11, fontweight="bold", y=0.985)

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
    if "D" in dyn and len(dyn["D"]) == len(dyn["theta"]):
        ax_energy.plot(dyn["theta"], dyn["D"], "yx-", lw=0.6, alpha=0.7, label=r"$\hat D$ (dissipated)")
    ax_energy.plot(qs["theta"],  qs["total"],  "k.-", lw=0.6, alpha=0.7, label="Total QS")
    ax_energy.plot(dyn["theta"], dyn["total"], "kx-", lw=0.6, alpha=0.7, label="Total Dyn")
    ax_energy.set_xlabel(r"$\theta(t)$"); ax_energy.set_ylabel("Energy")
    ax_energy.set_title("Energy evolution"); ax_energy.grid(True, alpha=0.3)
    ax_energy.legend(fontsize=9, ncol=2, loc="best")

    # Mark crack-nucleation generations (QS and dynamic).
    _mark_crack_events(ax_energy, result.get("qs_events", []),  "0.25",    "QS")
    _mark_crack_events(ax_energy, result.get("dyn_events", []), "tab:red", "dyn")

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
            tpc_qs  = ax_frag_qs.tripcolor(tri,  last_qs,  cmap="inferno", shading="gouraud", vmin=0, vmax=1)
            tpc_dyn = ax_frag_dyn.tripcolor(tri, last_dyn, cmap="inferno", shading="gouraud", vmin=0, vmax=1)
            fig.colorbar(tpc_qs,  ax=ax_frag_qs,  fraction=0.046, pad=0.04, label=r"$\alpha$")
            fig.colorbar(tpc_dyn, ax=ax_frag_dyn, fraction=0.046, pad=0.04, label=r"$\alpha$")
        for ax in (ax_frag_qs, ax_frag_dyn):
            ax.set_aspect("equal"); ax.set_xlabel(r"$\hat x$"); ax.set_ylabel(r"$\hat y$")
            ax.grid(True, alpha=0.3)
        ax_frag_qs.set_title("QS: final damage")
        ax_frag_dyn.set_title("Dynamic: final damage")

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])

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
    tag: str = "",
):
    """
    Write a time series of ``alpha`` (and ``u``) to an XDMF file.

    Parameters
    ----------
    domain        : dolfinx.mesh.Mesh
    alpha_history : list of tuples ``(t, alpha_Function)``
    u_history     : list of tuples ``(t, u_Function)`` -- can be empty
    tag           : optional suffix appended to the filename stem (e.g. ``"QS"``
                    or ``"dyn"``) so quasi-static and dynamic series don't clash.
    """
    if not HAVE_FENICS:
        return None
    if not alpha_history and not u_history:
        return None
    paths = output_paths(output_dir)
    stem  = filename_stub(physics_type, model_parameters, mesh_parameters,
                          loading_parameters, solver_parameters)
    if tag:
        stem = f"{stem}_{tag}"
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
