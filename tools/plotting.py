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
import math
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
    ax_force.set_title(r"Force-displacement: $\hat F$ vs $\hat U$")
    ax_force.grid(True, alpha=0.3); ax_force.legend()

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
        ax_dam_qs.set_title(r"QS: final damage $\alpha(\hat x,\hat y)$")
        ax_dam_dyn.set_title(r"Dynamic: final damage $\alpha(\hat x,\hat y)$")

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
    ax_energy.set_xlabel(r"$\hat U(t)$")
    ax_energy.set_ylabel(r"energy $\hat{\mathcal{E}}\in\{\hat K,\hat P_{el},\hat P_f,\hat S\}$")
    ax_energy.set_title(r"Energy evolution: $\hat{\mathcal{E}}$ vs $\hat U$")
    ax_energy.grid(True, alpha=0.3)
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
    ax_force.set_title(r"Mean stress: $\bar\sigma$ vs $\theta$")
    ax_force.grid(True, alpha=0.3); ax_force.legend()

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
    ax_energy.set_xlabel(r"$\theta(t)$")
    ax_energy.set_ylabel(r"energy $\hat{\mathcal{E}}\in\{\hat K,\hat P_{el},\hat P_f,\hat S\}$")
    ax_energy.set_title(r"Energy evolution: $\hat{\mathcal{E}}$ vs $\theta$")
    ax_energy.grid(True, alpha=0.3)
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
        ax_frag_qs.set_title(r"QS: fragmentation $\alpha(\hat x)$ per generation")
        ax_frag_dyn.set_title(r"Dynamic: fragmentation $\alpha(\hat x)$ per generation")
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
        ax_frag_qs.set_title(r"QS: final damage $\alpha(\hat x,\hat y)$")
        ax_frag_dyn.set_title(r"Dynamic: final damage $\alpha(\hat x,\hat y)$")

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])

    paths = output_paths(output_dir)
    stem  = filename_stub("thermal", model_parameters, mesh_parameters,
                          loading_parameters, solver_parameters)
    return _save_fig(fig, paths["png"], paths["pdf"], stem)


# =============================================================================
# Secondary cracking (semi-analytic, problems/secondary.py)
# =============================================================================
def plot_secondary_run(result: dict, output_dir: str | Path) -> tuple[str, str]:
    """
    Diagnostic figure for one semi-analytic secondary-cracking run
    (``problems.secondary.run_problem``).  The panels follow, *in order*, the
    "Semi-analytic list of questions" of the concept note:

    Step 1  Static release   -- (1a) relaxed strain profile e+(x);
                                (1b) released energy DeltaE_1/2 = tanh(Lambda).
    Step 2  Modal decomp.    -- coefficients A_n and energy quota Q_n vs
                                beta_n = q_n * ell_e.
    Step 3  Wave propagation -- reconstructed space-time strain e(x,t), with the
            & damping           c and c/sqrt(2) (group-velocity) characteristics.
    Step 4  Stress amplif.   -- (4a) R_crack(t) = max_x |e|/e_crit vs round-trips
                                with the exp(-gamma t) damping envelope (the
                                R_crack > 1 test); (4b) where: max_t |e|/e_crit
                                vs x.

    Step 5 (parameter map) is a *separate* figure: see
    :func:`plot_secondary_regime_map` / ``problems.secondary.trigger_map``.
    """
    p       = result["parameters"]
    x, t    = result["x"], result["t"]
    e_xt    = result["e_xt"]
    e_crit  = result["e_crit"]
    tau_rt  = result["tau_rt"]
    gam     = result["gamma_bar"]
    ell_e   = p["ell_e"]
    theta   = p["theta"]
    c       = p["c"]
    Lam     = p["Lambda_bar"]
    Gam     = p["Gamma"]
    l_hat   = result.get("l_hat", p.get("l_hat"))
    ell     = result.get("ell", ell_e / Lam)
    ell_d   = result.get("ell_d", (l_hat * ell if l_hat is not None else None))
    model   = p.get("model", "AT1")
    trig    = result["trigger"]
    x_excl  = result.get("x_excl", p["x_exclude"] * ell_e)
    ratio   = np.abs(e_xt) / e_crit

    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(
        rf"Secondary cracking (semi-analytic):  "
        rf"$\Lambda=\ell_e/\ell={Lam:g}$ | "
        rf"$\Gamma=\gamma\tau_{{rt}}={Gam:g}$ | "
        rf"$\hat\ell=\ell_d/\ell={l_hat:g}$ | "
        rf"$e_{{crit}}/\theta={e_crit/theta:g}$ ({model})"
        + ("   --   TRIGGERED" if trig else "   --   no secondary crack") + "\n"
        + rf"$\ell/\ell_e={ell/ell_e:g}$ | $\ell_d={ell_d:g}$ | "
        rf"$\gamma={gam:.3g}$ | $\tau_{{rt}}={tau_rt:.3g}$ | "
        rf"$\theta={theta:g}$ | $N_{{modes}}={p['n_modes']}$ | "
        rf"$N_{{rt}}={p['n_roundtrips']:g}$",
        fontsize=11, fontweight="bold")

    # ---- Step 1 (a): static release, relaxed strain profile -----------------
    ax = axs[0, 0]
    ax.plot(x / ell_e, result["e_plus"] / theta, "k-", lw=1.5,
            label=r"$e^+(x)/\theta$ (relaxed)")
    # PDF convention e = u_x - theta: pre-stress is e_0 = -theta, e^+ -> -theta
    # far from the crack and e^+(0)=0; the |e|>=e_crit test is symmetric.
    ax.axhline(-1.0, color="b", ls=":", lw=1.0, label=r"pre-stress $e_0=-\theta$")
    ax.axhline(e_crit / theta, color="r", ls="--", lw=1.0,
               label=r"$\pm e_{crit}/\theta$")
    ax.axhline(-e_crit / theta, color="r", ls="--", lw=1.0)
    ax.set_xlabel(r"$x/\ell_e$"); ax.set_ylabel(r"$e/\theta$")
    ax.set_title("Step 1 - Static release: relaxed strain $e^+(x)$")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # ---- Step 1 (b): released energy ----------------------------------------
    # With Lambda = ell_e/ell the released energy is
    #   dE_1/2 / (1/2 E_h theta^2 ell_e) = tanh(ell/ell_e) = tanh(1/Lambda).
    ax = axs[0, 1]
    Lam_axis = np.linspace(max(1e-2, 0.1 * Lam), max(2.0 * Lam, 3.0), 300)
    ax.plot(Lam_axis, np.tanh(1.0 / Lam_axis), "k-", lw=1.5,
            label=r"$\Delta E_{1/2} / (\frac{1}{2} E_h\theta^2\ell_e)"
                  r" = \tanh(1/\Lambda)$")
    ax.plot(Lam_axis, np.clip(1.0 / Lam_axis, None, 1.5), "b:", lw=1.0,
            label=r"$\sim 1/\Lambda$  ($\ell\ll\ell_e$)")
    ax.axhline(1.0, color="g", ls=":", lw=1.0,
               label=r"$\sim 1$  ($\ell\gg\ell_e$)")
    ax.plot(Lam, np.tanh(1.0 / Lam), "r*", ms=14, mec="k", label="this run")
    ax.set_ylim(0, 1.6)
    ax.set_xlabel(r"$\Lambda=\ell_e/\ell$")
    ax.set_ylabel(r"$\Delta E_{1/2}/(\frac{1}{2} E_h\theta^2\ell_e)$")
    ax.set_title(r"Step 1 - Released energy: $\Delta E_{1/2}=\tanh(1/\Lambda)$")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # ---- Step 2: modal decomposition (coefficients A_n and quota Q_n) -------
    ax = axs[0, 2]
    beta = result["beta_n"]
    a0n = np.abs(result["a0"]) / np.max(np.abs(result["a0"]))
    l1 = ax.semilogx(beta, result["Q_n"], "ko-", ms=3, lw=0.8,
                     label=r"$Q_n$ (energy quota)")
    ax.set_xlabel(r"$\beta_n = q_n \ell_e$")
    ax.set_ylabel(r"$Q_n$")
    ax.axvline(1.0, color="r", ls="--", lw=1.0, label=r"$\beta_n = 1$")
    axb = ax.twinx()
    l2 = axb.semilogx(beta, a0n, "C0.", ms=3, alpha=0.6,
                      label=r"$|A_n|$ (normalised)")
    l3 = axb.semilogx(beta, 1.0 / (1.0 + beta ** 2), "C0--", lw=1.0,
                      label=r"$1/(1+\beta_n^2)$")
    axb.set_ylabel(r"$|A_n|$", color="C0")
    axb.tick_params(axis="y", labelcolor="C0")
    ax.legend(handles=[l1[0], l2[0], l3[0]], fontsize=8, loc="upper right")
    ax.set_title("Step 2 - Modal decomposition: $A_n$, $Q_n$")
    ax.grid(True, alpha=0.3, which="both")

    # ---- Step 3: wave propagation & damping (space-time reconstruction) -----
    ax = axs[1, 0]
    pcm = ax.pcolormesh(t / tau_rt, x / ell_e, ratio,
                        cmap="inferno", shading="auto",
                        vmin=0.0, vmax=max(1.2, ratio.max()))
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04,
                 label=r"$|e(x,t)|/e_{crit}$")
    if ratio.max() >= 1.0:
        ax.contour(t / tau_rt, x / ell_e, ratio, levels=[1.0],
                   colors="cyan", linewidths=1.2)
    ax.plot(t / tau_rt, c * t / ell_e, "w--", lw=0.8, alpha=0.7)
    ax.plot(t / tau_rt, (c / np.sqrt(2.0)) * t / ell_e, "w:", lw=0.8,
            alpha=0.7)
    ax.axhline(x_excl / ell_e, color="0.6", ls="-", lw=0.8)
    if trig:
        ax.plot(trig["t"] / tau_rt, trig["x"] / ell_e, "c*", ms=14,
                mec="k", label="secondary crack")
        ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, x.max() / ell_e)
    ax.set_xlabel(r"$t/\tau_{rt}$"); ax.set_ylabel(r"$x/\ell_e$")
    ax.set_title("Step 3 - Wave propagation & damping: $e(x,t)$")

    # ---- Step 4 (a): stress amplification R_crack(t) (the R_crack>1 test) ---
    ax = axs[1, 1]
    mask = x >= x_excl
    hist = np.abs(e_xt[mask, :]).max(axis=0) / e_crit
    ax.plot(t / tau_rt, hist, "k-", lw=1.2,
            label=r"$\mathcal{R}_{crack}(t)=\max_x |e|/e_{crit}$")
    env = hist[0] * np.exp(-gam * t) if hist[0] > 0 else np.exp(-gam * t)
    ax.plot(t / tau_rt, env, "b:", lw=1.0, label=r"$\propto e^{-\gamma t}$")
    ax.axhline(1.0, color="r", ls="--", lw=1.0,
               label=r"$\mathcal{R}_{crack}=1$")
    for k in range(1, int(np.floor(t[-1] / tau_rt)) + 1):
        ax.axvline(k, color="0.7", ls=":", lw=0.8)
    if trig:
        ax.plot(trig["t"] / tau_rt, trig["e"] / e_crit, "c*", ms=14, mec="k",
                label=fr"trigger, $t={trig['t']/tau_rt:.2f}\,\tau_{{rt}}$")
    ax.set_xlabel(r"$t/\tau_{rt}$")
    ax.set_ylabel(r"$\mathcal{R}_{crack}=\max_x |e|/e_{crit}$")
    ax.set_title(r"Step 4 - Stress amplification: $\mathcal{R}_{crack}(t)$")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # ---- Step 4 (b): where the amplification peaks --------------------------
    ax = axs[1, 2]
    ax.plot(x / ell_e, ratio.max(axis=1), "k-", lw=1.5,
            label=r"$\max_t |e(x,t)|/e_{crit}$")
    ax.axhline(1.0, color="r", ls="--", lw=1.0, label="threshold")
    ax.axvspan(0.0, x_excl / ell_e, color="0.85",
               label="excluded (first crack)")
    if trig:
        ax.axvline(trig["x"] / ell_e, color="c", ls="-.", lw=1.2)
    ax.set_xlabel(r"$x/\ell_e$"); ax.set_ylabel(r"$\max_t |e|/e_{crit}$")
    ax.set_title(r"Step 4 - Stress amplification: $\max_t |e|/e_{crit}$ vs $x/\ell_e$")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    paths = output_paths(output_dir)
    stem = (f"secondary_run_Lam{Lam:g}_Gam{Gam:g}_lhat{l_hat:g}"
            f"_ecrit{e_crit / theta:g}_{model}_nm{p['n_modes']}"
            f"_nrt{p['n_roundtrips']:g}")
    return _save_fig(fig, paths["png"], paths["pdf"], stem)


def _secondary_base_str(base: dict) -> str:
    """One-line summary of the fixed parameters of a regime sweep."""
    return (rf"$e_{{crit}}$ from {base.get('model', 'AT1')}, "
            rf"$G_c={base.get('Gc', 1.0):g}$ | "
            rf"$N_{{modes}}={base.get('n_modes', '?')}$, "
            rf"$N_{{rt}}={base.get('n_roundtrips', 4.0):g}$ | "
            rf"$\theta={base.get('theta', 1.0):g}$")


def plot_secondary_regime_map(sweep: dict,
                              output_dir: str | Path) -> tuple[str, str]:
    """
    Regime map in the ``(Lambda, Gamma)`` plane at a *single* ``l_hat`` from
    ``problems.secondary.trigger_map``: colour = trigger margin
    ``max|e| / e_crit``; the level-1 contour separates *secondary cracking*
    from *wave damped out*.  ``sweep["margin"]`` is 2-D and ``sweep["l_hat"]``
    is the scalar internal-length ratio of this map (put in the title and the
    file name so the nine maps do not overwrite each other).
    """
    Lam  = np.asarray(sweep["Lambda_bar"])
    Gam  = np.asarray(sweep["Gamma"])
    M    = np.asarray(sweep["margin"])
    lh   = float(np.asarray(sweep["l_hat"]).ravel()[0])
    base = sweep.get("base_parameters", {})

    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.pcolormesh(Lam, Gam, M, cmap="RdBu_r", shading="auto",
                        vmin=2.0 - M.max() if M.max() > 1 else None,
                        vmax=M.max())
    fig.colorbar(pcm, ax=ax, label=r"trigger margin $\max|e|/e_{crit}$")
    if M.min() < 1.0 < M.max():
        cs = ax.contour(Lam, Gam, M, levels=[1.0], colors="k",
                        linewidths=1.8)
        ax.clabel(cs, fmt={1.0: "secondary-cracking boundary"}, fontsize=8)
    ax.set_xlabel(r"$\Lambda = \ell_e/\ell$")
    ax.set_ylabel(r"$\Gamma = \gamma\,\tau_{rt}$")
    ax.set_title(rf"Step 5 - Parameter map at $\hat\ell=\ell_d/\ell={lh:g}$"
                 "\n" r"(margin $\mathcal{R}_{crack}\geq 1$: the released wave "
                 r"re-cracks the film)   --   " + _secondary_base_str(base))
    plt.tight_layout()

    paths = output_paths(output_dir)
    stem = (f"secondary_map_lhat{lh:g}"
            f"_Lam{Lam.min():g}-{Lam.max():g}"
            f"_Gam{Gam.min():g}-{Gam.max():g}"
            f"_{base.get('model', 'AT1')}_nm{base.get('n_modes', 'NA')}")
    return _save_fig(fig, paths["png"], paths["pdf"], stem)


def plot_secondary_regime_summary(sweep: dict,
                                  output_dir: str | Path) -> tuple[str, str]:
    """
    Paper-style summary of the full ``(Lambda, Gamma, l_hat)`` sweep from
    ``problems.secondary.trigger_map``.  A single ``(Lambda, Gamma)`` axis
    carries the ``R_crack = 1`` secondary-cracking boundary for **every**
    ``l_hat``, coloured by ``l_hat`` -- so the reader sees at a glance how the
    third parameter shifts the boundary.  ``sweep["margin"]`` is the 3-D array
    ``margin[h, i, j]`` (indices ``l_hat, Gamma, Lambda``).
    """
    Lam    = np.asarray(sweep["Lambda_bar"])
    Gam    = np.asarray(sweep["Gamma"])
    l_hats = np.atleast_1d(np.asarray(sweep["l_hat"], dtype=float))
    M      = np.asarray(sweep["margin"])
    base   = sweep.get("base_parameters", {})

    import matplotlib as mpl
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.LogNorm(vmin=l_hats.min(), vmax=l_hats.max()) \
        if l_hats.min() > 0 and l_hats.max() > l_hats.min() \
        else mpl.colors.Normalize(vmin=l_hats.min(), vmax=l_hats.max() + 1e-12)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    n_drawn = 0
    for h, lh in enumerate(l_hats):
        Mh = M[h]
        if Mh.min() < 1.0 < Mh.max():
            ax.contour(Lam, Gam, Mh, levels=[1.0],
                       colors=[cmap(norm(lh))], linewidths=2.0)
            n_drawn += 1

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=r"$\hat\ell = \ell_d/\ell$")

    ax.set_xlim(Lam.min(), Lam.max())
    ax.set_ylim(Gam.min(), Gam.max())
    ax.set_xlabel(r"$\Lambda = \ell_e/\ell$")
    ax.set_ylabel(r"$\Gamma = \gamma\,\tau_{rt}$")
    note = (r"each curve: $\mathcal{R}_{crack}=1$ boundary for one $\hat\ell$"
            if n_drawn else
            r"no boundary in range (all margins on one side of 1)")
    ax.set_title(r"Secondary-cracking boundaries in the $(\Lambda,\Gamma)$ "
                 r"plane" "\n"
                 "below/left of a curve: secondary cracking; above/right: "
                 "wave damped out\n"
                 f"({note})   --   " + _secondary_base_str(base),
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    paths = output_paths(output_dir)
    stem = (f"secondary_summary_lhat{l_hats.min():g}-{l_hats.max():g}"
            f"_Lam{Lam.min():g}-{Lam.max():g}"
            f"_Gam{Gam.min():g}-{Gam.max():g}"
            f"_{base.get('model', 'AT1')}_nm{base.get('n_modes', 'NA')}")
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
        if u_history:
            for (t, alpha), (_, u) in zip(alpha_history, u_history):
                xdmf.write_function(alpha, float(t))
                xdmf.write_function(u, float(t))
        else:
            for t, alpha in alpha_history:
                xdmf.write_function(alpha, float(t))
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


# =============================================================================
# problems/eta_convergence.py -- norm-based dynamic -> quasi-static study
# =============================================================================
_ETA_LOAD_LABEL = {"mechanical": r"imposed displacement $\hat U$",
                   "thermal":    r"thermal load $\theta$"}


_ETA_NORM_STYLE = {"u_L2": ("tab:blue",   "o", r"$\|u_{dyn}-u_{qs}\|_{L^2}$"),
          "u_H1": ("tab:cyan",   "s", r"$\|u_{dyn}-u_{qs}\|_{H^1}$"),
          "a_L2": ("tab:orange", "^", r"$\|\alpha_{dyn}-\alpha_{qs}\|_{L^2}$"),
          "a_H1": ("tab:red",    "v", r"$\|\alpha_{dyn}-\alpha_{qs}\|_{H^1}$"),
          "y_H1": ("k",          "D", r"$\|y_{dyn}-y_{qs}\|_{H^1\times H^1}$")}


def plot_eta_norm_study(study: dict, output_dir, verbose: bool = True):
    out_dir = Path(output_dir)
    physics, etas, grid = study["physics"], study["etas"], study["grid"]
    label = _ETA_LOAD_LABEL[physics]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # -- left: the state distance along the loading path, one curve per eta ---
    cmap = plt.cm.viridis(np.linspace(0.12, 0.92, len(etas)))
    for k, (eta, c) in enumerate(zip(etas, cmap)):
        axL.plot(grid, study["dist"]["y_H1"][k] / study["scale"]["y_H1"],
                 color=c, lw=1.7, label=fr"$\eta={eta:g}$")
    axL.set_xlabel(label)
    axL.set_ylabel(r"$\|y_{dyn}-y_{qs}\|_{H^1\times H^1}\,/\,\max\|y_{qs}\|$")
    axL.set_title(f"{physics} ($\\Lambda={study['Lambda']:g}$): state distance "
                  "along the loading path")
    axL.set_yscale("log"); axL.grid(alpha=0.3, which="both")
    axL.legend(fontsize=8, ncol=2)

    axL.axvline(study["load_crack"], color="r", ls="--", lw=1.4)
    axL.text(study["load_crack"], axL.get_ylim()[1], " crack", color="r",
             fontsize=9, va="top")

    # -- right: PRE-CRACK convergence -- every norm falls together -----------
    # A metric that is IDENTICALLY ZERO cannot be drawn on a log axis, and it is
    # not a failure: with AT1 the damage field is exactly zero in BOTH branches
    # throughout the elastic phase, so ||alpha_dyn - alpha_qs|| == 0 there.  Say
    # so in words instead of plotting an invisible line labelled "order nan".
    vanishing = []
    for m in study["metrics"]:
        col, mk, lab = _ETA_NORM_STYLE[m]
        y = np.asarray(study["pre_sup"][m], float)
        if not np.any(y > 0):
            vanishing.append(lab)
            continue
        axR.plot(etas, y, marker=mk, color=col,
                 lw=2.0 if m == "y_H1" else 1.3,
                 ms=7 if m == "y_H1" else 5,
                 label=lab + fr"  (order {study['order'][m]:.2f})")
    if vanishing:
        axR.text(0.03, 0.03,
                 "identically zero on the elastic branch\n(AT1: $\\alpha\\equiv0$ in both):\n"
                 + "\n".join(vanishing),
                 transform=axR.transAxes, fontsize=8, va="bottom", ha="left",
                 bbox=dict(fc="white", ec="0.7", alpha=0.9))
    e = np.array(etas)
    axR.plot(e, study["pre_sup"]["y_H1"][0] * (e / e[0]), "k:", lw=1.2,
             label=r"first order in $\eta$")
    axR.set_xscale("log"); axR.set_yscale("log"); axR.invert_xaxis()
    axR.set_xlabel(r"loading time-scale $\eta$   (slower loading $\rightarrow$)")
    axR.set_ylabel("relative distance  (sup over the ELASTIC part of the path)")
    axR.set_title("Before the crack: clean convergence in every norm\n"
                  "(equivalent norms $\\Rightarrow$ same order)")
    axR.grid(alpha=0.3, which="both"); axR.legend(fontsize=8)

    fig.suptitle(r"Dynamic $\rightarrow$ quasi-static limit measured with Sobolev norms "
                 r"of the state $y=(u,\alpha)$, not with energies", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("png", "pdf"):
        f = out_dir / f"eta_norm_{physics}.{ext}"
        fig.savefig(f, dpi=150, bbox_inches="tight")
        if verbose:
            print(f"  saved {f}")
    save_final_figure(fig, out_dir, f"eta_norm_{physics}", verbose=verbose)
    plt.show()
    plt.close(fig)


# =============================================================================
# problems/secondary_wave.py -- direct wave solver with cracking and splitting
# =============================================================================
_WAVE_GEN_COLORS = ["#00e5ff", "#7CFC00", "#FFD400", "#FF7A00", "#FF2D55", "#FF00FF"]


def _wave_panel_title(r):
    n, g = r["n_secondary"], r["n_generations"]
    if n == 0:
        tag = "no secondary crack"
    elif g <= 1:
        tag = f"{n} secondary crack" + ("s" if n > 1 else "")
    else:
        tag = f"cascade: {n} cracks, {g} generations"
    return (fr"$\Lambda$={r['Lambda_bar']:.2f}, $\Gamma$={r['Gamma']:.2f}" "\n" + tag)


def plot_wave_spacetime(runs, output_dir, fname="secondary_wave_spacetime",
                          row_labels=("fixed $\\Lambda$: increasing damping $\\Gamma$",
                                      "fixed $\\Gamma$: increasing $\\Lambda$ (shorter film)"),
                          verbose=True):
    """2x3 space-time maps of ``|e(x,t)|`` with every crack marked.

    ``runs`` is a list of six results: the first three form the top row, the
    last three the bottom row.  All panels share one colour scale.
    """
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    vmax = max(np.abs(r["e_xt"]).max() for r in runs)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5), constrained_layout=True)
    for ax, r in zip(axes.ravel(), runs):
        X = r["x_el"] / r["ell_e"]
        T = r["t_snap"] / r["tau_rt"]
        pc = ax.pcolormesh(X, T, np.abs(r["e_xt"]).T, shading="auto", cmap="magma",
                           vmin=0.0, vmax=vmax, rasterized=True)
        ax.contour(X, T, np.abs(r["e_xt"]).T, levels=[r["e_crit"]],
                   colors="w", linewidths=0.9, alpha=0.65)
        # every crack: it exists from its birth time onwards and blocks the wave
        for cr in r["cracks"]:
            xg, tg, gg = cr["x"] / r["ell_e"], cr["t"] / r["tau_rt"], cr["gen"]
            col = _WAVE_GEN_COLORS[min(gg, len(_WAVE_GEN_COLORS) - 1)]
            ax.plot([xg, xg], [tg, T[-1]], color=col, lw=1.4, ls="--", alpha=0.95)
            if gg > 0:
                ax.plot(xg, tg, "o", ms=7, mfc=col, mec="k", mew=0.8, zorder=6)
        ax.set_xlim(X[0], X[-1]); ax.set_ylim(T[0], T[-1])
        ax.set_title(_wave_panel_title(r), fontsize=10)
        ax.set_xlabel(r"$x/\ell_e$")
    for i, lab in enumerate(row_labels):
        axes[i, 0].set_ylabel(lab + "\n" + r"$t/\tau_{rt}$ (round-trips)", fontsize=9)
    fig.colorbar(pc, ax=axes, label=r"$|e(x,t)|$   (one scale for all panels)",
                 shrink=0.85)
    fig.suptitle("Release waves with real cracking and domain splitting: each new crack "
                 "is a new free face that launches its own wave\n"
                 "(dashed lines = cracks, coloured by generation; markers = birth)",
                 fontsize=12)
    paths = []
    for ext in ("png", "pdf"):
        p = out / f"{fname}.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight"); paths.append(p)
    save_final_figure(fig, out, fname, verbose=False)
    if verbose:
        print("saved", paths[0], f"| common colour scale 0 .. {vmax:.3f}")
    return fig


def plot_wave_energy(runs, output_dir, fname="secondary_wave_energy",
                       verbose=True):
    """2x3 energy histories matching :func:`plot_wave_spacetime`."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.5), sharex=True,
                             constrained_layout=True)
    for ax, r in zip(axes.ravel(), runs):
        h, T = r["hist"], r["hist"]["t"] / r["tau_rt"]
        ax.plot(T, h["P_el"], color="tab:blue",   lw=1.8, label=r"elastic $\mathcal{P}_{el}$")
        ax.plot(T, h["P_f"],  color="tab:green",  lw=1.4, label=r"foundation $\mathcal{P}_f$")
        ax.plot(T, h["K"],    color="tab:red",    lw=1.4, label=r"kinetic $\mathcal{K}$")
        ax.plot(T, h["S"],    color="tab:purple", lw=1.4, label=r"fracture $\mathcal{S}$")
        ax.plot(T, h["D"],    color="tab:orange", lw=1.4, ls="--", label=r"dissipated $\mathcal{D}$")
        ax.plot(T, h["total"], "k-", lw=2.2, label="total (should be conserved)")
        for cr in r["cracks"][1:]:
            ax.axvline(cr["t"] / r["tau_rt"],
                       color=_WAVE_GEN_COLORS[min(cr["gen"], len(_WAVE_GEN_COLORS) - 1)],
                       lw=1.0, ls=":", alpha=0.9)
        ax.set_title(_wave_panel_title(r), fontsize=10)
        ax.grid(alpha=0.3)
    for a in axes[1, :]:
        a.set_xlabel(r"$t/\tau_{rt}$ (round-trips)")
    for a in axes[:, 0]:
        a.set_ylabel("energy")
    axes[0, 2].legend(fontsize=7.5, loc="center right")
    fig.suptitle("Energy budget in time: every crack converts stored elastic energy into "
                 "kinetic energy and new surface\n"
                 "(vertical dotted lines = crack events, coloured by generation)",
                 fontsize=12)
    paths = []
    for ext in ("png", "pdf"):
        p = out / f"{fname}.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight"); paths.append(p)
    save_final_figure(fig, out, fname, verbose=False)
    if verbose:
        print("saved", paths[0])
    return fig


# =============================================================================
# Animation: watch the wave travel, cross the threshold, and split the film
# =============================================================================
def _wave_broken_mask(res, t_now):
    """Boolean mask of the elements already cracked at time ``t_now``."""
    x_el = res["x_el"]
    broken = np.zeros(x_el.size, bool)
    for cr in res["cracks"]:
        if cr["t"] <= t_now and cr["x"] > 0.0:
            broken[int(np.argmin(np.abs(x_el - cr["x"])))] = True
    return broken


def animate_wave_run(res, output_dir, fname="secondary_wave_movie",
                     n_frames=170, fps=8, dpi=95, verbose=True, fmt="gif",
                     figsize=(10.5, 7.4)):
    """Animate one run: the film strip, the strain profile against ``e_crit``
    and the energy budget, with the over-threshold region highlighted.

    Three stacked panels, all sharing the same instant:

    1. **the film** -- a strip coloured by ``|e|``; a cracked element is drawn
       black, so the fragmentation is literally visible as the strip breaking
       into pieces;
    2. **the strain profile** ``|e(x)|`` against the threshold ``e_crit``.  The
       area above the threshold is filled **red** and the peak is marked, so
       the instant a point goes critical is unmistakable;
    3. **the energy budget** with a moving time cursor.
    """
    import matplotlib as mpl
    from matplotlib.animation import FuncAnimation, PillowWriter
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

    x   = res["x_el"] / res["ell_e"]
    T   = res["t_snap"] / res["tau_rt"]
    AE  = np.abs(res["e_xt"])
    ec  = res["e_crit"]
    idx = np.linspace(0, T.size - 1, min(n_frames, T.size)).astype(int)

    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(3, 1, height_ratios=[1.0, 2.6, 2.0], hspace=0.55)
    ax_bar, ax_pro, ax_en = (fig.add_subplot(gs[i]) for i in range(3))

    # ---- panel 1: the film strip -------------------------------------------
    cmap = mpl.colormaps["magma"].copy()
    cmap.set_bad("black")                       # cracked elements go black
    vmax = max(AE.max(), ec * 1.1)
    im = ax_bar.imshow(AE[:, 0][None, :], aspect="auto", cmap=cmap,
                       vmin=0.0, vmax=vmax,
                       extent=[x[0], x[-1], 0, 1], interpolation="nearest")
    ax_bar.set_yticks([]); ax_bar.set_xlim(x[0], x[-1])
    ax_bar.set_xlabel(r"$x/\ell_e$", fontsize=8, labelpad=1)
    bar_marks = []
    ax_bar.set_title("the film  (colour = $|e|$, black = crack)", fontsize=10)
    cb = fig.colorbar(im, ax=ax_bar, pad=0.01, fraction=0.05)
    cb.ax.set_title(r"$|e|$", fontsize=8, pad=4)

    # ---- panel 2: the strain profile vs the threshold ----------------------
    (ln,) = ax_pro.plot(x, AE[:, 0], color="tab:blue", lw=2.0, zorder=3)
    ax_pro.axhline(ec, color="red", ls="--", lw=1.8, zorder=2,
                   label=fr"threshold $e_{{crit}}={ec:g}$")
    ax_pro.axhline(res["theta"], color="grey", ls=":", lw=1.2, zorder=2,
                   label=r"pre-stress $\theta$")
    fill = [ax_pro.fill_between(x, ec, AE[:, 0], where=AE[:, 0] >= ec,
                                color="red", alpha=0.55, zorder=1)]
    (pk,) = ax_pro.plot([], [], "o", ms=11, mfc="yellow", mec="k", mew=1.2, zorder=5)
    crack_lines = []
    ax_pro.set_xlim(x[0], x[-1]); ax_pro.set_ylim(0, vmax * 1.08)
    ax_pro.set_xlabel(r"$x/\ell_e$"); ax_pro.set_ylabel(r"$|e(x,t)|$")
    ax_pro.grid(alpha=0.3); ax_pro.legend(fontsize=8, loc="upper left")
    banner = ax_pro.text(0.5, 0.93, "", transform=ax_pro.transAxes, ha="center",
                         va="top", fontsize=13, fontweight="bold", color="red",
                         zorder=6)

    # ---- panel 3: the energy budget ----------------------------------------
    h, tE = res["hist"], res["hist"]["t"] / res["tau_rt"]
    for key, col, lab in (("P_el", "tab:blue", r"elastic $\mathcal{P}_{el}$"),
                          ("P_f", "tab:green", r"foundation $\mathcal{P}_f$"),
                          ("K", "tab:red", r"kinetic $\mathcal{K}$"),
                          ("D", "tab:orange", r"dissipated $\mathcal{D}$"),
                          ("total", "k", "total")):
        ax_en.plot(tE, h[key], color=col, lw=2.0 if key == "total" else 1.4,
                   ls="--" if key == "D" else "-", label=lab)
    cursor = ax_en.axvline(0.0, color="k", lw=1.6)
    for cr in res["cracks"][1:]:
        ax_en.axvline(cr["t"] / res["tau_rt"],
                      color=_WAVE_GEN_COLORS[min(cr["gen"], len(_WAVE_GEN_COLORS) - 1)],
                      lw=1.0, ls=":", alpha=0.9)
    ax_en.set_xlim(tE[0], tE[-1]); ax_en.set_xlabel(r"$t/\tau_{rt}$ (round-trips)")
    ax_en.set_ylabel("energy"); ax_en.grid(alpha=0.3)
    ax_en.legend(fontsize=7.5, ncol=5, loc="upper center")

    fig.suptitle(fr"$\Lambda$={res['Lambda_bar']:.2f}, $\Gamma$={res['Gamma']:.2f}"
                 fr"  --  release wave, threshold crossing and fragmentation",
                 fontsize=12)

    def update(k):
        j = idx[k]
        t_now, e_now = T[j], AE[:, j]

        # panel 1: strip, with cracked elements masked out
        arr = np.ma.masked_where(_wave_broken_mask(res, res["t_snap"][j]), e_now)
        im.set_array(arr[None, :])

        # panel 2: profile + over-threshold fill
        ln.set_ydata(e_now)
        fill[0].remove()
        fill[0] = ax_pro.fill_between(x, ec, e_now, where=e_now >= ec,
                                      color="red", alpha=0.55, zorder=1)
        i_max = int(np.argmax(e_now))
        over = e_now[i_max] >= ec
        pk.set_data([x[i_max]], [e_now[i_max]])
        pk.set_markerfacecolor("red" if over else "yellow")
        banner.set_text("CRITICAL  --  a crack is nucleating here" if over else "")

        # cracks already born: show them on BOTH the strip and the profile
        for ln_ in crack_lines + bar_marks:
            ln_.remove()
        crack_lines.clear(); bar_marks.clear()
        for cr in res["cracks"]:
            if cr["t"] <= res["t_snap"][j]:
                col = _WAVE_GEN_COLORS[min(cr["gen"], len(_WAVE_GEN_COLORS) - 1)]
                crack_lines.append(
                    ax_pro.axvline(cr["x"] / res["ell_e"], color=col, ls="--",
                                   lw=1.6, alpha=0.95, zorder=4))
                bar_marks.append(
                    ax_bar.axvline(cr["x"] / res["ell_e"], color=col, lw=2.4,
                                   alpha=1.0, zorder=5))
        n_now = sum(1 for cr in res["cracks"] if cr["t"] <= res["t_snap"][j]) - 1
        ax_bar.set_title(f"the film  (colour = $|e|$, black = crack)   --   "
                         f"t = {t_now:.2f} $\\tau_{{rt}}$,   secondary cracks: {n_now}",
                         fontsize=10)
        cursor.set_xdata([t_now, t_now])
        return ()

    anim = FuncAnimation(fig, update, frames=len(idx), blit=False)
    if fmt == "mp4":
        # h264 so the talk can *pause* mid-cascade; a GIF cannot be stopped.
        from matplotlib.animation import FFMpegWriter
        try:
            import imageio_ffmpeg
            mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
        path = out / f"{fname}.mp4"
        anim.save(path, writer=FFMpegWriter(
            fps=fps, bitrate=-1,
            extra_args=["-pix_fmt", "yuv420p", "-crf", "18",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]), dpi=dpi)
    else:
        path = out / f"{fname}.gif"
        anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    if verbose:
        import os
        print(f"saved {path}  ({os.path.getsize(path)/1e6:.1f} MB, "
              f"{len(idx)} frames @ {fps} fps)")
    return path


# =============================================================================
# Publication-ready figures: one consistent style, vector PDF + raster PNG
# =============================================================================
FINAL_FIGS = "final_figs"   # sub-directory of output/, safe to point LaTeX at


def figure_paths(output_dir, stem: str) -> dict:
    """Where a publication-ready figure is written: ``output/final_figs/<stem>.*``."""
    d = Path(output_dir) / FINAL_FIGS
    d.mkdir(parents=True, exist_ok=True)
    return {"png": d / f"{stem}.png", "pdf": d / f"{stem}.pdf"}


def save_final_figure(fig, output_dir, stem: str, verbose: bool = True):
    """Save a figure in both formats, at publication quality.

    PDF is vector (what LaTeX should ``\\includegraphics``), PNG is a 200-dpi
    raster for quick viewing.  Both go to ``output/final_figs/`` so a
    document can point at one stable directory.
    """
    p = figure_paths(output_dir, stem)
    fig.savefig(p["pdf"], bbox_inches="tight")             # vector
    fig.savefig(p["png"], dpi=200, bbox_inches="tight")    # raster
    if verbose:
        print(f"  figure -> {p['pdf'].parent}/{stem}.{{pdf,png}}")
    return p


def plot_model_comparison(output_dir, l_hats=(0.01, 0.02, 0.05, 0.1),
                          verbose: bool = True):
    """The two phase-field variants side by side -- a standard comparison figure.

    Four panels:

    1. the dissipation ``w(alpha)``: linear (AT1) vs quadratic (AT2).  The
       *slope at the origin* is what decides whether an elastic phase exists;
    2. the degradation ``g(alpha) = (1-alpha)^2``, shared by both variants;
    3. the homogeneous 1-D response ``sigma(e)``: AT1 stays exactly linear up
       to ``e_crit`` and then softens, AT2 leaves the elastic line immediately;
    4. the nucleation threshold ``e_crit`` against the regularisation length,
       in the two normalisations that coexist in this repository.
    """
    from .solvers import MODELS, critical_strain, _g

    a = np.linspace(0.0, 1.0, 400)
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))

    for name, col in (("AT1", "tab:blue"), ("AT2", "tab:red")):
        w = MODELS[name]["w"]
        ax[0].plot(a, [float(w(x)) for x in a], color=col, lw=2, label=name)
    ax[0].set_xlabel(r"$\alpha$"); ax[0].set_ylabel(r"$w(\alpha)$")
    ax[0].set_title("dissipation $w(\\alpha)$\n"
                    "$w'(0)>0$ (AT1) $\\Rightarrow$ elastic phase", fontsize=10)
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

    ax[1].plot(a, [_g(x) for x in a], "k", lw=2)
    ax[1].set_xlabel(r"$\alpha$"); ax[1].set_ylabel(r"$g(\alpha)$")
    ax[1].set_title(r"degradation $g(\alpha)=(1-\alpha)^2$" "\n(shared by both)",
                    fontsize=10)
    ax[1].grid(alpha=0.3)

    # homogeneous response: minimise 0.5 g(a) e^2 + w(a)/c_w over a >= 0
    e = np.linspace(0.0, 3.0, 600)
    for name, col in (("AT1", "tab:blue"), ("AT2", "tab:red")):
        cw = MODELS[name]["c_w"]
        sig = []
        for ee in e:
            grid = np.linspace(0.0, 1.0, 2001)
            psi = 0.5 * (1 - grid) ** 2 * ee ** 2 + (
                grid if name == "AT1" else grid ** 2) / cw
            ah = grid[int(np.argmin(psi))]
            sig.append((1 - ah) ** 2 * ee)
        ax[2].plot(e, sig, color=col, lw=2, label=name)
        ax[2].axvline(math.sqrt(1.0 / cw) if name == "AT1" else 0.0,
                      color=col, ls=":", lw=1.2)
    ax[2].plot(e, e, "k--", lw=1, alpha=0.5, label="undamaged $\\sigma=Ee$")
    ax[2].set_xlim(0, 2.0); ax[2].set_ylim(0, 1.0)
    ax[2].set_xlabel(r"strain $e$"); ax[2].set_ylabel(r"stress $\sigma$")
    ax[2].set_title("homogeneous response\n(dotted: elastic limit)", fontsize=10)
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

    ld = np.geomspace(min(l_hats) / 2, max(l_hats) * 2, 200)
    ax[3].loglog(ld, [critical_strain("AT1", ell_d=x) for x in ld],
                 color="tab:blue", lw=2,
                 label=r"note: $e_c=\sqrt{G_c/(E\ell_d)}$")
    ax[3].axhline(math.sqrt(1.0 / MODELS["AT1"]["c_w"]), color="tab:green",
                  lw=2, label=r"FEM: $e_c=\sqrt{w'(0)/(c_w E)}$")
    ax[3].set_xlabel(r"$\ell_d$ (or $\hat\ell$)"); ax[3].set_ylabel(r"$e_{crit}$")
    ax[3].set_title("AT1 threshold: the two normalisations\n"
                    "(ratio $\\sqrt{8/3}=1.63$)", fontsize=10)
    ax[3].legend(fontsize=8); ax[3].grid(alpha=0.3, which="both")

    fig.suptitle("Phase-field variants used in this work: AT1 has a genuine elastic phase, AT2 does not",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_final_figure(fig, output_dir, "model_AT1_vs_AT2", verbose=verbose)
    return fig


def adopt_final_figure(png_src, pdf_src, output_dir, stem: str,
                        verbose: bool = True):
    """Copy an already-saved run figure into ``output/final_figs/``.

    The per-run figures (``plot_mechanical_run`` / ``plot_thermal_run``) carry
    the whole parameter set in their file name, which is what stops a sweep
    overwriting itself but is useless in a document.  This gives one of them
    a short, stable name in the final directory without redrawing it.
    """
    import shutil
    p = figure_paths(output_dir, stem)
    for src, dst in ((pdf_src, p["pdf"]), (png_src, p["png"])):
        if src and Path(src).exists():
            shutil.copyfile(src, dst)
    if verbose:
        print(f"  figure -> {p['pdf'].parent}/{stem}.{{pdf,png}}")
    return p


def plot_eta_energy_study(study: dict, output_dir, verbose: bool = True):
    """The **energy-based** eta study -- the first attempt, and why it fails.

    Three panels, arranged so the two defects of the comparison are visible
    rather than argued:

    1. the family of total-energy curves, dynamic against quasi-static;
    2. the resulting "gap" against ``eta`` -- it falls and then **flattens onto
       a floor**, and is not even monotone;
    3. the cumulative viscous dissipation the dynamic total leaves out, which
       is the same size as the floor: the comparison is asymmetric, because the
       quasi-static branch has no dissipation channel at all.
    """
    out_dir = Path(output_dir)
    etas, grid = study["etas"], study["load_grid"]
    label = _ETA_LOAD_LABEL[study["physics"]]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 4.4))

    cmap = plt.cm.viridis(np.linspace(0.12, 0.92, len(etas)))
    a1.plot(grid, study["E_qs"], "k-", lw=3, label="quasi-static (reference)")
    for eta, Ed, c in zip(etas, study["E_dyn"], cmap):
        a1.plot(grid, Ed, "-", color=c, lw=1.5, label=fr"dyn $\eta={eta:g}$")
    a1.set_xlabel(label)
    a1.set_ylabel(r"total energy $\mathcal{K}+\mathcal{P}_{el}+\mathcal{P}_f+\mathcal{S}$")
    a1.set_title(f"{study['physics']} ($\\Lambda={study['Lambda']:g}$): "
                 "the two energy branches")
    a1.legend(fontsize=7, ncol=2); a1.grid(alpha=0.3)

    a2.plot(etas, study["gaps"], "o-", color="tab:red", lw=2, ms=7)
    floor = float(np.min(study["gaps"]))
    a2.axhline(floor, color="grey", ls=":", lw=1.2, label=f"floor $\\approx${floor:.2e}")
    a2.set_xscale("log"); a2.set_yscale("log"); a2.invert_xaxis()
    a2.set_xlabel(r"loading time-scale $\eta$   (slower loading $\rightarrow$)")
    a2.set_ylabel(r"$\|E_{dyn}-E_{qs}\|/\max|E_{qs}|$")
    a2.set_title("the gap falls, then stalls on a floor\n(and is not even monotone)")
    a2.legend(fontsize=9); a2.grid(alpha=0.3, which="both")

    a3.plot(etas, study["D_end"] / study["norm"], "s-", color="tab:orange", lw=2, ms=7,
            label=r"$\mathcal{D}$ at the end of the run")
    a3.plot(etas, study["gaps"], "o--", color="tab:red", lw=1.2, ms=5, alpha=0.7,
            label="the gap, for scale")
    a3.set_xscale("log"); a3.set_yscale("log"); a3.invert_xaxis()
    a3.set_xlabel(r"loading time-scale $\eta$")
    a3.set_ylabel(r"normalised by $\max|E_{qs}|$")
    a3.set_title("what the comparison LEAVES OUT:\nthe dissipation, absent from the QS branch")
    a3.legend(fontsize=9); a3.grid(alpha=0.3, which="both")

    fig.suptitle("Comparing TOTAL ENERGIES: the natural first attempt, and why it does not "
                 "measure how close the two solutions are", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ("png", "pdf"):
        f = out_dir / f"eta_energy_{study['physics']}.{ext}"
        fig.savefig(f, dpi=150, bbox_inches="tight")
        if verbose:
            print(f"  saved {f}")
    save_final_figure(fig, out_dir, f"eta_energy_{study['physics']}", verbose=verbose)
    plt.show()
    return fig
