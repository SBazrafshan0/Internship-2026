"""
tools/animations.py
===================
Three-panel run animations, one per solved problem.

Layout, fixed for every animation so the audience learns it once:

    [ schematic (static) ]  [ energy history + marker ]  [ damage field ]

* **left** -- what the specimen *is*: geometry, supports, load, foundation.
  Drawn once and never redrawn, so the eye can use it as an anchor.
* **middle** -- where the energy is: the full history is drawn faintly from
  the start, and a marker walks it, so the audience sees both the whole
  budget and the current instant at once.
* **right** -- what the damage *does*: the 1D profile alpha(x), or the 2D
  field, at the current step.

One frame per recorded solver step (``field_recorder`` fires at every
converged step), so the animation is as fine as the run itself.

Output is MP4 (h264) so it can be paused mid-talk; ``also_gif=True``
additionally writes a GIF for platforms that will not host video.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import animation
from matplotlib.patches import Rectangle, FancyArrow, Polygon
from pathlib import Path

try:                                     # bundled ffmpeg, if present
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:                        # pragma: no cover
    pass

C = {
    "P_el": "#0072B2", "P_f": "#009E73", "K": "#D55E00",
    "S": "#CC79A7", "D": "#E69F00", "total": "#000000",
    "at1": "#0072B2", "crit": "#A2192E", "muted": "#7F7F7F",
    "bar": "#5A6B7D", "found": "#009E73", "grid": "#D8D8D8",
}

_RC = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": .6,
    "lines.linewidth": 1.8, "figure.dpi": 130,
    "mathtext.fontset": "cm",   # \mathcal etc. need the CM fontset
}


# ---------------------------------------------------------------------------
# left panel: the specimen
# ---------------------------------------------------------------------------
def _coil(ax, x0, y0, y1, *, amp=0.022, n=6, lw=0.9, color="k", zorder=2):
    """A Winkler spring in the usual engineering-drawing form:
    a short straight lead, a coil, a short straight lead."""
    pre = 0.16 * (y1 - y0)
    ys = np.linspace(y0 + pre, y1 - pre, 220)
    t = (ys - ys[0]) / (ys[-1] - ys[0])
    xs = x0 + amp * np.sin(2 * np.pi * n * t)
    ax.plot([x0, x0], [y0, y0 + pre], color=color, lw=lw, zorder=zorder)
    ax.plot(xs, ys, color=color, lw=lw, zorder=zorder, solid_joinstyle="round")
    ax.plot([x0, x0], [y1 - pre, y1], color=color, lw=lw, zorder=zorder)


def _hatch_ground(ax, x0, x1, y, depth=0.075, lw=0.8):
    """Hatched half-space below a line: the rigid substrate."""
    ax.plot([x0, x1], [y, y], color="k", lw=lw, zorder=2)
    ax.add_patch(Rectangle((x0, y - depth), x1 - x0, depth, fc="none",
                           ec="k", lw=0.0, hatch="/////", zorder=1))


def _hatch_wall(ax, x, y0, y1, side=-1, width=0.055, lw=0.9):
    """Hatched clamped wall at ``x`` (side=-1 draws it to the left)."""
    ax.plot([x, x], [y0, y1], color="k", lw=1.2, zorder=5)
    x_a = x + side * width
    ax.add_patch(Rectangle((min(x, x_a), y0), width, y1 - y0, fc="none",
                           ec="k", lw=0.0, hatch="/////", zorder=1))
    ax.plot([x_a, x_a], [y0, y1], color="k", lw=0.0)


_ASSETS = Path(__file__).resolve().parent / "assets"


def _plate_2d(ax, kind: str):
    """Place the prepared 2D schematic (`assets/schem2d_*.png`) in the panel.

    This was redrawn in matplotlib for a while and it never read well: the
    plate has to be seen in perspective, with L_x along the front edge and
    L_y going back into the page, and every hand-rolled version of that came
    out worse than a drawn figure.  So the two panels are shipped as images
    next to this module and shown directly.
    """
    name = "schem2d_therm.png" if "therm" in kind else "schem2d_mech.png"
    img = mpimg.imread(str(_ASSETS / name))
    h, w = img.shape[0], img.shape[1]
    ax.imshow(img, extent=[0.0, 1.0, 0.0, h / w], zorder=3,
              interpolation="antialiased")
    # a band under the figure for the running load readout that animate_run
    # stamps into this axes
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.20, h / w + 0.04)


def _schematic(ax, kind: str, title: str = ""):
    """Draw the static specimen sketch, in a black-and-white line style."""
    two_d_case = "2d" in kind
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9.5, pad=6)

    # A 2D specimen is a plate seen in perspective: use the prepared image,
    # which sets its own limits.
    if two_d_case:
        _plate_2d(ax, kind)
        return

    ax.set_xlim(-0.34, 1.30)
    ax.set_ylim(-0.95, 0.75)

    foundation = "found" in kind
    thermal = "therm" in kind
    clamped = "clamp" in kind
    two_d = "2d" in kind

    hgt = 0.16 if not two_d else 0.30
    y_b, y_t = 0.0, hgt                      # bar sits on y=0
    y_gnd = -0.42                            # foundation base line

    if foundation:
        _hatch_ground(ax, -0.10, 1.10, y_gnd)
        for xs in np.linspace(0.06, 0.94, 10):
            _coil(ax, xs, y_gnd, y_b)
        ax.annotate(r"$\mathsf{K},\ \Lambda=1/\ell_e$",
                    xy=(0.96, (y_gnd + y_b) / 2), xytext=(1.14, -0.18),
                    fontsize=7.5, color="k", ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", lw=0.5, color="k"))

    # the film / bar itself
    ax.add_patch(Rectangle((0, y_b), 1.0, hgt, fc="#EDEDED", ec="k",
                           lw=0.9, zorder=3))

    if thermal:
        # Double-headed: an eigenstrain is not a directed load, it is the
        # material wanting to change length along the axis.  Single-headed
        # arrows read as a traction applied one way, which is wrong.
        # Four long arrows rather than five short ones: a double head needs a
        # shaft to sit on, and at the old span the two heads met and the
        # arrow read as a small blob rather than as an arrow.
        for xc in np.linspace(0.17, 0.83, 4):
            ax.annotate("", xy=(xc + 0.075, y_b + hgt/2),
                        xytext=(xc - 0.075, y_b + hgt/2),
                        arrowprops=dict(arrowstyle="<->", lw=0.7, color="k",
                                        mutation_scale=7),
                        zorder=6)
        ax.text(0.5, y_t + 0.30, r"eigenstrain $\theta(t)$", ha="center",
                fontsize=8.5, color="k")
        ax.annotate("", xy=(0.5, y_t + 0.04), xytext=(0.5, y_t + 0.26),
                    arrowprops=dict(arrowstyle="->", lw=0.5, color="k"))
        if clamped:
            _hatch_wall(ax, 0.0, y_b - 0.10, y_t + 0.10, side=-1)
            _hatch_wall(ax, 1.0, y_b - 0.10, y_t + 0.10, side=+1)
            ax.text(0.5, y_b - 0.20, "both ends held", ha="center",
                    fontsize=7.5, color="k")
        else:
            ax.text(-0.13, y_b + hgt/2, "free", fontsize=7.5, ha="right",
                    va="center")
            ax.text(1.32 if two_d else 1.13, y_b + hgt/2, "free",
                    fontsize=7.5, ha="left", va="center")
    else:
        _hatch_wall(ax, 0.0, y_b - 0.10, y_t + 0.10, side=-1)
        x_arr = 1.20 if not two_d else 1.34
        ax.annotate("", xy=(x_arr + 0.18, y_b + hgt/2),
                    xytext=(x_arr, y_b + hgt/2),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="k"))
        ax.text(x_arr + 0.09, y_b + hgt/2 + 0.10, r"$\hat U(t)$",
                fontsize=8.5, ha="center")
        for xe, ha in ((0.03, "left"), (0.97, "right")):
            ax.text(xe, y_t + 0.06, r"$\alpha=0$", fontsize=6.5, ha=ha,
                    va="bottom", color="k")

    if two_d:
        ax.annotate("", xy=(1.05, y_b), xytext=(1.05, y_t),
                    arrowprops=dict(arrowstyle="<->", color="k", lw=0.6))
        ax.text(1.08, y_b + hgt/2, r"$L_y$", fontsize=7.5, va="center")

    y_dim = y_gnd - 0.16 if foundation else y_b - 0.30
    ax.annotate("", xy=(0, y_dim), xytext=(1.0, y_dim),
                arrowprops=dict(arrowstyle="<->", color="k", lw=0.6))
    ax.text(0.5, y_dim - 0.04, r"$\Omega=(0,L)$", fontsize=7.5, color="k",
            ha="center", va="top")


# ---------------------------------------------------------------------------
# the animation itself
# ---------------------------------------------------------------------------
_ENERGY_KEYS = (("P_el", r"$\mathcal{P}_{el}$"), ("P_f", r"$\mathcal{P}_{f}$"),
                ("S", r"$\mathcal{S}$"), ("K", r"$\mathcal{K}$"),
                ("D", r"$\mathcal{D}$"))


def animate_run(hist, loads, frames_alpha, x, out_stem, *,
                schematic="mech", schematic_title="",
                load_label=r"load", damage_title=r"damage $\alpha(x)$",
                triang=None, fps=10, also_gif=False, dpi=130,
                ref_hist=None, ref_label="quasi-static total",
                figsize=(12.6, 3.5), verbose=True):
    """Write ``out_stem``.mp4 (and optionally .gif) for one run.

    hist          : the branch history dict (keys of _ENERGY_KEYS + "total")
    loads         : load value per recorded frame
    frames_alpha  : list of damage arrays, one per frame
    x             : nodal coordinates (1D) -- ignored when ``triang`` is given
    triang        : matplotlib Triangulation for a 2D field animation
    """
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    loads = np.asarray(loads, dtype=float)
    n = min(len(loads), len(frames_alpha))
    loads, frames_alpha = loads[:n], frames_alpha[:n]

    with plt.rc_context(_RC):
        # A 2D damage panel is as wide as the specimen; a 5:1 strip drawn in
        # a square-ish axes wastes most of the panel on margin, so give the
        # right-hand column more of the figure when the field is 2D.
        wr = [1.40, 1.15, 1.75] if triang is not None else [1.0, 1.25, 1.25]
        fig, (axL, axM, axR) = plt.subplots(
            1, 3, figsize=figsize,
            gridspec_kw=dict(width_ratios=wr, wspace=0.28))

        _schematic(axL, schematic, schematic_title)

        # ---- middle: the energy budget, drawn whole, then walked ----------
        eload = np.asarray(hist.get("theta", hist.get("U", loads)), float)
        drawn = []
        for key, lab in _ENERGY_KEYS:
            if key in hist and len(hist[key]) == len(eload) and np.any(np.abs(hist[key]) > 1e-14):
                axM.plot(eload, hist[key], color=C[key], lw=1.5, label=lab, alpha=.85)
                drawn.append(key)
        if "total" in hist and len(hist["total"]) == len(eload):
            axM.plot(eload, hist["total"], color=C["total"], lw=2.0, ls="--",
                     label="total", alpha=.9)
        # the quasi-static total, drawn as the reference the dynamic branch is
        # to be read against (and harmless when the animated branch *is* QS).
        if ref_hist is not None:
            rload = np.asarray(ref_hist.get("theta", ref_hist.get("U", [])), float)
            if "total" in ref_hist and len(ref_hist["total"]) == len(rload) and len(rload):
                axM.plot(rload, ref_hist["total"], color=C["muted"], lw=1.6,
                         ls=":", label=ref_label, alpha=.95, zorder=2)
        axM.set_xlabel(load_label)
        axM.set_ylabel("energy")
        axM.set_title("energy budget", fontsize=9.5, pad=6)
        axM.legend(loc="upper left", ncol=2, frameon=False,
                   handlelength=1.4, columnspacing=1.0)
        vline = axM.axvline(eload[0] if len(eload) else 0.0,
                            color=C["crit"], lw=1.4, zorder=5)
        dot, = axM.plot([], [], "o", color=C["crit"], ms=5.5, zorder=6)

        # ---- right: the damage field --------------------------------------
        axR.set_title(damage_title, fontsize=9.5, pad=6)
        if triang is None:
            line, = axR.plot(x, frames_alpha[0], color=C["at1"], lw=2.0)
            fill = axR.fill_between(x, 0, frames_alpha[0], color=C["at1"], alpha=.18, lw=0)
            axR.set_xlim(float(np.min(x)), float(np.max(x)))
            axR.set_ylim(-0.03, 1.05)
            axR.set_xlabel(r"$x$")
            axR.set_ylabel(r"$\alpha$")
        else:
            tp = axR.tripcolor(triang, frames_alpha[0], cmap="magma_r",
                               vmin=0.0, vmax=1.0, shading="gouraud")
            axR.set_aspect("equal")
            axR.grid(False)
            axR.set_xlabel(r"$x$")
            fig.colorbar(tp, ax=axR, fraction=0.026, pad=0.02, label=r"$\alpha$")

        # the running load lives under the schematic: the one place that is
        # empty in every configuration, 1D and 2D alike, so it can never sit
        # on top of the data.
        # 0.045 of the axes height reproduces the old 1D position exactly and
        # follows the box when the 2D panel sets different limits.
        stamp = axL.text(0.5, 0.045, "", transform=axL.transAxes, ha="center",
                         va="center", fontsize=13, color=C["crit"],
                         fontweight="bold")

        state = {"fill": fill if triang is None else None}

        def update(i):
            a = frames_alpha[i]
            if triang is None:
                line.set_ydata(a)
                if state["fill"] is not None:
                    state["fill"].remove()
                state["fill"] = axR.fill_between(x, 0, a, color=C["at1"],
                                                 alpha=.18, lw=0)
            else:
                tp.set_array(a)
            L = loads[i]
            vline.set_xdata([L, L])
            if "total" in hist and len(hist["total"]) == len(eload) and len(eload):
                dot.set_data([L], [np.interp(L, eload, hist["total"])])
            stamp.set_text(f"{load_label} = {L:.3f}")
            return ()

        anim = animation.FuncAnimation(fig, update, frames=n, blit=False,
                                       interval=1000 / fps)
        mp4 = out_stem.with_suffix(".mp4")
        anim.save(str(mp4), writer=animation.FFMpegWriter(
            fps=fps, bitrate=-1,
            extra_args=["-pix_fmt", "yuv420p", "-crf", "18",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]), dpi=dpi)
        if verbose:
            print(f"    {mp4}  ({n} frames, {mp4.stat().st_size/1e6:.1f} MB)")
        if also_gif:
            gif = out_stem.with_suffix(".gif")
            anim.save(str(gif), writer=animation.PillowWriter(fps=fps), dpi=90)
            if verbose:
                print(f"    {gif}  ({gif.stat().st_size/1e6:.1f} MB)")
        plt.close(fig)
    return mp4


def collect(result_branch_hist):
    """Convenience: pull (loads, energies) out of a run's history dict."""
    return np.asarray(result_branch_hist.get(
        "theta", result_branch_hist.get("U", [])), float)
