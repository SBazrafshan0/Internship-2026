"""
tools/solvers.py
================
*Model* definitions.

In the variational phase-field framework the damage variable :math:`\\alpha
\\in [0,1]` carries a *dissipation density* of the form

.. math::

    \\psi_d(\\alpha,\\nabla\\alpha) \\;=\\;
        \\frac{G_c}{c_w}\\,
        \\Big(\\,\\frac{w(\\alpha)}{\\ell} \\;+\\; \\ell\\,|\\nabla\\alpha|^2\\,\\Big)

The model variant fixes the shape of :math:`w(\\alpha)` (and the elastic
degradation :math:`g(\\alpha)`).  Only *one* place in the repository owns
that decision -- this file.  Adding a new variant means adding a new entry to
:data:`MODELS` and the rest of the code (problem files, sweep file) inherits
it automatically.

Two variants are shipped by default:

================  ===========  ============  ===========================
Variant           ``w(alpha)`` ``c_w``       Sub-critical regime?
================  ===========  ============  ===========================
``"AT1"``         ``alpha``    ``8/3``       Yes (elastic phase up to the
                                             damage threshold)
``"AT2"``         ``alpha**2`` ``2``         No (damage grows from t=0+)
================  ===========  ============  ===========================

The elastic degradation function is the standard one for both variants,
:math:`g(\\alpha) = (1-\\alpha)^2`.

Everything is non-dimensionalised so that :math:`G_c / c_w = 1`.  The internal
length is the dimension-less ``l_hat`` from :mod:`tools.parameters`.

Adding a new model
------------------
Append a dictionary to :data:`MODELS` with the keys ``w`` and ``c_w``::

    MODELS["AT_custom"] = {
        "w":   lambda a: ...,                # UFL or sympy expression
        "c_w": <float>,
    }

The problem files use :func:`get_model` to fetch the dictionary, then build
the fracture-energy density as

.. math::

    \\psi_d \\;=\\; \\frac{1}{c_w}\\,\\big(\\,w(\\alpha) +
    \\hat\\ell^{\\,2}|\\nabla\\alpha|^2\\big)
"""

import math

# ufl is only needed when the model lambdas are fed UFL expressions (FEM
# problems).  The semi-analytic problem (problems/secondary.py) evaluates the
# same lambdas on floats, so the import is optional.
try:
    import ufl
except ImportError:                                       # pragma: no cover
    ufl = None


def _g(alpha):
    """Elastic degradation function ``g(alpha) = (1-alpha)^2``."""
    return (1.0 - alpha) ** 2


# -----------------------------------------------------------------------------
# Model registry
# -----------------------------------------------------------------------------
MODELS = {
    "AT1": {
        "w":   lambda a: a,
        "c_w": 8.0 / 3.0,
        "alpha_lb0": 0.0,
        "description": "AT1: linear w(alpha) = alpha (sub-critical elastic phase).",
    },
    "AT2": {
        "w":   lambda a: a * a,
        "c_w": 2.0,
        "alpha_lb0": 0.0,
        "description": "AT2: quadratic w(alpha) = alpha^2 (no sub-critical phase).",
    },
}


def get_model(name: str) -> dict:
    """Return the dictionary defining the model variant ``name``."""
    if name not in MODELS:
        raise KeyError(
            f"Unknown phase-field model {name!r}. Available: {list(MODELS)}"
        )
    return MODELS[name]


def g_degradation(alpha):
    """Elastic degradation function -- exposed for problem files."""
    return _g(alpha)


def critical_strain(model_name: str, Gc: float = 1.0, E_h: float = 1.0,
                    ell_d: float = 1.0, h: float = 1.0e-8) -> float:
    """
    Damage-nucleation strain threshold of the concept note (secondary
    cracking).  For the energy density

        psi(e, alpha) = 0.5 * E_h * g(alpha) * e**2
                        + (Gc/ell_d) * w(alpha) + Gc * ell_d * |alpha_x|**2,

    damage nucleates from the intact state alpha = 0 when

        e**2 >= e_crit**2 = 2 * Gc * w'(0) / (E_h * ell_d * |g'(0)|).

    ``w'(0)`` and ``g'(0)`` are evaluated by a forward difference on the model
    lambdas, so any entry of :data:`MODELS` works.  AT1 (w'(0)=1) gives a
    finite elastic phase; AT2 (w'(0)=0) gives e_crit = 0 (no sub-critical
    regime), consistent with the table above.

    .. warning::
       **Two normalisations of the fracture term coexist in this repository.**
       This function uses the *concept note's* form
       ``psi_d = (Gc/ell_d) w(a) + Gc ell_d |a_x|^2`` (no ``1/c_w``), whereas
       :func:`fracture_energy_density` -- the one the FEM problems actually
       assemble -- uses ``psi_d = (w(a) + l_hat^2 |grad a|^2) / c_w``.

       The literature (Pham et al. 2011a; Tanne et al. 2018; and eq. (33) of
       Leon Baldelli & Maurini, JMPS 152 (2021) 104424, in ``Articles/``)
       calibrates the model through ``Gc = 4 c_w^lit w1 ell`` with
       ``c_w^lit = int_0^1 sqrt(w(a)) da`` and the elastic limit
       ``sigma_c = sqrt(w1 E)``.  The ``c_w`` stored in :data:`MODELS`
       (8/3 for AT1, 2 for AT2) is exactly that ``4 c_w^lit``, so

           FEM  (fracture_energy_density):  e_crit = sqrt(w'(0)/(c_w E))
                                                   = sqrt(3/8) = 0.6124  (AT1)
           here (note's normalisation):     e_crit = sqrt(Gc/(E ell_d))
                                                   = 1.0                 (AT1)

       i.e. a factor ``sqrt(8/3) = 1.633``.  **The FEM value is the one that
       matches the literature calibration.**  More importantly the two have
       *opposite* dependence on the regularisation length: here ``Gc`` is held
       fixed and ``e_crit ~ 1/sqrt(ell_d)``, while in the FEM form ``e_crit``
       is independent of ``l_hat`` and it is the effective toughness that
       grows, ``Gc = l_hat``.  Sweeping ``l_hat`` therefore means *different
       physical experiments* in the two settings -- rescale before comparing
       the semi-analytic maps of ``problems/secondary.py`` with FEM runs.
    """
    m = get_model(model_name)
    wp0 = (float(m["w"](h)) - float(m["w"](0.0))) / h
    gp0 = (_g(h) - _g(0.0)) / h
    return math.sqrt(2.0 * Gc * wp0 / (E_h * ell_d * abs(gp0)))


def damage_band_width(e_crit: float, Gc: float = 1.0, E_h: float = 1.0) -> float:
    """Width of the damage band implied by a nucleation threshold.

    Inverse of :func:`critical_strain` in the concept note's normalisation:
    ``e_crit = sqrt(Gc / (E_h ell_d))`` for AT1, hence

        ell_d = Gc / (E_h * e_crit**2).

    A strain spike narrower than ``ell_d`` cannot nucleate anything, because a
    real crack is a *band* of that width -- which is why the secondary-cracking
    criterion has to be filtered over ``ell_d`` before it is applied
    (see ``problems/secondary_theory.ipynb`` section 10).
    """
    return Gc / (E_h * e_crit ** 2)


def fracture_energy_density(alpha, grad_alpha_sq, l_hat, model_name: str):
    """
    Return the UFL expression of the *fracture energy density*

        psi_d(alpha, grad alpha) = (1/c_w) (w(alpha) + l_hat**2 * |grad alpha|**2).

    Parameters
    ----------
    alpha          : UFL expression for the damage field.
    grad_alpha_sq  : UFL expression for |grad alpha|**2 (so the same
                     function works in 1D and 2D).
    l_hat          : UFL constant for the regularisation length.
    model_name     : "AT1" or "AT2" (see :data:`MODELS`).
    """
    m = get_model(model_name)
    return (m["w"](alpha) + l_hat ** 2 * grad_alpha_sq) / m["c_w"]
