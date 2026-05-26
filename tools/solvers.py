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

import ufl


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
