"""
tools/norms.py
==============
Lebesgue and Sobolev norms of **P1 finite-element fields**, used to measure
distances between solutions (see :mod:`problems.eta_convergence`).

Why these norms
---------------
Comparing two computed evolutions through their *energies* makes the answer
depend on how the energy is bookkept.  Comparing the **states** does not.  For
a gradient-damage model the state is the pair ``y = (u, alpha)`` and the
finiteness of the energy

.. math::
   \\int_\\Omega g(\\alpha)|u'|^2 + |u|^2 + w(\\alpha) + |\\alpha'|^2 \\;<\\; \\infty

places both fields in the Sobolev space :math:`H^1(\\Omega)`, whose norm is the
natural distance:

.. math::
   \\|v\\|_{L^2}=\\Big(\\int_\\Omega |v|^2\\Big)^{1/2},\\qquad
   \\|v\\|_{H^1}=\\Big(\\int_\\Omega |v|^2+\\int_\\Omega |v'|^2\\Big)^{1/2},\\qquad
   \\|y\\|^2=\\|u\\|^2_{H^1}+\\|\\alpha\\|^2_{H^1}.

Exactness
---------
Every integral below is evaluated **exactly** for a piecewise-linear field --
no quadrature error enters the measurement:

* on an element of length ``h`` the integral of the square of the linear
  interpolant is ``h (v_i^2 + v_i v_j + v_j^2) / 3``;
* the derivative is piecewise constant, so ``int (v')^2 = sum_e (v_j-v_i)^2/h_e``.

Norm equivalence
----------------
On a fixed finite-element space all norms are equivalent -- there is a ``C>0``
with ``C||v||_1 <= ||v||_2 <= (1/C)||v||_1`` -- so the *conclusion* of a
convergence study must not depend on which one is used.  Reporting several of
them side by side (as :mod:`problems.eta_convergence` does) turns that
statement into a check.

All functions take the **sorted** node coordinates ``x`` and the nodal values
``v``; :func:`sort_dofs` builds the permutation that sorts a dolfinx dof
vector, whose natural order is the solver's, not left-to-right.
"""

from __future__ import annotations

import numpy as np


def sort_dofs(x_dofs):
    """Permutation that orders 1-D dof coordinates left to right.

    dolfinx numbers degrees of freedom in solver order, so a nodal array is
    *not* laid out along the bar.  Every norm below assumes it is.
    """
    return np.argsort(np.asarray(x_dofs, float))


def l2_norm_sq(x, v):
    """``int_Omega v^2 dx`` for a P1 field -- exact."""
    x = np.asarray(x, float); v = np.asarray(v, float)
    h = np.diff(x)
    vi, vj = v[:-1], v[1:]
    return float(np.sum(h * (vi * vi + vi * vj + vj * vj) / 3.0))


def h1_semi_norm_sq(x, v):
    """``int_Omega (v')^2 dx`` for a P1 field -- exact.

    The derivative of a linear interpolant is constant on each element, so the
    integral is the plain sum ``(v_j - v_i)^2 / h_e``.
    """
    x = np.asarray(x, float); v = np.asarray(v, float)
    h = np.diff(x)
    return float(np.sum(np.diff(v) ** 2 / h))


def l2_norm(x, v):
    """``||v||_{L^2}``."""
    return float(np.sqrt(l2_norm_sq(x, v)))


def h1_norm(x, v):
    """``||v||_{H^1} = (||v||_{L^2}^2 + |v|_{H^1}^2)^{1/2}``."""
    return float(np.sqrt(l2_norm_sq(x, v) + h1_semi_norm_sq(x, v)))


def state_norm(x_u, u, x_alpha, alpha):
    """``||y||`` on the product space ``H^1 x H^1``, with ``y = (u, alpha)``."""
    return float(np.sqrt(l2_norm_sq(x_u, u) + h1_semi_norm_sq(x_u, u)
                         + l2_norm_sq(x_alpha, alpha)
                         + h1_semi_norm_sq(x_alpha, alpha)))


#: The norms this module exposes, in the order they are usually reported.
NORMS = {
    "u_L2": ("displacement", l2_norm),
    "u_H1": ("displacement", h1_norm),
    "a_L2": ("damage",       l2_norm),
    "a_H1": ("damage",       h1_norm),
}
