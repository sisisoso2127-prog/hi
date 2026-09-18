"""Exploration of the edges carrying the alternative optima (Definition 2).

At the optimum ``x_k`` of the truncated problem, the non-basic variables whose
reduced gradient vanishes,

    Gamma_l = { j in N_l : gamma_{l,j} = 0 } ,

generate edges along which ``Phi`` stays constant.  If one of them contains an
*efficient* integer point, that point attains the current upper bound
``Phi(x_k)`` and is therefore globally optimal for ``(P_E)``: the algorithm can
stop immediately instead of paying another Sylva-Crema iteration.

Along the edge of ``j``, Definition 2 moves

    x_i = x_{k,i} - theta * y_{k,ij}   (i basic),   x_j = theta,   x_alpha = 0,

for integer ``theta`` in ``1 .. theta0``, where

    theta0 = floor( min { x_{k,i} / y_{k,ij} : y_{k,ij} > 0 } )

is the largest step that keeps the point feasible.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import floor
from typing import Callable, List, Optional, Sequence, Tuple

from .model import MOILP, Model
from .rational import ZERO, dot, fmt, is_integral
from .simplex import FractionalPricing, Tableau


@dataclass
class EdgeCandidate:
    """An integer point met while walking an edge of ``Gamma_l``."""

    j: int                       # entering (non-basic) column defining the edge
    theta: int                   # step length along the edge
    x: List[Fraction]            # the point, restricted to the model variables
    phi: Fraction


def reduced_gradient(tableau: Tableau, phi) -> List[Fraction]:
    """The vector ``gamma_k`` of the paper at the vertex held by *tableau*.

        gamma_{k,j} = Z_{k,2} (U_j - p_{k,j}) - Z_{k,1} (V_j - q_{k,j})

    The formula is invariant under the simultaneous sign flip of
    ``(U, alpha, V, beta)``, so the vertices sitting in the negative-denominator
    branch are priced with the flipped -- equivalent -- objective and yield the
    very same ``gamma``.
    """
    obj = phi.lift(tableau.n)
    z2 = dot(obj.V, tableau.solution()) + obj.beta
    if z2 < 0:
        obj = obj.__class__([-u for u in obj.U], [-v for v in obj.V], -obj.alpha, -obj.beta)
    elif z2 == 0:
        raise ZeroDivisionError("vanishing denominator at the current vertex")
    return FractionalPricing(obj.U, obj.alpha, obj.V, obj.beta).prices(tableau)


def alternative_optima_columns(tableau: Tableau, phi) -> List[int]:
    """``Gamma_l = { j non-basic : gamma_j = 0 }`` (exact test, no tolerance)."""
    gamma = reduced_gradient(tableau, phi)
    return [j for j in tableau.nonbasic() if gamma[j] == 0]


def max_step(tableau: Tableau, j: int) -> int:
    """``theta0``: integer part of the minimum ratio along the edge of ``j``."""
    ratios = [tableau.xb[i] / tableau.T[i][j]
              for i in range(tableau.m) if tableau.T[i][j] > 0]
    if not ratios:
        return 0            # unbounded edge: the truncated region is not bounded
    return int(floor(min(ratios)))


def walk_edge(tableau: Tableau, j: int, n_model: int) -> List[Tuple[int, List[Fraction]]]:
    """Integer points ``(theta, x)`` of the edge ``E_j``, far end first.

    The paper decreases ``theta`` from ``theta0`` down to 1, which probes the
    most distant -- hence the most promising -- alternative optima first.
    """
    points = []
    for theta in range(max_step(tableau, j), 0, -1):
        x = tableau.solution()
        x[j] = Fraction(theta)
        for i, basic in enumerate(tableau.basis):
            x[basic] = tableau.xb[i] - theta * tableau.T[i][j]
        if any(v < 0 for v in x):
            continue
        candidate = x[:n_model]
        if not all(is_integral(v) for v in candidate):
            continue
        points.append((theta, candidate))
    return points


def explore_edges(tableau: Tableau, problem: MOILP, phi,
                  is_efficient: Callable[[Sequence[Fraction]], bool],
                  n_model: Optional[int] = None) -> Optional[EdgeCandidate]:
    """Search ``Gamma_l`` for an efficient integer point; return the first found.

    Every candidate is validated against the *original* region ``D`` and the
    integrality requirements before its efficiency is tested, so a numerically
    or structurally wrong step can never contaminate the result -- the edge
    walk is an accelerator, never a source of truth.
    """
    n_model = n_model or problem.n
    for j in alternative_optima_columns(tableau, phi):
        for theta, x in walk_edge(tableau, j, n_model):
            if not problem.model.is_feasible(x):
                continue
            if not is_efficient(x):
                continue
            return EdgeCandidate(j, theta, x, phi(x))
    return None
