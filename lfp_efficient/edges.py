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

Two practical notes, both measured rather than assumed.  The basis must be a
basis of ``D_l`` itself -- see :func:`clean_tableau_at` -- and the step must be
bounded by ``D`` rather than by the cut rows of ``D_l`` -- see
:func:`max_step_in`.  With both in place the step is sound and cheap, but it
stays rare: most of ``Gamma_l`` consists of columns that move only the
machinery the cuts brought with them, and those cannot produce anything but
``x_l``.  One limit is left standing: at a degenerate vertex a single basis
exposes only some of the incident edges, and the one built here is picked by a
fixed rule.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import floor
from typing import Callable, List, Optional, Sequence, Tuple

from .model import EQ, GE, LE, MOILP, Model
from .rational import ZERO, dot, fmt, is_integral
from .simplex import FractionalPricing, Tableau, tableau_at


def clean_tableau_at(model: Model, x: Sequence[Fraction]) -> Optional[Tableau]:
    """A tableau of *model* alone, at the point *x*, with no branching rows.

    The paper reads its edges at the optimum of the truncated region ``D_l``.
    Branch & bound reaches that optimum through a node whose tableau also holds
    the bound rows of the path taken to it; those rows pin variables, their
    slacks sit basic at zero, and the ratio test of :func:`max_step` then
    returns ``theta0 = 0`` for nearly every edge.  Rebuilding the basis from
    ``D_l`` itself restores the room the edges are supposed to have.

    Returns ``None`` when *x* is not a vertex of the relaxation of *model* --
    an integer optimum need not be one -- in which case there is no basis to
    read and the caller simply skips the step.
    """
    A, b, n_structural = model.to_standard_form()
    if not A:
        return None
    n = len(A[0])
    values = list(x[:n_structural])
    values += [ZERO] * (n - len(values))
    for i, row in enumerate(A):
        residual = b[i] - dot(row[:n_structural], values[:n_structural])
        slack = next((j for j in range(n_structural, n) if row[j] != 0), None)
        if slack is None:
            continue                      # equality row: tableau_at checks it
        values[slack] = residual / row[slack]
        if values[slack] < 0:
            return None                   # x is not feasible for this row
    return tableau_at(A, b, values)


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


def edge_direction(tableau: Tableau, j: int, n_model: int) -> List[Fraction]:
    """The direction of the edge ``E_j``, restricted to the model variables.

    Definition 2 moves ``x_j`` up by one and each basic variable down by
    ``y_{k,ij}``; projected on the first ``n_model`` columns that is the vector
    returned here.  It is **zero** whenever the edge only stirs the machinery a
    cut brought with it -- its binaries, its slacks -- and leaves ``x`` where it
    was.  Those edges make up the bulk of ``Gamma_l`` once cuts accumulate, and
    they can only ever reproduce ``x_l``, which the efficiency test has just
    rejected; skipping them is what keeps the step cheap.
    """
    d = [ZERO] * n_model
    if j < n_model:
        d[j] = Fraction(1)
    for i, basic in enumerate(tableau.basis):
        if basic < n_model:
            d[basic] = -tableau.T[i][j]
    return d


def max_step_in(model: Model, x: Sequence[Fraction], d: Sequence[Fraction],
                cap: int = 1000) -> int:
    """Largest integer ``theta`` with ``x + theta d`` in the relaxation of *model*.

    The literal ``theta0`` of Definition 2 is the minimum ratio inside the
    *truncated* region, and it is read on every row of the tableau -- including
    the rows a Sylva-Crema cut added.  Those rows stop the walk long before
    ``x`` itself would leave ``D``: measured, 149 of the 154 edges with any room
    at all had a minimum ratio below 1, so the integer step floored to zero and
    the walk never started.

    Stopping at ``D`` instead loses nothing, because the walk does not need its
    points to be in the truncated region.  What the step claims is that an
    *efficient* point scores the round's upper bound, and that rests on three
    facts checked elsewhere: ``gamma_j = 0`` holds ``Phi`` constant along the
    whole edge, the point is validated against ``D``, and its efficiency is
    tested.  Whether it also satisfies the current cuts plays no part in that
    argument.  Nor can the wider walk smuggle in a stale answer: a point inside
    a cut slice is either dominated, or shares its criterion vector with the
    centre of the cut and was therefore already banked by ``Q``, so it cannot
    score above the incumbent -- let alone reach the upper bound.
    """
    limits = []
    for j, dj in enumerate(d):
        if dj < 0:
            limits.append(-x[j] / dj)
    for c in model.constraints:
        slope = dot(c.coeffs, d)          # zip stops at the shorter vector
        if slope == 0:
            continue
        lhs = dot(c.coeffs, x)
        if c.sense == LE and slope > 0:
            limits.append((c.rhs - lhs) / slope)
        elif c.sense == GE and slope < 0:
            limits.append((lhs - c.rhs) / -slope)
        elif c.sense == EQ:
            return 0
    if not limits:
        return cap
    top = min(limits)
    return cap if top > cap else max(0, int(floor(top)))


def walk_edge(tableau: Tableau, j: int, n_model: int,
              model: Optional[Model] = None) -> List[Tuple[int, List[Fraction]]]:
    """Integer points ``(theta, x)`` of the edge ``E_j``, far end first.

    The paper decreases ``theta`` from ``theta0`` down to 1, which probes the
    most distant -- hence the most promising -- alternative optima first.  With
    a *model* the step is bounded by that model rather than by the tableau's own
    rows; see :func:`max_step_in` for why the two differ and why the wider one
    is still sound.
    """
    if model is None:
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

    d = edge_direction(tableau, j, n_model)
    if not any(d):
        return []                      # the edge leaves x where it is
    base = tableau.solution()[:n_model]
    points = []
    for theta in range(max_step_in(model, base, d), 0, -1):
        candidate = [base[i] + theta * d[i] for i in range(n_model)]
        if any(v < 0 for v in candidate):
            continue
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
    try:
        columns = alternative_optima_columns(tableau, phi)
    except (ZeroDivisionError, ValueError):
        return None                      # Phi is not defined at this vertex
    for j in columns:
        for theta, x in walk_edge(tableau, j, n_model, problem.model):
            if not problem.model.is_feasible(x):
                continue
            try:
                value = phi(x)
            except ZeroDivisionError:
                # gamma_j = 0 keeps the ratio constant along the edge, but the
                # common denominator may still vanish at an isolated step
                continue
            if not is_efficient(x):
                continue
            return EdgeCandidate(j, theta, x, value)
    return None
