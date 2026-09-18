"""Exhaustive enumeration -- the reference implementation used for testing.

Nothing here is part of the algorithm: this module simply enumerates every
integer point of a box, keeps the feasible ones, filters the efficient ones by
the definition (Definition 1 of the paper) and maximises ``Phi`` over them.
It is exponential and only usable on toy instances, which is exactly the point:
it gives an independent ground truth for the algorithm.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from typing import List, Optional, Sequence, Tuple

from .model import MOILP
from .rational import F


@dataclass
class Enumeration:
    feasible: List[List[Fraction]] = field(default_factory=list)
    efficient: List[List[Fraction]] = field(default_factory=list)

    def best(self, phi) -> Tuple[Optional[List[Fraction]], Optional[Fraction]]:
        """``argmax`` and ``max`` of ``Phi`` over the efficient set."""
        best_x, best_v = None, None
        for x in self.efficient:
            try:
                v = phi(x)
            except ZeroDivisionError:      # Phi undefined there, skip the point
                continue
            if best_v is None or v > best_v:
                best_x, best_v = x, v
        return best_x, best_v


def enumerate_efficient_set(problem: MOILP, bounds: Sequence[int]) -> Enumeration:
    """Enumerate ``D`` inside ``0 <= x_j <= bounds[j]`` and extract ``E(P_D)``.

    A point ``x0`` is efficient when no feasible ``x1`` satisfies
    ``C x1 >= C x0`` with at least one strict inequality (Definition 1).
    """
    result = Enumeration()
    for combo in product(*(range(b + 1) for b in bounds)):
        x = [F(v) for v in combo]
        if problem.model.is_feasible(x):
            result.feasible.append(x)

    for x0 in result.feasible:
        if not any(problem.dominates(x1, x0) for x1 in result.feasible):
            result.efficient.append(x0)
    return result


# --------------------------------------------------------------------------
# The "naive" exact method -- the one the paper sets out to avoid
# --------------------------------------------------------------------------
@dataclass
class FullEnumeration:
    """Every non-dominated criterion vector of ``(P_D)`` with a representative."""

    vectors: List[List[Fraction]] = field(default_factory=list)
    representatives: List[List[Fraction]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.vectors)


def enumerate_nondominated(problem: MOILP, max_points: int = 100_000) -> FullEnumeration:
    """Generate the whole non-dominated set by repeated Sylva-Crema cuts.

    Start from ``D``; pick any feasible point, turn it into an efficient one
    with the test of Theorem 1, record its criterion vector, cut away
    everything it dominates, repeat until the region is empty.  Proposition 1
    guarantees that every point generated this way is efficient for ``(P_D)``,
    and the loop stops only once no non-dominated vector is left.

    This is *complete enumeration of the non-dominated set* -- exactly the work
    the algorithm of the paper is designed not to do.  It is kept here as an
    independent exact reference for instances that are too large for the
    box enumeration above, and as the baseline to measure the saving against.
    """
    from .efficiency import add_sylva_crema_cut, lower_bounds, test_efficiency
    from .milp import solve_linear_milp
    from .simplex import OPTIMAL

    M = lower_bounds(problem)
    region = problem.model.copy()
    # any linear direction works to pull out a feasible point; the sum of the
    # criteria tends to land on an efficient one straight away
    direction = [sum(col, F(0)) for col in zip(*problem.criteria)]

    result = FullEnumeration()
    for _ in range(max_points):
        probe = solve_linear_milp(region, direction + [F(0)] * (region.n - len(direction)))
        if probe.status != OPTIMAL:
            return result
        x = probe.x[:problem.n]
        test = test_efficiency(problem, x)
        x_eff = x if test.efficient else test.witness
        result.vectors.append(problem.C(x_eff))
        result.representatives.append(x_eff)
        region = add_sylva_crema_cut(region, problem, x_eff, M)
    raise RuntimeError("non-dominated set larger than max_points")


def maximize_by_full_enumeration(problem: MOILP, phi,
                                 max_points: int = 100_000):
    """Reference answer to ``(P_E)``: best ``Phi`` over every non-dominated slice.

    All the points sharing a non-dominated vector ``v`` are efficient, so the
    optimum of ``(P_E)`` is ``max_v max { Phi(x) : x in D, C x = v }``.
    Returns ``(x, value, enumeration)``.
    """
    from .efficiency import best_with_same_criterion

    enumeration = enumerate_nondominated(problem, max_points)
    best_x, best_v = None, None
    for rep in enumeration.representatives:
        res = best_with_same_criterion(problem.model, problem, rep, phi)
        if res.feasible and (best_v is None or res.objective > best_v):
            best_x, best_v = res.x, res.objective
    return best_x, best_v, enumeration


# --------------------------------------------------------------------------
# A fast exact verifier for instances too large for the naive methods
# --------------------------------------------------------------------------
def _fast_feasible_points(problem: MOILP, bounds: Sequence[int]) -> List[tuple]:
    """Integer points of ``D`` inside the box, by depth-first search with pruning.

    For every constraint ``a'x <= r`` and every position ``k`` one knows the
    smallest contribution the *remaining* variables can still make,
    ``sum_{j>=k} min(0, a_j * ub_j)``.  A partial assignment whose running sum
    already exceeds ``r`` minus that quantity can never be completed into a
    feasible point, and the whole subtree is cut.  Constraints with ``>=`` or
    ``=`` are simply checked at the leaves.
    """
    n = problem.n
    rows, suffix_min, others = [], [], []
    for con in problem.model.constraints:
        if con.sense == "<=":
            a = list(con.coeffs)
            tail = [0] * (n + 1)
            for k in range(n - 1, -1, -1):
                tail[k] = tail[k + 1] + min(0, a[k] * bounds[k])
            rows.append((a, con.rhs))
            suffix_min.append(tail)
        else:
            others.append(con)

    points: List[tuple] = []
    partial = [F(0)] * len(rows)
    assignment = [0] * n

    def descend(j: int) -> None:
        if j == n:
            x = [F(v) for v in assignment]
            if all(c.holds(x) for c in others):
                points.append(tuple(assignment))
            return
        for value in range(bounds[j] + 1):
            ok = True
            saved = partial[:]
            for k, (a, rhs) in enumerate(rows):
                partial[k] += a[j] * value
                if partial[k] + suffix_min[k][j + 1] > rhs:
                    ok = False
                    break
            if ok:
                assignment[j] = value
                descend(j + 1)
            partial[:] = saved
        assignment[j] = 0

    descend(0)
    return points


def best_over_efficient_set_by_scan(problem: MOILP, phi, bounds: Sequence[int]):
    """Exact answer to ``(P_E)`` by scanning the feasible points in ``Phi`` order.

    The efficient set never has to be built: sort ``D`` by decreasing ``Phi``
    and return the **first** point that survives the dominance test.  Every
    point ranked above it is infeasible for ``(P_E)`` because it is dominated,
    so the first efficient one in that order is the global optimum.

    Only a handful of dominance tests are usually needed, which makes this the
    practical reference for instances where enumerating ``E(P_D)`` -- let alone
    the ``|D|^2`` pairwise comparisons -- would be out of reach.
    Returns ``(x, value, n_feasible, n_tested)``.
    """
    raw = _fast_feasible_points(problem, bounds)
    criteria = [[int(c) if c.denominator == 1 else c for c in row]
                for row in problem.criteria]

    scored = []
    for pt in raw:
        x = [F(v) for v in pt]
        try:
            scored.append((phi(x), pt))
        except ZeroDivisionError:            # Phi undefined there
            continue
    scored.sort(key=lambda t: t[0], reverse=True)

    vectors = [tuple(sum(a * b for a, b in zip(row, pt)) for row in criteria)
               for pt in raw]

    for tested, (value, pt) in enumerate(scored, start=1):
        cx = tuple(sum(a * b for a, b in zip(row, pt)) for row in criteria)
        dominated = False
        for cy in vectors:
            if cy != cx and all(a >= b for a, b in zip(cy, cx)):
                dominated = True
                break
        if not dominated:
            return [F(v) for v in pt], value, len(raw), tested
    return None, None, len(raw), len(scored)
