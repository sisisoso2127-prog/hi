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

from .model import Constraint, MOILFP
from .rational import F, vec


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


def enumerate_efficient_set(problem: MOILFP, bounds: Sequence[int]) -> Enumeration:
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


def enumerate_nondominated(problem: MOILFP, max_points: int = 100_000) -> FullEnumeration:
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
    from .efficiency import add_dominance_cut, repair_to_efficient, test_efficiency
    from .milp import solve_linear_milp
    from .simplex import OPTIMAL

    region = problem.model.copy()
    # any linear direction works to pull out a feasible point; the sum of the
    # criteria numerators tends to land on an efficient one straight away
    direction = [sum(col, F(0)) for col in zip(*(z.U for z in problem.criteria))]

    result = FullEnumeration()
    for _ in range(max_points):
        probe = solve_linear_milp(region, direction + [F(0)] * (region.n - len(direction)))
        if probe.status != OPTIMAL:
            return result
        x = probe.x[:problem.n]
        x_eff = repair_to_efficient(problem, x)
        result.vectors.append(problem.Z(x_eff))
        result.representatives.append(x_eff)
        region = add_dominance_cut(region, problem, x_eff)
    raise RuntimeError("non-dominated set larger than max_points")


def maximize_by_full_enumeration(problem: MOILFP, phi,
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
def _fast_feasible_points(problem: MOILFP, bounds: Sequence[int],
                          extra_le_rows: Sequence = ()) -> List[tuple]:
    """Integer points of ``D`` inside the box, by depth-first search with pruning.

    For every constraint ``a'x <= r`` and every position ``k`` one knows the
    smallest contribution the *remaining* variables can still make,
    ``sum_{j>=k} min(0, a_j * ub_j)``.  A partial assignment whose running sum
    already exceeds ``r`` minus that quantity can never be completed into a
    feasible point, and the whole subtree is cut.  Constraints with ``>=`` or
    ``=`` are simply checked at the leaves.

    ``extra_le_rows`` are additional ``(coeffs, rhs)`` pairs meaning
    ``coeffs'x <= rhs``.  They take part in the pruning exactly like the
    model's own rows, which is what lets a caller enumerate a *slice* of ``D``
    without paying for the whole region.
    """
    n = problem.n
    rows, suffix_min, others = [], [], []
    constraints = list(problem.model.constraints)
    constraints += [Constraint(vec(c), "<=", F(r)) for c, r in extra_le_rows]
    for con in constraints:
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


def best_over_efficient_set_by_scan(problem: MOILFP, phi, bounds: Sequence[int]):
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

    # Z_k(x) >= Z_k(y)  <=>  N_k(x) D_k(y) >= N_k(y) D_k(x)  (denominators > 0),
    # so dominance stays exact integer arithmetic on ratios too.
    def pairs(pt):
        x = [F(v) for v in pt]
        return tuple((problem.numerator(k, x), problem.denominator(k, x))
                     for k in range(problem.p))

    scored = []
    for pt in raw:
        x = [F(v) for v in pt]
        try:
            scored.append((phi(x), pt))
        except ZeroDivisionError:            # Phi undefined there
            continue
    scored.sort(key=lambda t: t[0], reverse=True)

    vectors = [pairs(pt) for pt in raw]

    for tested, (value, pt) in enumerate(scored, start=1):
        cx = pairs(pt)
        dominated = False
        for cy in vectors:
            if cy != cx and all(ny * dx >= nx * dy
                                for (nx, dx), (ny, dy) in zip(cx, cy)):
                dominated = True
                break
        if not dominated:
            return [F(v) for v in pt], value, len(raw), tested
    return None, None, len(raw), len(scored)


# --------------------------------------------------------------------------
# Certification: proving an answer right without enumerating E(P_D) or D
# --------------------------------------------------------------------------
@dataclass
class Certificate:
    """Outcome of :func:`certify_optimum`."""

    valid: bool
    #: feasible points strictly better than the claimed value -- all dominated
    challengers: int = 0
    #: how many of them needed an exact efficiency test (the rest were settled
    #: by a dominance witness already in hand)
    tests: int = 0
    reason: str = ""

    def __str__(self) -> str:
        if not self.valid:
            return f"NOT PROVED: {self.reason}"
        return (f"proved: the point is efficient and each of the {self.challengers} "
                f"feasible points with a strictly greater Phi is dominated "
                f"({self.tests} efficiency test(s) needed)")


def certify_optimum(problem: MOILFP, phi, x_opt: Sequence[Fraction],
                    value: Fraction, bounds: Sequence[int]) -> Certificate:
    """Prove that *x_opt* solves ``(P_E)`` -- without building ``D`` or ``E(P_D)``.

    Two things make an answer to ``(P_E)`` correct:

    1. ``x_opt`` is feasible and **efficient**, so it is admissible;
    2. every feasible point with ``Phi(x) > value`` is **dominated**, so none of
       them is admissible.

    Both are checked here directly. The second one only has to look at
    ``{ x in D : Phi(x) > value }``, which the depth-first search prunes down to
    a small set -- the full feasible region is never enumerated, and neither is
    the efficient set. A challenger is discarded as soon as some efficient point
    already in hand dominates it; only the survivors pay an exact efficiency
    test (Theorem 1), and each test that comes back negative hands over a new
    efficient point that helps settle the following ones.

    This shares no logic with the algorithm -- no cut, no bound, no reduced
    gradient -- so it is an independent proof, and it is the only one of the
    reference methods that stays affordable once ``|D|`` runs into the hundreds
    of thousands.
    """
    from .efficiency import repair_to_efficient, test_efficiency

    x_opt = list(x_opt)
    if not problem.model.is_feasible(x_opt):
        return Certificate(False, reason="the claimed point is not feasible")
    if phi(x_opt) != value:
        return Certificate(False, reason="the claimed point does not realise the value")
    if not test_efficiency(problem, x_opt).efficient:
        return Certificate(False, reason="the claimed point is not efficient")

    # "Phi(x) > value" is a *linear* condition whenever the denominator is
    # positive:  (U'x + alpha) / (V'x + beta) > v  <=>  (U - vV)'x + (alpha -
    # v*beta) > 0.  Handing it to the search as one more row prunes the tree
    # down to the challengers themselves, so the certificate never walks the
    # whole feasible region.  When the denominator can change sign the
    # equivalence breaks and the row is simply left out -- slower, still exact,
    # since every point is re-tested with Phi below anyway.
    from .milp import denominator_stays_positive

    extra = ()
    if denominator_stays_positive(problem.model, phi):
        lifted = phi.lift(problem.n)
        w = [u - value * v for u, v in zip(lifted.U, lifted.V)]
        rhs = lifted.alpha - value * lifted.beta
        extra = ([[-c for c in w], rhs],)        # -(U - vV)'x <= alpha - v*beta

    witnesses = [problem.Z(x_opt)]          # criterion vectors of known efficient points
    challengers = tests = 0

    for pt in _fast_feasible_points(problem, bounds, extra):
        x = [F(v) for v in pt]
        try:
            if phi(x) <= value:
                continue
        except ZeroDivisionError:           # Phi undefined: not a competitor
            continue
        challengers += 1

        cx = problem.Z(x)
        if any(all(a >= b for a, b in zip(w, cx)) and w != cx for w in witnesses):
            continue                        # dominated by something already known

        tests += 1
        outcome = test_efficiency(problem, x)
        if outcome.efficient:
            return Certificate(False, challengers, tests,
                               reason=f"the efficient point {pt} has Phi > {value}")
        witnesses.append(problem.Z(repair_to_efficient(problem, outcome.witness)))

    return Certificate(True, challengers, tests)
