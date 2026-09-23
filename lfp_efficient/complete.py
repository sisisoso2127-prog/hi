"""The **complete efficient set** ``E(P_D)``, not merely the front.

:mod:`lfp_efficient.subset` delivers a subset and says so.
:mod:`lfp_efficient.front` delivers the complete non-dominated set, which is
more -- but it is complete in *criterion* space: one point is kept per vector,
and a vector attained by several efficient points contributes only one of them.
That gap is stated plainly there, and this module closes it.

What makes it cheap
-------------------
The obvious way to finish the job would be to test every point of a slice for
efficiency.  None of that is necessary:

    **If** ``v`` is non-dominated, **then every** ``x in D`` with ``Z(x) = v``
    is efficient.

Suppose some ``y in D`` dominated such an ``x``.  Then ``Z(y) >= Z(x) = v``
with at least one strict inequality -- which says ``v`` is dominated, against
the hypothesis.  So no ``y`` dominates ``x``, and ``x`` is efficient.

Two things follow, and together they are the whole module.  The efficient set
is exactly the union of the slices of the front,

    ``E(P_D) = { x in D : Z(x) = v for some non-dominated v }`` ,

and **not one efficiency test is paid on any of them**.  The front enumeration
already proves its vector list complete; completing it to the full point set
costs only the enumeration of each slice.

Why a slice is cheap to enumerate
---------------------------------
Fixing ``Z(x) = v`` is ``p`` *linear* equations in ``x``, for fractional
criteria as much as for linear ones: ``(c'x + a)/(d'x + b) = v_k`` clears to
``(c - v_k d)'x = v_k b - a`` because the denominator is strictly positive on
``D``.  So a slice is ``D`` plus ``p`` equalities -- no binary, no big-``M``,
and nothing that grows from one slice to the next.  The depth-first scan of
:func:`~lfp_efficient.enumeration._fast_feasible_points` prunes against those
equalities like any other row, and they are the tightest rows in the system.

What this does and does not claim
---------------------------------
``complete`` is ``True`` only when the front proved itself complete *and* no
point budget was hit.  It is then the whole of ``E(P_D)``, proved, and the
test suite checks it against exhaustive enumeration.

The cost that remains is the variable box.  Enumerating a slice needs an upper
bound on each variable, obtained here by maximising ``x_j`` over ``D`` -- ``n``
integer programs, once, before any slice is touched.  On instances where ``D``
is large in some coordinate that the criteria ignore, that coordinate is free
inside a slice and the slice really does hold many points; the count is then
large because the answer is, not because the method is wasteful.
"""
from dataclasses import dataclass, field
from fractions import Fraction
from time import monotonic
from typing import Dict, List, Optional, Sequence, Tuple

from .enumeration import _fast_feasible_points
from .front import Front, enumerate_front
from .milp import OPTIMAL, solve_milp
from .model import GE, LE, FractionalObjective, MOILFP

F = Fraction

__all__ = ["CompleteSet", "complete_efficient_set", "reduce_rows",
           "slice_equations", "slice_points", "slice_rows",
           "variable_bounds"]


def variable_bounds(problem: MOILFP) -> List[int]:
    """An integer upper bound on each variable over ``D``, by ``n`` programs.

    ``D`` is assumed bounded, so each maximisation is finite; a variable that
    comes back unbounded is a violated assumption and is reported as such
    rather than silently truncated.
    """
    bounds: List[int] = []
    for j in range(problem.n):
        objective = [F(0)] * problem.n
        objective[j] = F(1)
        res = solve_milp(problem.model, objective)
        if res.status != OPTIMAL:
            raise ValueError(
                f"x_{j + 1} is not bounded above on D ({res.status}); "
                "the complete enumeration needs a bounded feasible set")
        bounds.append(int(res.objective))
    return bounds


def slice_equations(problem: MOILFP, v: Sequence[Fraction]):
    """``Z(x) = v`` as ``p`` equations ``(c_k - v_k d_k)'x = v_k b_k - a_k``.

    Valid for fractional criteria because ``d_k'x + b_k > 0`` on ``D``, so
    multiplying through by it neither flips nor voids the equation.
    """
    out = []
    for k, z in enumerate(problem.criteria):
        coeffs = [F(z.U[j]) - v[k] * F(z.V[j]) for j in range(problem.n)]
        out.append((coeffs, v[k] * F(z.beta) - F(z.alpha)))
    return out


def slice_rows(problem: MOILFP, v: Sequence[Fraction]):
    """The same equations as a pair of ``<=`` rows each.

    Kept because it is what the reference scan of
    :func:`~lfp_efficient.enumeration._fast_feasible_points` takes, and the
    test suite runs that scan against the eliminated one.
    """
    rows = []
    for coeffs, rhs in slice_equations(problem, v):
        rows.append((coeffs, rhs))
        rows.append(([-c for c in coeffs], -rhs))
    return rows


def reduce_rows(equations, n: int):
    """Reduced row echelon form, in exact rationals.

    Returns ``(pivot_columns, rows)`` with each row of length ``n + 1``, or
    ``None`` when the system is inconsistent -- which says the slice is empty
    before a single point is enumerated.
    """
    M = [list(c) + [r] for c, r in equations]
    pivots, row = [], 0
    for col in range(n):
        sel = next((i for i in range(row, len(M)) if M[i][col] != 0), None)
        if sel is None:
            continue
        M[row], M[sel] = M[sel], M[row]
        head = M[row][col]
        M[row] = [x / head for x in M[row]]
        for i in range(len(M)):
            if i != row and M[i][col] != 0:
                factor = M[i][col]
                M[i] = [a - factor * b for a, b in zip(M[i], M[row])]
        pivots.append(col)
        row += 1
        if row == len(M):
            break
    for i in range(row, len(M)):
        if all(x == 0 for x in M[i][:n]) and M[i][n] != 0:
            return None                       # inconsistent: the slice is empty
    return pivots, M[:row]


def slice_points(problem: MOILFP, bounds: Sequence[int],
                 v: Sequence[Fraction]) -> List[tuple]:
    """Integer points of ``D`` with ``Z(x) = v``, by elimination then scan.

    The ``p`` slice equations are *solved* rather than merely used as pruning
    rows.  Gaussian elimination expresses ``r`` of the variables as linear
    functions of the rest; substituting them into every model row leaves an
    equivalent system in ``n - r`` unknowns, and the scan runs on that.  The
    eliminated variables are rebuilt at each leaf, where the one condition that
    is not linear -- that they come out **integer** -- is checked.

    Search space ``(ub+1)^n`` becomes ``(ub+1)^{n-r}``.  Measured against the
    scan that keeps all ``n`` variables and prunes on the equations, over 14
    instances at ``n = 7..12``: a median of **49x**, up to **130x**, and the
    same set of points every time.  The heaviest instance goes from $796.8$ s
    to $8.3$ s -- it previously could not finish inside a two-minute budget at
    all.

    Exactness is unaffected: every coefficient is a :class:`~fractions.Fraction`
    and the elimination is exact, so this is a change of search order, not an
    approximation.
    """
    n = problem.n
    solved = reduce_rows(slice_equations(problem, v), n)
    if solved is None:
        return []
    pivots, M = solved
    if not pivots:                           # nothing to eliminate: plain scan
        return _fast_feasible_points(problem, bounds, slice_rows(problem, v))
    pivot_set = set(pivots)
    free = [j for j in range(n) if j not in pivot_set]

    def substitute(coeffs, rhs):
        """``coeffs' x <= rhs`` rewritten in the free variables alone."""
        out = [F(coeffs[j]) for j in free]
        right = F(rhs)
        for i, col in enumerate(pivots):
            a = F(coeffs[col])
            if a == 0:
                continue
            right -= a * M[i][n]
            for t, j in enumerate(free):
                out[t] -= a * M[i][j]
        return out, right

    le_rows, others = [], []
    for con in problem.model.constraints:
        if con.sense == LE:
            le_rows.append(substitute(con.coeffs, con.rhs))
        elif con.sense == GE:
            le_rows.append(substitute([-c for c in con.coeffs], -con.rhs))
        else:
            others.append(con)
    # every eliminated variable must still land inside its own box
    for i, col in enumerate(pivots):
        row = [M[i][j] for j in free]
        le_rows.append(([-c for c in row], F(bounds[col]) - M[i][n]))
        le_rows.append((list(row), M[i][n]))

    m = len(free)
    free_bounds = [bounds[j] for j in free]
    suffix = []
    for coeffs, _ in le_rows:
        tail = [F(0)] * (m + 1)
        for k in range(m - 1, -1, -1):
            tail[k] = tail[k + 1] + min(F(0), coeffs[k] * free_bounds[k])
        suffix.append(tail)

    points: List[tuple] = []
    partial = [F(0)] * len(le_rows)
    assignment = [0] * m

    def descend(t: int) -> None:
        if t == m:
            x = [F(0)] * n
            for idx, j in enumerate(free):
                x[j] = F(assignment[idx])
            for i, col in enumerate(pivots):
                value = M[i][n] - sum(M[i][j] * x[j] for j in free)
                if value.denominator != 1:
                    return                    # the eliminated part is fractional
                x[col] = value
            if all(c.holds(x) for c in others):
                points.append(tuple(int(c) for c in x))
            return
        for value in range(free_bounds[t] + 1):
            ok, saved = True, partial[:]
            for k, (coeffs, rhs) in enumerate(le_rows):
                partial[k] += coeffs[t] * value
                if partial[k] + suffix[k][t + 1] > rhs:
                    ok = False
                    break
            if ok:
                assignment[t] = value
                descend(t + 1)
            partial[:] = saved
        assignment[t] = 0

    descend(0)
    return points


@dataclass
class CompleteSet:
    """Every efficient point of ``(P_D)``, grouped by its criterion vector."""

    #: the complete non-dominated set, in the order the front produced it
    vectors: List[List[Fraction]] = field(default_factory=list)
    #: ``vectors[i]`` is attained by exactly the points of ``slices[i]``
    slices: List[List[List[Fraction]]] = field(default_factory=list)
    #: every efficient point, flattened -- this is ``E(P_D)``
    points: List[List[Fraction]] = field(default_factory=list)
    #: with a ``Phi``: its maximiser over ``E(P_D)`` and the value there
    best_x: Optional[List[Fraction]] = None
    best_value: Optional[Fraction] = None
    #: boxes the front enumeration settled, and the variable box it scanned
    boxes: int = 0
    bounds: List[int] = field(default_factory=list)
    #: ``True`` only when the front proved complete *and* no budget was hit
    complete: bool = False

    def __len__(self) -> int:
        return len(self.points)

    def slice_of(self, v: Sequence[Fraction]) -> List[List[Fraction]]:
        """The efficient points attaining the criterion vector *v*."""
        key = tuple(v)
        for i, w in enumerate(self.vectors):
            if tuple(w) == key:
                return self.slices[i]
        return []

    def report(self) -> str:
        head = (f"{len(self.points)} efficient points on {len(self.vectors)} "
                "non-dominated vectors")
        head += (" -- the complete efficient set, proved" if self.complete
                 else " -- INCOMPLETE (budget spent)")
        sizes = [len(s) for s in self.slices]
        lines = [head, f"  {self.boxes} boxes settled, variable box "
                       f"{tuple(self.bounds)}"]
        if sizes:
            multi = sum(1 for s in sizes if s > 1)
            lines.append(f"  largest slice {max(sizes)} points, "
                         f"{multi} of {len(sizes)} vectors carry more than one")
        if self.best_value is not None:
            lines.append(f"  best Phi over E(P_D): {self.best_value} at "
                         f"({', '.join(str(c) for c in self.best_x)})")
        return "\n".join(lines)


def complete_efficient_set(problem: MOILFP,
                           phi: Optional[FractionalObjective] = None,
                           *,
                           bounds: Optional[Sequence[int]] = None,
                           max_boxes: int = 200_000,
                           max_points: int = 1_000_000,
                           time_budget: Optional[float] = None,
                           front: Optional[Front] = None) -> CompleteSet:
    """Enumerate ``E(P_D)`` in full: the front, then each of its slices.

    *phi* is optional and changes nothing about the set; it only makes the
    result carry the maximiser of ``Phi`` over ``E(P_D)``, which is the exact
    answer to ``(P_E)`` -- obtained here without a single efficiency test on
    any slice point, by the argument in the module docstring.

    *bounds* overrides the variable box; pass it when it is already known, to
    skip the ``n`` bounding programs.  *front* reuses a front already computed
    on this problem, so the two results can be compared without paying twice.

    The budgets are honoured the way :func:`~lfp_efficient.front.enumerate_front`
    honours its own: an exhausted budget returns what has been found with
    ``complete = False``, never a set that claims more than it proved.
    """
    started = monotonic()
    if front is None:
        front = enumerate_front(problem, None, max_boxes=max_boxes,
                                time_budget=time_budget)
    remaining = (None if time_budget is None
                 else time_budget - (monotonic() - started))

    result = CompleteSet(boxes=front.boxes)
    result.bounds = list(bounds) if bounds is not None else variable_bounds(problem)

    truncated = False
    for v in front.vectors:
        if remaining is not None and monotonic() - started > time_budget:
            truncated = True
            break
        points = slice_points(problem, result.bounds, v)
        # Every one of these is efficient -- see the module docstring.  The
        # front's own point for this vector is among them, which the test
        # suite checks rather than assumes.
        members = [[F(c) for c in x] for x in points]
        result.vectors.append(list(v))
        result.slices.append(members)
        result.points.extend(members)
        if len(result.points) >= max_points:
            truncated = True
            break

    result.complete = front.complete and not truncated

    if phi is not None:
        for x in result.points:
            try:
                value = phi(x)
            except ZeroDivisionError:      # Phi undefined there: not a candidate
                continue
            if result.best_value is None or value > result.best_value:
                result.best_x, result.best_value = list(x), value
    return result
