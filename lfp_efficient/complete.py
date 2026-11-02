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
**The deliverable is not new.**  Chergui and Moulai (2008) generated the whole
efficient set of a MOILFP by branch-and-cut, and every paper in this
literature cites them; this module claims no priority for producing ``E(P_D)``.
What it offers is a different *route* -- the set assembled from the front by
slices rather than reached point by point -- and the difference is a count
rather than an adjective.  :mod:`lfp_efficient.decision_space` implements the
other route with the efficient cut those papers share, and on thirty instances
where both return the same ``E(P_D)``: **zero** efficiency tests here against a
median of 119 and a maximum of 1342 there.  That count is deterministic; the
median 2.99x in seconds is not, and its censoring falls only on the other side.

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
from math import gcd
from fractions import Fraction
from time import monotonic
from typing import Dict, List, Optional, Sequence, Tuple

from .efficiency import assert_bounded, assert_denominators_positive
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
    cached = getattr(problem, "_variable_bounds", None)
    if cached is not None:
        return list(cached)
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
    problem._variable_bounds = list(bounds)   # the entry points re-ask freely
    return bounds


def box_shortfall(problem: MOILFP, bounds: Sequence[int]):
    """Variables whose true reach on ``D`` exceeds the box: ``(j, reach, allowed)``.

    A box smaller than ``D`` is the single most dangerous argument in this
    package, because nothing downstream can detect it: every routine that takes
    a box searches it honestly and reports a complete-looking answer about a
    feasible set that is not ``D``.  Four separate routines were caught doing
    exactly that.  An empty list means the box is a valid enclosure.

    ``variable_bounds`` caches on the problem, so asking costs nothing after
    the first call and every entry point can afford to ask.
    """
    true_bounds = variable_bounds(problem)
    return [(j, true_bounds[j], bounds[j])
            for j in range(problem.n) if bounds[j] < true_bounds[j]]


def assert_box_contains(problem: MOILFP, bounds: Sequence[int],
                        consequence: str) -> None:
    """Raise unless ``bounds`` encloses ``D``, naming what the clipping costs."""
    narrow = box_shortfall(problem, bounds)
    if narrow:
        raise ValueError(
            "the supplied variable box does not contain D: "
            + ", ".join(f"x_{j + 1} reaches {r} but the box allows {a}"
                        for j, r, a in narrow)
            + ".  " + consequence)


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


#: how many propagation sweeps the slice scan runs before giving up on
#: further tightening.  One sweep can enable the next, but the sequence
#: converges fast in practice and the cap keeps a pathological system
#: from spending the budget on bookkeeping.
_PROPAGATION_SWEEPS = 8


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


def points_satisfying(problem: MOILFP, bounds: Sequence[int],
                      equations, inequalities=()) -> List[tuple]:
    """Integer points of ``D`` with ``Z(x) = v``, by elimination then scan.

    Takes a conjunction of linear *equations* and, optionally, linear
    ``a'x <= b`` *inequalities*.  The inequalities are what let a level set be
    a finite union of **polyhedra** rather than only of affine subspaces: see
    :mod:`lfp_efficient.levelsets`, where that is the difference between
    reaching ``|c'x+a|/(d'x+b)`` and reaching ``max`` or ``min`` of ratios.

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
    solved = reduce_rows(equations, n)
    if solved is None:
        return []
    pivots, M = solved
    if not pivots:                           # nothing to eliminate: plain scan
        rows = []
        for coeffs, rhs in equations:
            rows.append((list(coeffs), rhs))
            rows.append(([-c for c in coeffs], -rhs))
        rows.extend((list(c), r) for c, r in inequalities)
        return _fast_feasible_points(problem, bounds, rows)
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
    # extra ``a'x <= b`` rows that came with the branch, not with the model --
    # they go through the same substitution as the model's own rows
    for coeffs, rhs in inequalities:
        le_rows.append(substitute(coeffs, rhs))
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

    # Clear the denominators, once, before the scan.
    #
    # The elimination above leaves every row rational, and the scan then spent
    # its whole life in ``Fraction``: a profile at n=10 put 76% of the run in
    # ``fractions.py``, with 8.6 million Fraction constructions for 67 boxes --
    # the arithmetic, not the search, was the cost.  Multiplying a ``a'x <= b``
    # row by a positive integer leaves it the same constraint, so scaling each
    # row by the lcm of its denominators makes the hot loop run on Python
    # ``int``: exact as before, with none of the boxing.  Nothing about which
    # points are returned changes, which is what the suite checks.
    #
    # Measured, same instances through both versions, 16 configurations at a
    # 40 s budget: median 1.18x (range 0.71x to 4.38x) on the 9 that finish
    # either way, TWO crossing from censored to complete (n=12 p=2, n=10 p=3),
    # none regressing, and identical |E| and box counts throughout.  So the
    # operating range is the implementation's and not the method's -- but it
    # moved by little, and the remaining 74% is now in the simplex pivot,
    # where the lever would be a fraction-free (Bareiss) tableau.
    int_rows = []
    for coeffs, rhs in le_rows:
        scale = 1
        for value in (*coeffs, rhs):
            scale = scale * value.denominator // gcd(scale, value.denominator)
        int_rows.append(([int(c * scale) for c in coeffs], int(rhs * scale)))

    # Narrow each free variable's own interval before scanning anything.
    #
    # The scan loops ``range(lo[t], hi[t] + 1)`` at depth ``t``, so the tree it
    # walks is the product of those interval widths: tightening them pays
    # multiplicatively, where tightening the pruning test pays once per node.
    # Measured before this existed: 198,614 nodes to return 27 points on one
    # n=10 instance, about 7,400 nodes per point.  Most of that tree could not
    # contain a feasible assignment at all, and a row says so without being
    # searched.
    #
    # The rule is the textbook one.  For a row ``a'x <= b`` and an index k,
    # every other variable contributes at least ``min(a_j lo_j, a_j hi_j)``, so
    #
    #     a_k > 0  =>  x_k <= floor((b - rest_min) / a_k)
    #     a_k < 0  =>  x_k >= ceil ((b - rest_min) / a_k)
    #
    # which only ever removes values no feasible point can take: the returned
    # set is unchanged, and the suite checks that against brute force.  Sweeps
    # run to a fixpoint, capped, because one tightening can enable the next.
    # An empty interval means the slice holds no integer point and the scan is
    # skipped entirely -- the cheapest case of all.
    lo = [0] * m
    hi = list(free_bounds)
    for _ in range(_PROPAGATION_SWEEPS):
        changed = False
        for coeffs, rhs in int_rows:
            nz = [k for k in range(m) if coeffs[k]]
            if not nz:
                continue
            contributions = [min(coeffs[k] * lo[k], coeffs[k] * hi[k])
                             for k in range(m)]
            total = sum(contributions)
            for k in nz:
                rest = total - contributions[k]
                slack = rhs - rest
                a = coeffs[k]
                if a > 0:
                    bound = slack // a                      # floor, a > 0
                    if bound < hi[k]:
                        hi[k] = bound
                        changed = True
                else:
                    # ceil(slack / a) for a < 0, written on positive ints:
                    # with q = -a > 0 that is ceil(-slack / q) = -(slack // q)
                    bound = -(slack // (-a))
                    if bound > lo[k]:
                        lo[k] = bound
                        changed = True
            if any(lo[k] > hi[k] for k in nz):
                return []                    # no integer point on this slice
        if not changed:
            break
    if any(lo[k] > hi[k] for k in range(m)):
        return []

    # Lift the integrality of the eliminated variables out of the leaf.
    #
    # A node-count by depth, on one n=10 instance, read:
    #
    #     depth 6: 76,295    depth 7: 82,333    depth 8 (leaf): 30
    #
    # The tree grew to eighty thousand nodes and the last level destroyed all
    # but thirty of them.  What destroyed them was not a row -- the rows were
    # already satisfied -- but the condition checked only at the leaf: that
    # each eliminated variable
    #
    #     x_col = M[i][n] - sum_j M[i][j] x_j
    #
    # comes out an INTEGER.  That is a congruence, and a congruence can be
    # tested long before its last term is known.
    #
    # Scale row i by L_i, the lcm of its denominators, so every coefficient is
    # integral; the requirement becomes
    #
    #     target_i - sum_j c_ij x_j  ==  0   (mod L_i) .
    #
    # With variables 0..t assigned, whatever the rest contribute is a multiple
    # of g = gcd(L_i, c_i,t+1, ..., c_i,m-1) -- every term is -- so
    #
    #     (target_i - partial_i)  ==  0   (mod g)
    #
    # is NECESSARY, and it can be checked the moment variable t is fixed.  At
    # t = m-1 the tail is empty, g is L_i, and the test is exactly the leaf
    # check: this strictly generalises it rather than approximating it.  No
    # assignment that could complete to an integer point is ever refused, which
    # is what the brute-force comparisons in the suite verify.
    cong = []                       # (coefficients, target, tail gcds) per row
    for i in range(len(pivots)):
        scale = 1
        entries = [M[i][j] for j in free] + [M[i][n]]
        for value in entries:
            scale = scale * value.denominator // gcd(scale, value.denominator)
        coeffs_i = [int(M[i][j] * scale) for j in free]
        target_i = int(M[i][n] * scale)
        tail = [0] * (m + 1)
        tail[m] = scale
        for k in range(m - 1, -1, -1):
            tail[k] = gcd(tail[k + 1], coeffs_i[k])
        cong.append((coeffs_i, target_i, tail))

    suffix = []
    for coeffs, _ in int_rows:
        tail = [0] * (m + 1)
        for k in range(m - 1, -1, -1):
            tail[k] = tail[k + 1] + min(coeffs[k] * lo[k], coeffs[k] * hi[k])
        suffix.append(tail)

    points: List[tuple] = []
    partial = [0] * len(int_rows)
    cong_partial = [0] * len(cong)
    assignment = list(lo)
    rows = list(enumerate(int_rows))
    cong_rows = list(enumerate(cong))

    last = m - 1

    def residue_window(t: int):
        """At the last free variable, the values a congruence still permits.

        Testing the congruence one value at a time is the wrong use of it.  It
        says that
            a x  ==  target - partial   (mod L) ,
        which with g = gcd(a, L) is solvable only when g divides the right-hand
        side, and then pins x to a single residue class modulo L/g.  So the
        deepest loop need not walk its interval at all: it can step through one
        class, which is L/g times shorter.

        Returns ``(start, step)`` for the tightest row, or ``None`` when some
        row makes the remaining variable impossible -- in which case the whole
        subtree is empty and the caller skips it.
        """
        best = (lo[t], 1)
        for i, (coeffs_i, target_i, tail) in cong_rows:
            a, L = coeffs_i[t], tail[m]
            want = (target_i - cong_partial[i]) % L
            if a == 0:
                if want:
                    return None               # nothing left can fix this row
                continue
            g = gcd(a, L)
            if want % g:
                return None
            modulus = L // g
            if modulus <= best[1]:
                continue
            inverse = pow(a // g, -1, modulus)
            x0 = (want // g) * inverse % modulus
            start = lo[t] + (x0 - lo[t]) % modulus
            best = (start, modulus)
        return best

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
        suffix_t = t + 1
        if t == last:
            window = residue_window(t)
            if window is None:
                return
            start, step = window
        else:
            start, step = lo[t], 1
        for value in range(start, hi[t] + 1, step):
            ok, saved, monotone = True, partial[:], False
            for k, (coeffs, rhs) in rows:
                a = coeffs[t]
                partial[k] += a * value
                if partial[k] + suffix[k][suffix_t] > rhs:
                    ok = False
                    # ``partial[k]`` is a * value plus a constant, and values
                    # are tried in increasing order, so with a > 0 this row is
                    # violated for every larger value as well.  Stopping here
                    # skips only assignments that provably violate it; with
                    # a <= 0 the row can recover and the loop must go on.
                    monotone = a > 0
                    break
            if ok:
                saved_cong = cong_partial[:]
                for i, (coeffs_i, target_i, tail) in cong_rows:
                    cong_partial[i] += coeffs_i[t] * value
                    g = tail[suffix_t]
                    if g and (target_i - cong_partial[i]) % g:
                        ok = False
                        break
                if ok:
                    assignment[t] = value
                    descend(t + 1)
                cong_partial[:] = saved_cong
            partial[:] = saved
            if monotone:
                break
        assignment[t] = lo[t]
        return

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


def slice_points(problem: MOILFP, bounds: Sequence[int],
                 v: Sequence[Fraction]) -> List[tuple]:
    """Integer points of ``D`` with ``Z(x) = v``.

    The slice of a *linear fractional* criterion vector: one equation per
    criterion, solved by :func:`points_satisfying`.  See
    :mod:`lfp_efficient.levelsets` for why one equation per criterion is
    exactly the fractional case and what lies just outside it.
    """
    return points_satisfying(problem, bounds, slice_equations(problem, v))


def _assert_front_belongs(problem: MOILFP, front) -> None:
    """A front handed in must be this problem's.  Free, and exact.

    ``front=`` exists so the front and the slices can be timed apart, or a
    front reused.  Nothing tied it to the problem: handing over a front built
    for a *different* criterion matrix returned ``|E| = 0`` with
    ``complete = True``, on an instance with six efficient points -- the
    slices simply found nothing, since no feasible point attains a foreign
    vector.

    Each recorded point must be feasible here and attain its own vector under
    *this* problem's ``Z``.  That is arithmetic over ``|F|`` points and no
    program at all.
    """
    for index, (vector, point) in enumerate(zip(front.vectors, front.points)):
        if not problem.model.is_feasible(point):
            raise ValueError(
                f"the front handed in does not belong to this problem: its "
                f"point {index + 1} is not feasible here")
        if problem.Z(point) != list(vector):
            raise ValueError(
                f"the front handed in does not belong to this problem: its "
                f"point {index + 1} has Z = {problem.Z(point)} here, not the "
                f"{list(vector)} recorded with it")


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
    assert_bounded(problem)
    assert_denominators_positive(problem)
    started = monotonic()
    if front is not None:
        _assert_front_belongs(problem, front)
    if front is None:
        front = enumerate_front(problem, None, max_boxes=max_boxes,
                                time_budget=time_budget)
    remaining = (None if time_budget is None
                 else time_budget - (monotonic() - started))

    result = CompleteSet(boxes=front.boxes)
    if bounds is None:
        result.bounds = variable_bounds(problem)
    else:
        # A box smaller than D loses efficient points from every slice it
        # clips, and the run still reports ``complete``: supplying
        # ``[2, 2, 1]`` where ``[2, 2, 2]`` is right returned 4 points of 6
        # and called them the whole set.  Checking costs nothing after the
        # first call, since variable_bounds caches on the problem.
        assert_box_contains(problem, bounds,
                            "Slices clipped by it lose efficient points silently")
        result.bounds = list(bounds)

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
