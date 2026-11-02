"""Enumerating ``E(P_D)`` in *decision* space, with the efficient cut.

Why this module exists
----------------------
Every performance figure in this project compares criterion space against
decision space **on the optimisation task** -- maximising one function over
``E(P_D)`` -- against the method of :mod:`lfp_efficient.algorithm`, whose cut is
the Sylva--Crema disjunction.  Nothing here ever compared the *complete-set*
route of :mod:`lfp_efficient.complete` against a decision-space method that
also produces all of ``E(P_D)``, because none was implemented.

That gap matters, because such a method exists and predates this work.  Chergui
and Moulaï (2008) generate the whole efficient set of a MOILFP by
branch-and-cut; every paper in this literature cites them, and one states it
without ambiguity: *authors avoid trying to generate the whole efficient set,
however Chergui and Moulaï (2008) used a branch-and-cut procedure to generate
the whole efficient set*.  So "this work produces all of ``E(P_D)``" is not a
first, and the honest claim is about the route.  A claim about a route has to
be measured against the other route.

What this is, and what it is not
--------------------------------
**It is not a reimplementation of Chergui and Moulaï (2008).**  That paper could
not be reached from the environment this was written in, so its own text was
never read: the node objective it drives each relaxation with, in particular,
is unknown here.

**It is the efficient-cut mechanism as four papers that use it state it**, read
in full: Drici, Ouaïl and Moulaï (2018), Ouaïl and Chergui (2019), Chaiblaine,
Moulaï and Cherfaoui (2020), and Moulaï and Mekhilef (2021).  All four
attribute the cut to Chergui and Moulaï (2008) and all four write it the same
way.  With ``N_l`` the non-basic columns at the current vertex and
``gamma^k`` the reduced gradient of criterion ``Z_k``,

    H_l = { j in N_l : exists k, gamma^k_j > 0 }
          union
          { j in N_l : gamma^k_j = 0 for every k }

and the cut is the single row

    sum_{j in H_l} x_j >= 1 .

**No binaries and no big-M**: that is the whole point of the comparison, and it
is why the cost model this project stated about "decision space" had to be
narrowed to the Sylva--Crema branch of the literature.

The two theorems the papers prove about it, and which this code relies on:

* an efficient ``x != x*`` that lies in ``X_l`` still lies in ``X_{l+1}``, so
  the cut removes ``x*`` and points dominated by it and nothing else;
* ``H_l = {}`` means ``x*`` is an ideal point of ``X_l``, so ``X_l \\ {x*}``
  holds no efficient solution and the region is finished.

Correctness here is not taken on trust from either: :func:`enumerate_by_cuts`
is checked against brute force on every instance the suite runs it on, which is
what would catch a misreading of the formula above.

The tree, and why it has to be a tree
------------------------------------
A first attempt here kept one global region, asked branch & bound for an
integer point in it and cut that point away.  **That is wrong**, and the
instance in the test suite caught it: the cut's validity theorem is stated
about ``X_l``, the region *of the node*, so a cut earned inside one branch is
not valid outside it.  Adding it globally removes efficient points living in
other branches.

So this is a genuine branch-and-cut tree.  Each node carries the base model
plus the branch rows and the cuts accumulated along its own path:

* solve the node's **LP relaxation**;
* infeasible -> fathom;
* an integer-constrained coordinate fractional -> branch
  ``x_r <= floor(v)`` / ``x_r >= ceil(v)`` and push both children;
* integer -> record the point if the exact efficiency test accepts it, read
  ``H_l`` off this node's own tableau, fathom when ``H_l`` is empty, otherwise
  push the node again with its cut appended.

The node objective steers only the order points are met in, not which points
are met.  ``objective=None`` drives it with ``Z_1``, as Chaiblaine et al. drive
theirs with ``f_1``; the suite checks that a different objective returns the
same set.

Work counts
-----------
:class:`CutEnumeration` reports ``nodes`` (LP relaxations solved), ``cuts`` and
``tests`` (exact efficiency tests paid) -- all deterministic -- next to the
seconds.  ``tests`` is the one to watch against
:mod:`lfp_efficient.complete`, which pays **none**.
"""

from fractions import Fraction
from time import perf_counter
from typing import List, Optional, Sequence, Tuple

from .efficiency import test_efficiency
from .edges import clean_tableau_at, reduced_gradient
from .model import GE, LE, Model, MOILFP
from .rational import ZERO, dot

ONE = Fraction(1)


def _slack_map(model: Model) -> List[Optional[Tuple[int, str]]]:
    """Column -> ``(constraint index, sense)`` for the slack columns.

    ``to_standard_form`` appends one column per ``<=``/``>=`` row, in the order
    those rows appear.  A cut written over tableau columns can hold slack
    indices, and a slack is an affine function of ``x`` -- ``b_i - A_i x`` for a
    ``<=`` row, ``A_i x - b_i`` for a ``>=`` one -- so the cut comes back to a
    row over the model variables.  Equality rows get no column and are skipped.
    """
    out: List[Optional[Tuple[int, str]]] = []
    for i, c in enumerate(model.constraints):
        if c.sense in (LE, GE):
            out.append((i, c.sense))
    return out


def _cut_row(model: Model, columns: Sequence[int]) -> Tuple[List[Fraction], Fraction]:
    """``sum_{j in columns} x_j >= 1`` rewritten over the model variables.

    Returns ``(coeffs, rhs)`` meaning ``coeffs . x >= rhs``.  Structural
    columns contribute a ``+1``; a slack column contributes its own affine
    expression, which moves constants to the right-hand side.
    """
    n = model.n
    coeffs = [ZERO] * n
    rhs = ONE
    slacks = _slack_map(model)
    for j in columns:
        if j < n:
            coeffs[j] += ONE
            continue
        k = j - n
        if k >= len(slacks):
            continue                      # a column of a row this model lost
        i, sense = slacks[k]
        row = model.constraints[i]
        if sense == LE:                   # s_i = b_i - A_i x
            for t in range(n):
                coeffs[t] -= row.coeffs[t]
            rhs -= row.rhs
        else:                             # s_i = A_i x - b_i
            for t in range(n):
                coeffs[t] += row.coeffs[t]
            rhs += row.rhs
    return coeffs, rhs


def improving_columns(problem: MOILFP, tableau) -> List[int]:
    """``H_l``, exactly as the four papers write it.

    A non-basic column enters ``H_l`` when some criterion's reduced gradient is
    **positive** there -- moving along it can improve that criterion -- or when
    *every* criterion's reduced gradient is zero, which is the alternative-optima
    case that can reach a different point of the same criterion vector.

    The test is exact: these are ``Fraction`` comparisons against zero, with no
    tolerance anywhere.
    """
    gammas = []
    for k in range(problem.p):
        try:
            gammas.append(reduced_gradient(tableau, problem.criteria[k]))
        except ZeroDivisionError:
            return []                     # vanishing denominator: no basis to read
    out = []
    for j in tableau.nonbasic():
        if any(g[j] > 0 for g in gammas):
            out.append(j)
        elif all(g[j] == 0 for g in gammas):
            out.append(j)
    return out


class CutEnumeration:
    """What :func:`enumerate_by_cuts` returns.

    ``complete`` is ``False`` when a budget stopped the run, and then
    ``efficient`` is a *subset* of ``E(P_D)`` and says so rather than being
    passed off as the whole of it.
    """

    def __init__(self, efficient, *, complete, nodes, cuts, tests,
                 seconds, stopped=None):
        self.efficient = efficient
        self.complete = complete
        self.nodes = nodes
        self.cuts = cuts
        self.tests = tests
        self.seconds = seconds
        self.stopped = stopped

    def __len__(self) -> int:
        return len(self.efficient)

    def __iter__(self):
        return iter(self.efficient)

    def __str__(self) -> str:
        head = f"{len(self.efficient)} efficient points"
        if not self.complete:
            head += f" (SUBSET -- {self.stopped})"
        return (f"{head}; {self.nodes} nodes, {self.cuts} cuts, "
                f"{self.tests} efficiency tests, {self.seconds:.2f}s")


def _fractional_coordinate(x: Sequence[Fraction], integral) -> Optional[int]:
    """The first integer-constrained coordinate that is not an integer."""
    for j, v in enumerate(x):
        if integral[j] and v.denominator != 1:
            return j
    return None


def enumerate_by_cuts(problem: MOILFP, *, objective=None,
                      max_nodes: int = 200_000,
                      time_budget: Optional[float] = None) -> CutEnumeration:
    """All of ``E(P_D)``, in decision space, with the efficient cut.

    See the module docstring for what this is a reimplementation of and what it
    is not.  *objective* drives each node's relaxation and changes only the
    order points are met in; it defaults to ``Z_1``.
    """
    from .milp import solve_relaxation

    if problem.p < 1:
        raise ValueError("a MOILFP needs at least one criterion")
    objective = objective if objective is not None else problem.criteria[0]

    started = perf_counter()
    deadline = (started + time_budget) if time_budget else None
    integral = list(problem.model.integrality)

    stack = [problem.model.copy()]
    found: List[tuple] = []
    seen = set()
    nodes = cuts = tests = 0
    stopped = None

    while stack:
        if deadline and perf_counter() >= deadline:
            stopped = "time budget"
            break
        if nodes >= max_nodes:
            stopped = f"node limit ({max_nodes})"
            break

        region = stack.pop()
        result = solve_relaxation(region, objective)
        nodes += 1
        if result.status != "optimal":
            continue                       # infeasible or unbounded: fathom

        x = list(result.x[:problem.n])
        j = _fractional_coordinate(x, integral)
        if j is not None:
            v = x[j]
            lo = Fraction(v.numerator // v.denominator)        # floor
            row = [ONE if t == j else ZERO for t in range(problem.n)]
            stack.append(region.copy().add(row, LE, lo))
            stack.append(region.copy().add(row, GE, lo + 1))
            continue

        key = tuple(x)
        if key not in seen:
            seen.add(key)
            # Efficiency is a property of the PROBLEM, so the test runs on the
            # original feasible set and never on the truncated node region.
            tests += 1
            if test_efficiency(problem, x).psi == 0:
                found.append(key)

        columns = improving_columns(problem, result.tableau)
        if not columns:
            continue                       # H_l empty: x is ideal here, fathom
        coeffs, rhs = _cut_row(region, columns)
        cuts += 1
        stack.append(region.copy().add(coeffs, GE, rhs))

    return CutEnumeration(found, complete=stopped is None, nodes=nodes,
                          cuts=cuts, tests=tests,
                          seconds=perf_counter() - started, stopped=stopped)
