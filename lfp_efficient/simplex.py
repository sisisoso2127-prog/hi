"""A small exact (rational) simplex, in tableau form.

Why not SciPy / PuLP / CBC?
--------------------------
The algorithm of the paper does not only need *an optimal solution*: several of
its steps are expressed directly in the language of the simplex tableau.

  * ``B_k``      the basis associated with the current solution ``x_k``,
  * ``y_{k,j} = (B_k)^-1 a_{k,j}``  the updated column of a non-basic variable,
  * ``gamma_{k,j} = Z_{k,2} (U_j - p_{k,j}) - Z_{k,1} (V_j - q_{k,j})``
    the reduced gradient of the *linear fractional* objective
    (Cambini & Martein, Theorem 3 of the paper),
  * ``Gamma_l = { j in N_l : gamma_j = 0 }`` the edges carrying the alternative
    optima that the algorithm walks along (Definition 2).

A black-box MILP solver returns none of that.  So the package carries its own
tableau, which additionally keeps everything exact.

Standard form used everywhere here::

    max  f(x)      s.t.   A x = b ,  x >= 0 ,  b >= 0

``f`` is either linear (``LinearPricing``) or linear fractional
(``FractionalPricing``); the only difference between the two is how the
reduced costs are priced, the pivoting machinery is shared.

Performance note
----------------
Reduced costs are *maintained* through the pivots rather than recomputed from
``c_B' B^-1 A`` at every iteration: the tableau carries one cost row per cost
vector (one for a linear objective, two -- ``U`` and ``V`` -- for a fractional
one) and each pivot updates them in ``O(n)`` instead of ``O(mn)``.  With exact
rational arithmetic that difference dominates the whole run time.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Sequence

from .rational import F, ZERO, dot

OPTIMAL = "optimal"
INFEASIBLE = "infeasible"
UNBOUNDED = "unbounded"


# --------------------------------------------------------------------------
# Pricing rules: they turn the maintained cost rows into "reduced costs".
# A column j is attractive (for a maximisation) when its price is > 0.
# --------------------------------------------------------------------------
class LinearPricing:
    """Reduced costs of ``max c'x``:  ``r_j = c_j - sum_{i in I} c_{B(i)} y_ij``."""

    def __init__(self, c: Sequence[Fraction]):
        self.c = list(c)

    def cost_vectors(self) -> List[List[Fraction]]:
        return [self.c]

    def prices(self, tab: "Tableau") -> List[Fraction]:
        tab.ensure_costs(self)
        return tab.cost_rows[0]

    def value(self, x: Sequence[Fraction]) -> Fraction:
        return dot(self.c, x)


class FractionalPricing:
    """Reduced gradient of ``max (U'x + alpha) / (V'x + beta)``.

    With ``Z_1 = U'x + alpha``, ``Z_2 = V'x + beta`` at the current vertex and
    ``rU``, ``rV`` the maintained reduced-cost rows of ``U`` and ``V``, the
    paper's reduced gradient is simply

        gamma_j = Z_2 * rU_j - Z_1 * rV_j .

    Moving along the edge of a non-basic ``j`` increases the ratio iff
    ``gamma_j > 0``; ``gamma_j = 0`` means the edge keeps the objective
    constant, i.e. it carries *alternative optima* -- exactly the set
    ``Gamma_l`` the algorithm explores.  The denominator must stay positive on
    the feasible region (standard assumption of linear fractional programming).
    """

    def __init__(self, U: Sequence[Fraction], alpha: Fraction,
                 V: Sequence[Fraction], beta: Fraction):
        self.U, self.alpha = list(U), F(alpha)
        self.V, self.beta = list(V), F(beta)

    def cost_vectors(self) -> List[List[Fraction]]:
        return [self.U, self.V]

    def value(self, x: Sequence[Fraction]) -> Fraction:
        den = dot(self.V, x) + self.beta
        if den == 0:
            raise ZeroDivisionError("denominator V'x + beta vanishes at x")
        return (dot(self.U, x) + self.alpha) / den

    def prices(self, tab: "Tableau") -> List[Fraction]:
        tab.ensure_costs(self)
        z1, z2 = self.alpha, self.beta
        for i, j in enumerate(tab.basis):            # only basic variables are non-zero
            xb = tab.xb[i]
            if xb:
                z1 += self.U[j] * xb
                z2 += self.V[j] * xb
        if z2 <= 0:
            raise ValueError(
                "the denominator V'x + beta must stay > 0 on the feasible set; "
                f"got {z2} -- check the data of the fractional objective")
        rU, rV = tab.cost_rows
        return [z2 * rU[j] - z1 * rV[j] for j in range(tab.n)]


# --------------------------------------------------------------------------
# The tableau itself
# --------------------------------------------------------------------------
@dataclass
class Tableau:
    """Carries ``T = B^-1 A`` (m x n), ``xb = B^-1 b`` (m) and the basis."""

    T: List[List[Fraction]]
    xb: List[Fraction]
    basis: List[int]
    #: reduced-cost rows ``c - c_B' B^-1 A``, maintained through the pivots
    cost_rows: List[List[Fraction]] = field(default_factory=list)
    _cost_owner: object = None

    @property
    def m(self) -> int:
        return len(self.T)

    @property
    def n(self) -> int:
        return len(self.T[0]) if self.T else 0

    def clone(self) -> "Tableau":
        return Tableau([row[:] for row in self.T], self.xb[:], self.basis[:],
                       [row[:] for row in self.cost_rows], self._cost_owner)

    def solution(self) -> List[Fraction]:
        """The basic solution: non-basic variables are 0, basic ones read off ``xb``."""
        x = [ZERO] * self.n
        for i, j in enumerate(self.basis):
            x[j] = self.xb[i]
        return x

    def nonbasic(self) -> List[int]:
        """Indices ``N_k`` of the non-basic columns."""
        inb = set(self.basis)
        return [j for j in range(self.n) if j not in inb]

    def column(self, j: int) -> List[Fraction]:
        """``y_{k,j} = (B_k)^-1 a_{k,j}``, the updated column of variable j."""
        return [self.T[i][j] for i in range(self.m)]

    # ---- cost rows ------------------------------------------------------
    def ensure_costs(self, pricing) -> None:
        """(Re)build the cost rows for *pricing* -- done once, then maintained."""
        if self._cost_owner is pricing:
            return
        self.cost_rows = []
        for c in pricing.cost_vectors():
            cb = [c[j] for j in self.basis]
            row = c[:]
            for i in range(self.m):
                if cb[i]:
                    Ti, f = self.T[i], cb[i]
                    row = [a - f * b for a, b in zip(row, Ti)]
            self.cost_rows.append(row)
        self._cost_owner = pricing

    def pivot(self, r: int, c: int) -> None:
        """Classical pivot: variable ``c`` enters, the basic variable of row ``r`` leaves."""
        piv = self.T[r][c]
        if piv == 0:
            raise ZeroDivisionError("pivot on a zero element")
        if piv != 1:
            self.T[r] = [v / piv for v in self.T[r]]
            self.xb[r] = self.xb[r] / piv
        pivot_row = self.T[r]
        # constraint matrices are sparse and so are their tableaux: touching
        # only the non-zero entries of the pivot row cuts the number of exact
        # rational operations by a large factor.
        nz = [(k, v) for k, v in enumerate(pivot_row) if v]
        xbr = self.xb[r]
        for i in range(self.m):
            if i == r:
                continue
            factor = self.T[i][c]
            if factor == 0:
                continue
            Ti = self.T[i]
            for k, v in nz:
                Ti[k] -= factor * v
            self.xb[i] = self.xb[i] - factor * xbr
        for row in self.cost_rows:                        # keep the prices in sync
            factor = row[c]
            if factor:
                for k, v in nz:
                    row[k] -= factor * v
        self.basis[r] = c

    # ---- pivoting rules -------------------------------------------------
    def entering(self, prices: Sequence[Fraction], bland: bool) -> Optional[int]:
        """Choose the entering column.

        Dantzig's rule (steepest price) is used by default because it needs far
        fewer pivots; the caller falls back to **Bland's rule** (smallest index
        with a positive price) as soon as degenerate pivots pile up, which
        restores the anti-cycling guarantee.  Degeneracy is the rule rather than
        the exception here, once Sylva-Crema cuts accumulate on the region.
        """
        inb = set(self.basis)
        best, best_price = None, None
        for j in range(self.n):
            if j in inb or prices[j] <= 0:
                continue
            if bland:
                return j
            if best_price is None or prices[j] > best_price:
                best, best_price = j, prices[j]
        return best

    def ratio_test(self, c: int) -> Optional[int]:
        """Leaving row for entering column ``c``; ``None`` means unbounded ray."""
        best_row, best_ratio = None, None
        for i in range(self.m):
            if self.T[i][c] > 0:
                ratio = self.xb[i] / self.T[i][c]
                if (best_ratio is None or ratio < best_ratio
                        or (ratio == best_ratio and self.basis[i] < self.basis[best_row])):
                    best_row, best_ratio = i, ratio
        return best_row

    def run(self, pricing, max_iter: int = 200_000) -> str:
        """Pivot until optimality (all prices <= 0) or an unbounded ray shows up."""
        self.ensure_costs(pricing)
        degenerate_streak = 0
        bland = False
        for _ in range(max_iter):
            prices = pricing.prices(self)
            c = self.entering(prices, bland)
            if c is None:
                return OPTIMAL
            r = self.ratio_test(c)
            if r is None:
                return UNBOUNDED
            if self.xb[r] == 0:
                degenerate_streak += 1
                if degenerate_streak > self.m + 5:
                    bland = True          # anti-cycling from here on
            else:
                degenerate_streak = 0
            self.pivot(r, c)
        raise RuntimeError("simplex did not converge (possible cycling)")


STALLED = "stalled"


@dataclass
class SimplexResult:
    status: str
    x: List[Fraction] = field(default_factory=list)
    objective: Optional[Fraction] = None
    tableau: Optional[Tableau] = None
    #: original column indices kept after the redundant rows/artificials clean-up
    n_structural: int = 0
    #: the pricing object owning ``tableau.cost_rows``; carrying it around lets
    #: a child node reuse those rows instead of rebuilding them
    pricing: object = None


def solve_standard_form(A, b, pricing, n_structural: Optional[int] = None) -> SimplexResult:
    """Two-phase simplex for ``max f(x) s.t. Ax = b, x >= 0``.

    Phase I builds a feasible basis with artificial variables (minimising their
    sum); phase II optimises the real objective with the given *pricing* rule,
    so the very same code drives the linear and the linear fractional case.
    """
    A = [row[:] for row in A]
    b = b[:]
    m = len(A)
    n = len(A[0]) if m else 0
    if n_structural is None:
        n_structural = n

    # Phase I needs b >= 0.
    for i in range(m):
        if b[i] < 0:
            A[i] = [-v for v in A[i]]
            b[i] = -b[i]

    # ---- crash basis -----------------------------------------------------
    # A column that is the unit vector e_i (a "<=" slack whose row was not
    # sign-flipped) is already a feasible basic column for row i, because
    # b >= 0 at this point.  Rows covered this way need no artificial variable
    # at all -- and when every row is covered, phase I is skipped outright.
    # Most of the linear programs solved by the algorithm are of that shape.
    unit_of_row = _unit_columns(A, m, n)
    basis: List[Optional[int]] = [unit_of_row.get(i) for i in range(m)]
    need = [i for i in range(m) if basis[i] is None]

    # ---- phase I ---------------------------------------------------------
    if need:
        cols = n + len(need)
        rows = []
        for i in range(m):
            extra = [ZERO] * len(need)
            if i in need:
                extra[need.index(i)] = ONE
            rows.append(A[i] + extra)
        full_basis = [basis[i] if basis[i] is not None else n + need.index(i)
                      for i in range(m)]
        tab = Tableau(rows, b[:], full_basis)
        phase1_cost = [ZERO] * n + [F(-1)] * len(need)   # max -sum(artificials)
        status = tab.run(LinearPricing(phase1_cost))
        if status != OPTIMAL:
            return SimplexResult(INFEASIBLE)
        if any(tab.xb[i] != 0 for i in range(tab.m) if tab.basis[i] >= n):
            return SimplexResult(INFEASIBLE)
    else:
        tab = Tableau([row[:] for row in A], b[:], [j for j in basis])

    # Drive the remaining artificials out of the basis; rows that cannot be
    # pivoted are redundant constraints and get dropped.
    rows_to_drop = []
    for i in range(tab.m):
        if tab.basis[i] >= n:
            pivot_col = next((j for j in range(n) if tab.T[i][j] != 0), None)
            if pivot_col is None:
                rows_to_drop.append(i)
            else:
                tab.pivot(i, pivot_col)
    keep = [i for i in range(tab.m) if i not in rows_to_drop]
    tab = Tableau([tab.T[i][:n] for i in keep], [tab.xb[i] for i in keep],
                  [tab.basis[i] for i in keep])

    # ---- phase II --------------------------------------------------------
    status = tab.run(pricing)
    if status == UNBOUNDED:
        return SimplexResult(UNBOUNDED, tableau=tab, n_structural=n_structural,
                             pricing=pricing)
    x = tab.solution()
    return SimplexResult(OPTIMAL, x, pricing.value(x), tab, n_structural, pricing)


def _unit_columns(A, m: int, n: int) -> dict:
    """Map ``row -> column`` for columns that are already a unit vector ``e_i``.

    Slack and surplus columns appear in a single row, so the test is cheap and
    catches exactly the "<=" rows that survived the ``b >= 0`` normalisation.
    """
    found = {}
    taken = set()
    for j in range(n - 1, -1, -1):        # slacks live at the end of the matrix
        row_of_one = None
        ok = True
        for i in range(m):
            v = A[i][j]
            if v == 0:
                continue
            if v == 1 and row_of_one is None:
                row_of_one = i
            else:
                ok = False
                break
        if ok and row_of_one is not None and row_of_one not in found and j not in taken:
            found[row_of_one] = j
            taken.add(j)
    return found


ONE = Fraction(1)


# --------------------------------------------------------------------------
# Warm start: re-optimising a tableau after one extra bound row
# --------------------------------------------------------------------------
# Branch & bound differs from its parent by a single bound row, yet solving
# each node from scratch pays a full phase I to rebuild a basis the parent
# already had.  Measured on a hard instance that was 20.6 phase-I pivots per
# node against 4.1 phase-II ones: five sixths of the work thrown away.
#
# The parent basis stays a basis of the child once the new row's slack joins
# it; only the new row can be primal infeasible.  A dual-simplex-style
# restoration fixes that in a pivot or two, after which the ordinary primal
# loop finishes the job.

def extend_pricing(pricing, extra: int):
    """The same objective seen in a space with *extra* new (zero-cost) columns."""
    zeros = [ZERO] * extra
    if isinstance(pricing, LinearPricing):
        return LinearPricing(pricing.c + zeros)
    return FractionalPricing(pricing.U + zeros, pricing.alpha,
                             pricing.V + zeros, pricing.beta)


def add_bound_row(tab: Tableau, pricing, j: int, bound: Fraction, upper: bool):
    """Append ``x_j <= bound`` (or ``x_j >= bound``) to *tab*, keeping the basis.

    The row is written with its own slack, then the basic variables occurring
    in it are eliminated so that the slack can be made basic.  Its value comes
    out as ``bound - x_j`` (resp. ``x_j - bound``): negative exactly when the
    branch cuts off the parent's solution, which is the only reason the child
    is not immediately feasible.  Returns the extended pricing object.
    """
    for row in tab.T:
        row.append(ZERO)
    for row in tab.cost_rows:
        row.append(ZERO)          # a basic, zero-cost variable prices at 0
    new_pricing = extend_pricing(pricing, 1)
    tab._cost_owner = new_pricing
    c_new = tab.n - 1

    row = [ZERO] * tab.n
    row[j] = ONE if upper else -ONE
    row[c_new] = ONE
    rhs = F(bound) if upper else -F(bound)

    for i, basic in enumerate(tab.basis):
        if basic == j and row[j] != 0:
            factor = row[j]
            row = [a - factor * t for a, t in zip(row, tab.T[i])]
            rhs = rhs - factor * tab.xb[i]
            break

    tab.T.append(row)
    tab.xb.append(rhs)
    tab.basis.append(c_new)
    return new_pricing


def add_linear_row(tab: Tableau, pricing, coeffs: Sequence[Fraction],
                   rhs: Fraction):
    """Append ``coeffs . x <= rhs`` to *tab*, keeping the basis.

    The general form of :func:`add_bound_row`, which is the special case of a
    row touching a single structural variable.  A box of the criterion-space
    search differs from its parent by rows of this shape, so the parent's
    basis can be carried into the child exactly as a branch & bound node
    carries its parent's.

    The new row is written with its own slack, then **every** basic variable
    occurring in it is eliminated so that the slack can be made basic.  One
    pass suffices: the basic variable of row ``i`` has a unit column, so it is
    zero in every other row, and eliminating one never reintroduces another.

    The slack's value comes out as ``rhs - coeffs . x`` at the parent's vertex:
    negative exactly when the new row cuts that vertex off, which is the only
    reason the child is not immediately feasible.  Returns the extended
    pricing object.
    """
    for row in tab.T:
        row.append(ZERO)
    for row in tab.cost_rows:
        row.append(ZERO)          # a basic, zero-cost variable prices at 0
    new_pricing = extend_pricing(pricing, 1)
    tab._cost_owner = new_pricing
    c_new = tab.n - 1

    row = [ZERO] * tab.n
    for j, v in enumerate(coeffs):
        if v:
            row[j] = F(v)
    row[c_new] = ONE
    value = F(rhs)

    for i, basic in enumerate(tab.basis):
        factor = row[basic]
        if factor:
            row = [a - factor * t for a, t in zip(row, tab.T[i])]
            value = value - factor * tab.xb[i]

    tab.T.append(row)
    tab.xb.append(value)
    tab.basis.append(c_new)
    return new_pricing


def _reference_prices(tab: Tableau, pricing, z1: Fraction, z2: Fraction) -> List[Fraction]:
    """Prices used to steer the restoration, frozen at the parent's vertex.

    During the restoration the point is primal infeasible, so the fractional
    objective's own ``Z_1``/``Z_2`` are meaningless there (``Z_2`` may even be
    negative, where ``Phi`` is not defined).  Freezing them at the parent's --
    feasible -- vertex keeps the ratio test well defined.  It only steers the
    choice of the entering column: optimality is established afterwards by the
    ordinary primal loop, so no correctness rests on it.
    """
    if len(tab.cost_rows) == 1:
        return tab.cost_rows[0]
    rU, rV = tab.cost_rows
    return [z2 * rU[k] - z1 * rV[k] for k in range(tab.n)]


def restore_feasibility(tab: Tableau, pricing, z1: Fraction, z2: Fraction,
                        max_iter: int = 200) -> str:
    """Dual-simplex pivots until ``xb >= 0``.

    Returns ``OPTIMAL`` once the basis is primal feasible, ``INFEASIBLE`` when a
    row proves the child empty, or ``STALLED`` when the iteration cap is hit --
    in which case the caller falls back to a solve from scratch, so a
    restoration that fails costs time and never correctness.

    The infeasibility certificate needs no assumption on the prices: a row
    ``x_B(r) + sum_j y_rj x_j = xb_r`` with ``xb_r < 0`` and every non-basic
    ``y_rj >= 0`` cannot be satisfied by any ``x >= 0``.
    """
    for _ in range(max_iter):
        r, worst = None, ZERO
        for i in range(tab.m):
            if tab.xb[i] < worst:
                r, worst = i, tab.xb[i]
        if r is None:
            return OPTIMAL

        prices = _reference_prices(tab, pricing, z1, z2)
        inb = set(tab.basis)
        best, best_ratio = None, None
        row = tab.T[r]
        for k in range(tab.n):
            if k in inb or row[k] >= 0:
                continue
            ratio = prices[k] / row[k]            # prices <= 0 and row < 0
            if best_ratio is None or ratio < best_ratio:
                best, best_ratio = k, ratio
        if best is None:
            return INFEASIBLE
        tab.pivot(r, best)
    return STALLED


# --------------------------------------------------------------------------
# A basis at a *given* point, built from scratch
# --------------------------------------------------------------------------
# The edge walk of Definition 2 reads the tableau at the optimum ``x_l`` of the
# truncated region.  The tableau branch & bound leaves behind is not that one:
# it belongs to the node that produced the incumbent, and carries that node's
# branching rows.  Those rows pin variables at their bounds, so their slacks
# sit basic at zero and every ratio test collapses -- measured, 93% of the
# zero-gradient edges had ``theta0 = 0`` and the whole step was inert.
#
# The cure is to forget the search tree and rebuild a basis of the region
# itself at ``x_l``.  ``x_l`` is an integer optimum, so it is a feasible point
# of the region; when it is also a *vertex* of it -- which is what "there is a
# basis" means -- the construction below produces one.

def tableau_at(A, b, values) -> Optional["Tableau"]:
    """A tableau of ``{Ax = b, x >= 0}`` whose basic solution is *values*.

    Returns ``None`` when *values* is not a basic feasible solution: either it
    fails the equations, or it has more than ``m`` positive components, or the
    columns carrying them are linearly dependent.  In all three cases the point
    lies strictly inside a face rather than at a vertex, and the edges of
    Definition 2 -- which emanate from a vertex -- are simply not defined there.

    Variables that are already positive *must* be basic, since a non-basic one
    is zero by definition; the basis is completed with whatever further columns
    keep it non-singular, which makes it degenerate but legitimate.  Slack
    columns are preferred for that completion, so that the structural variables
    stay non-basic and their edges are available to the walk.
    """
    m = len(A)
    if m == 0:
        return None
    n = len(A[0])
    for i in range(m):
        if dot(A[i], values) != b[i]:
            return None
    positive = [j for j in range(n) if values[j] != 0]
    if len(positive) > m or any(values[j] < 0 for j in range(n)):
        return None

    work = [A[i][:] + [b[i]] for i in range(m)]
    basis_of_row: List[Optional[int]] = [None] * m
    free_rows = list(range(m))
    # the positive variables first -- they have no choice -- then the slacks
    # (highest indices) and finally the structural columns
    order = positive + [j for j in range(n - 1, -1, -1) if values[j] == 0]

    for c in order:
        if not free_rows:
            break
        r = next((i for i in free_rows if work[i][c] != 0), None)
        if r is None:
            if values[c] != 0:
                return None           # a positive variable cannot be made basic
            continue
        piv = work[r][c]
        if piv != 1:
            work[r] = [v / piv for v in work[r]]
        nz = [(k, v) for k, v in enumerate(work[r]) if v]
        for i in range(m):
            if i == r:
                continue
            factor = work[i][c]
            if factor == 0:
                continue
            row = work[i]
            for k, v in nz:
                row[k] -= factor * v
        basis_of_row[r] = c
        free_rows.remove(r)

    for i in free_rows:                # rows the elimination left empty
        if work[i][n] != 0:
            return None                # 0 = nonzero: the system is inconsistent
    keep = [i for i in range(m) if basis_of_row[i] is not None]
    return Tableau([work[i][:n] for i in keep], [work[i][n] for i in keep],
                   [basis_of_row[i] for i in keep])
