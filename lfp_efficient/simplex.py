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


@dataclass
class SimplexResult:
    status: str
    x: List[Fraction] = field(default_factory=list)
    objective: Optional[Fraction] = None
    tableau: Optional[Tableau] = None
    #: original column indices kept after the redundant rows/artificials clean-up
    n_structural: int = 0


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
        return SimplexResult(UNBOUNDED, tableau=tab, n_structural=n_structural)
    x = tab.solution()
    return SimplexResult(OPTIMAL, x, pricing.value(x), tab, n_structural)


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
