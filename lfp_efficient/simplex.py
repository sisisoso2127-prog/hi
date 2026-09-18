"""A small exact (rational) simplex, in tableau form.

Why not SciPy / PuLP / CBC?
--------------------------
The algorithm of the paper does not only need *an optimal solution*: several of
its steps are expressed directly in the language of the simplex tableau.

  * ``B_k``      the basis associated with the current solution ``x_k``,
  * ``y_{k,j} = (B_k)^-1 a_{k,j}``  the updated column of a non-basic variable,
  * ``gamma_{k,j} = Z_{k,2} (p_j - p_{k,j}) - Z_{k,1} (q_j - q_{k,j})``
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
"""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Sequence

from .rational import F, ZERO, dot

OPTIMAL = "optimal"
INFEASIBLE = "infeasible"
UNBOUNDED = "unbounded"


# --------------------------------------------------------------------------
# Pricing rules: they turn a tableau into a vector of "reduced costs".
# A column j is attractive (for a maximisation) when its price is > 0.
# --------------------------------------------------------------------------
class LinearPricing:
    """Reduced costs of ``max c'x``:  ``r_j = c_j - sum_{i in I} c_{B(i)} y_ij``."""

    def __init__(self, c: Sequence[Fraction]):
        self.c = list(c)

    def prices(self, tab: "Tableau") -> List[Fraction]:
        cb = [self.c[j] for j in tab.basis]
        out = []
        for j in range(tab.n):
            zj = sum((cb[i] * tab.T[i][j] for i in range(tab.m)), ZERO)
            out.append(self.c[j] - zj)
        return out

    def value(self, x: Sequence[Fraction]) -> Fraction:
        return dot(self.c, x)


class FractionalPricing:
    """Reduced gradient of ``max (U'x + alpha) / (V'x + beta)``.

    With ``Z_1 = U'x + alpha`` and ``Z_2 = V'x + beta`` at the current vertex,
    and ``p_{k,j} = sum_{i in I_k} U_{B(i)} y_{k,ij}`` (same for ``q`` with V),
    the paper's reduced gradient reads

        gamma_j = Z_2 * (U_j - p_{k,j}) - Z_1 * (V_j - q_{k,j}).

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

    def value(self, x: Sequence[Fraction]) -> Fraction:
        den = dot(self.V, x) + self.beta
        if den == 0:
            raise ZeroDivisionError("denominator V'x + beta vanishes at x")
        return (dot(self.U, x) + self.alpha) / den

    def prices(self, tab: "Tableau") -> List[Fraction]:
        x = tab.solution()
        z1 = dot(self.U, x) + self.alpha
        z2 = dot(self.V, x) + self.beta
        if z2 <= 0:
            raise ValueError(
                "the denominator V'x + beta must stay > 0 on the feasible set; "
                f"got {z2} -- check the data of the fractional objective")
        ub = [self.U[j] for j in tab.basis]
        vb = [self.V[j] for j in tab.basis]
        out = []
        for j in range(tab.n):
            pkj = sum((ub[i] * tab.T[i][j] for i in range(tab.m)), ZERO)
            qkj = sum((vb[i] * tab.T[i][j] for i in range(tab.m)), ZERO)
            out.append(z2 * (self.U[j] - pkj) - z1 * (self.V[j] - qkj))
        return out


# --------------------------------------------------------------------------
# The tableau itself
# --------------------------------------------------------------------------
@dataclass
class Tableau:
    """Carries ``T = B^-1 A`` (m x n), ``xb = B^-1 b`` (m) and the basis."""

    T: List[List[Fraction]]
    xb: List[Fraction]
    basis: List[int]

    @property
    def m(self) -> int:
        return len(self.T)

    @property
    def n(self) -> int:
        return len(self.T[0]) if self.T else 0

    def clone(self) -> "Tableau":
        return Tableau([row[:] for row in self.T], self.xb[:], self.basis[:])

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

    def pivot(self, r: int, c: int) -> None:
        """Classical pivot: variable ``c`` enters, the basic variable of row ``r`` leaves."""
        piv = self.T[r][c]
        if piv == 0:
            raise ZeroDivisionError("pivot on a zero element")
        self.T[r] = [v / piv for v in self.T[r]]
        self.xb[r] = self.xb[r] / piv
        for i in range(self.m):
            if i == r:
                continue
            factor = self.T[i][c]
            if factor == 0:
                continue
            self.T[i] = [a - factor * b for a, b in zip(self.T[i], self.T[r])]
            self.xb[i] = self.xb[i] - factor * self.xb[r]
        self.basis[r] = c

    # ---- pivoting rules -------------------------------------------------
    def entering(self, prices: Sequence[Fraction]) -> Optional[int]:
        """Bland's rule: the *smallest index* with a strictly positive price.

        Bland's rule is slower than Dantzig's but guarantees termination even
        on degenerate vertices -- and degeneracy is the rule, not the
        exception, once Sylva-Crema cuts pile up on the feasible region.
        """
        inb = set(self.basis)
        for j in range(self.n):
            if j not in inb and prices[j] > 0:
                return j
        return None

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

    def run(self, pricing, max_iter: int = 100_000) -> str:
        """Pivot until optimality (all prices <= 0) or an unbounded ray shows up."""
        for _ in range(max_iter):
            prices = pricing.prices(self)
            c = self.entering(prices)
            if c is None:
                return OPTIMAL
            r = self.ratio_test(c)
            if r is None:
                return UNBOUNDED
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

    # ---- phase I ---------------------------------------------------------
    A1 = [A[i] + [ONE_IF(i, k, m) for k in range(m)] for i in range(m)]
    tab = Tableau(A1, b[:], [n + i for i in range(m)])
    phase1_cost = [ZERO] * n + [F(-1)] * m           # max -sum(artificials)
    status = tab.run(LinearPricing(phase1_cost))
    if status != OPTIMAL:
        return SimplexResult(INFEASIBLE)
    if sum((tab.xb[i] for i in range(m) if tab.basis[i] >= n), ZERO) != 0:
        return SimplexResult(INFEASIBLE)

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


def ONE_IF(i: int, k: int, m: int) -> Fraction:
    """Identity matrix entry, kept as a helper to build the phase-I columns."""
    return Fraction(1) if i == k else ZERO
