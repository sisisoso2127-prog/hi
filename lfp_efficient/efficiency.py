"""Efficiency test, lower bounds and Sylva-Crema truncation of the feasible set."""

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Sequence

from .milp import MilpResult, solve_fractional_milp, solve_linear_milp, solve_milp
from .model import EQ, GE, LE, MOILP, Model
from .rational import F, ZERO, dot
from .simplex import OPTIMAL


@dataclass
class EfficiencyTest:
    """Result of the test of Theorem 1 applied to a point ``x*``."""

    efficient: bool
    #: optimal value ``psi*`` of the test; ``x*`` is efficient iff it is 0
    psi: Fraction
    #: when ``x*`` is not efficient, an efficient point dominating it
    witness: Optional[List[Fraction]] = None


def test_efficiency(problem: MOILP, x_star: Sequence[Fraction],
                    region: Optional[Model] = None) -> EfficiencyTest:
    """Isermann's test, Theorem 1 of the paper::

        (P_x*)   max  psi(x) = sum_i Psi_i
                 s.t. C x - I Psi = C x*
                      x in D,  Psi_i >= 0

    ``x*`` is efficient for ``(P_D)`` iff the optimal value is 0.  Otherwise the
    optimal ``x`` of the test is itself an efficient point dominating ``x*``
    (Ecker & Kouada), which is how the algorithm manufactures its first
    efficient solution out of an arbitrary one.

    The test is always run on the *original* region ``D``: efficiency is a
    property of ``(P_D)``, never of a truncated region.  ``region`` is only
    exposed so that callers can test over a sub-region on purpose.
    """
    base = (region or problem.model).copy()
    n = base.n
    psi_idx = base.add_variables(problem.p, integer=False)      # Psi_i >= 0, continuous

    cx_star = problem.C(x_star)
    for i, row in enumerate(problem.criteria):
        coeffs = list(row) + [ZERO] * (base.n - len(row))
        coeffs[psi_idx[i]] = F(-1)                              # C_i x - Psi_i = C_i x*
        base.add(coeffs, EQ, cx_star[i])

    c = [ZERO] * base.n
    for j in psi_idx:
        c[j] = F(1)
    res = solve_milp(base, c)

    if res.status != OPTIMAL:
        # x* itself is feasible for the test (with Psi = 0), so this cannot
        # happen unless x* is outside D.
        raise ValueError("the efficiency test is infeasible: is x* in D?")

    if res.objective == 0:
        return EfficiencyTest(True, res.objective, None)
    return EfficiencyTest(False, res.objective, res.x[:n])


def lower_bounds(problem: MOILP) -> List[Fraction]:
    """``M_i = min { C_i x : x in D }`` -- the "big-M" of the Sylva-Crema cuts.

    In the paper's example this returns ``M_1 = M_2 = -3``.
    """
    bounds = []
    for row in problem.criteria:
        res = solve_linear_milp(problem.model, row, minimize=True)
        if res.status != OPTIMAL:
            raise ValueError("D must be non-empty and bounded to compute the lower bounds")
        bounds.append(res.objective)
    return bounds


def add_sylva_crema_cut(region: Model, problem: MOILP, x_hat: Sequence[Fraction],
                        M: Sequence[Fraction]) -> Model:
    """Remove from *region* every point dominated by (or equal to) ``C x_hat``.

    This is the constraint block of equation (5) of the paper: for the stored
    efficient point ``x^s`` one introduces binaries ``y^s_i`` and imposes

        C_i x  >=  (C_i x^s + 1) y^s_i  +  M_i (1 - y^s_i),    i = 1..p
        sum_i y^s_i >= 1 .

    When ``y^s_i = 0`` the row degenerates into the always-true ``C_i x >= M_i``;
    when ``y^s_i = 1`` a *strict* improvement of the i-th criterion is forced.
    Since at least one binary must be 1, every remaining point improves at least
    one criterion with respect to ``C x^s``, i.e.

        D_new = D_old \\ { x : C x <= C x^s } .

    The ``+1`` is what makes the cut valid for *integer* criteria, and it is
    also what makes the truncation strict: ``x^s`` itself is cut away, which is
    why the best point sharing its criterion vector must have been harvested
    beforehand (sub-problem ``Q``).
    """
    region = region.copy()
    y = region.add_variables(problem.p, integer=True)
    cx = problem.C(x_hat)
    for i, row in enumerate(problem.criteria):
        # -C_i x + (C_i x^s + 1 - M_i) y_i <= -M_i
        #
        # This is the paper's row written with the opposite sign.  The "<="
        # form matters for speed, not for the mathematics: its slack column is
        # a +1 unit vector whenever -M_i >= 0, so the crash basis of the
        # simplex covers the row and no phase-I artificial is needed for it.
        # With p criteria and one cut per iteration that removes p artificial
        # variables per iteration from every single node of the branch & bound.
        coeffs = [-v for v in row] + [ZERO] * (region.n - len(row))
        coeffs[y[i]] = cx[i] + F(1) - M[i]
        region.add(coeffs, LE, -M[i])
    # sum_i y_i >= 1
    #
    # The explicit "y_i <= 1" rows of equation (5) are redundant and are left
    # out.  Writing K_i = C_i x^s + 1 - M_i >= 1, a point is feasible for the
    # block iff some y_i >= 1, and then C_i x >= K_i y_i + M_i >= K_i + M_i =
    # C_i x^s + 1.  Values y_i >= 2 only impose a *stronger* requirement, so
    # the union over the integer y is unchanged -- while the region loses p
    # rows per iteration.  The relaxation is not weakened either: the linear
    # program satisfies "sum y_i >= 1" as cheaply as it can and never raises a
    # y_i above 1 on its own.
    pick = [ZERO] * region.n
    for j in y:
        pick[j] = F(1)
    region.add(pick, GE, F(1))
    return region


def best_with_same_criterion(region: Model, problem: MOILP,
                             x_tilde: Sequence[Fraction], phi,
                             cutoff=None,
                             denominator_positive: bool = False) -> MilpResult:
    """Sub-problem ``Q(x~) = max { Phi(x) : x in region, C x = C x~ }``.

    All the points of this set share the non-dominated criterion vector
    ``C x~``, hence they are *all* efficient as soon as ``x~`` is.  The
    Sylva-Crema cut is about to delete that whole slice of the region, so the
    best value of ``Phi`` on it has to be collected first -- this is the step
    the paper writes as ``solve Q(x~_l)``.
    """
    # The slice { x : C x = C x~ } is never touched by the cuts already made,
    # so it is the same set in the truncated region and in D itself:
    # a previous cut would exclude it only if C x~ <= C x^s for some stored
    # efficient x^s, which forces C x~ = C x^s (both are non-dominated) and
    # would mean x~ had already been cut away -- impossible, since x~ dominates
    # a point of the current region.  Solving over D keeps the p binaries and
    # the 2p+1 rows of every accumulated cut out of this sub-problem.
    sub = problem.model.copy()
    cx = problem.C(x_tilde)
    for i, row in enumerate(problem.criteria):
        coeffs = list(row) + [ZERO] * (sub.n - len(row))
        sub.add(coeffs, EQ, cx[i])
    res = solve_fractional_milp(sub, phi, cutoff=cutoff,
                                denominator_positive=denominator_positive)
    if res.feasible:
        res.x = res.x[:problem.n]
    return res

