"""Efficiency test, dominance repair and truncation of the feasible set.

Everything here is written once, in terms of the linear form ``e_k`` of the
fractional criteria (:meth:`~lfp_efficient.model.MOILFP.e_row`):

    e_k(x ; a) = D_k(a) * D_k(x) * ( Z_k(x) - Z_k(a) )

which is *linear in x* and carries the **sign** of ``Z_k(x) - Z_k(a)`` because
both denominators are positive.  With integer data it is integer-valued on
integer points, so the three questions the method keeps asking --

    is x better than a on criterion k?      e_k(x) >= 1
    is x equal to a on criterion k?         e_k(x) == 0
    is x at least as good on criterion k?   e_k(x) >= 0

-- are all exact integer linear conditions.  On a *linear* criterion
(``d_k = 0``, ``b_k = 1``) ``e_k`` collapses to ``C_k x - C_k a`` and every
formula below becomes the one the paper writes.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Sequence

from .milp import (MilpResult, solve_fractional_milp, solve_linear_milp,
                   solve_milp)
from .model import EQ, GE, LE, MOILFP, Model
from .rational import F, ZERO, dot
from .simplex import OPTIMAL


# --------------------------------------------------------------------------
# Efficiency test
# --------------------------------------------------------------------------
@dataclass
class EfficiencyTest:
    """Result of the efficiency test applied to a point ``x*``."""

    efficient: bool
    #: optimal value ``theta*`` of the test; ``x*`` is efficient iff it is 0
    psi: Fraction
    #: when ``x*`` is not efficient, a point that **dominates** it.  In the
    #: linear case this point is itself efficient; in the fractional case it
    #: need not be -- see :func:`repair_to_efficient`.
    witness: Optional[List[Fraction]] = None


def test_efficiency(problem: MOILFP, x_star: Sequence[Fraction],
                    region: Optional[Model] = None) -> EfficiencyTest:
    """Exact efficiency test, in a single integer linear program::

        theta(x*) = max  sum_k e_k(x ; x*)
                    s.t. e_k(x ; x*) >= 0 ,  k = 1..p
                         x in D

    ``x*`` is feasible for it (with value 0), so ``theta >= 0`` always, and

        x* is efficient  <=>  theta(x*) = 0 .

    Every coefficient is an integer, so ``theta`` is an integer and the test is
    exact -- no tolerance, unlike a formulation written directly on the ratios.

    On linear criteria this is Isermann's test as the paper states it: ``e_k``
    becomes ``C_k x - C_k x*``, the constraints become ``C x >= C x*`` and the
    objective ``sum_k (C_k x - C_k x*)``, which is the paper's
    ``max sum Psi`` under ``C x - I Psi = C x*``, ``Psi >= 0``.

    **What the maximiser is, and is not.** When ``theta > 0`` the optimum
    dominates ``x*``.  With linear criteria it is moreover *efficient* (Ecker &
    Kouada), because the objective is then exactly
    ``sum_k (Z_k(x) - Z_k(x*))``, which is monotone in ``Z``.  With fractional
    criteria it is **not**: the k-th term carries a factor ``D_k(x)`` that
    varies from point to point, so a dominating point can score lower than a
    dominated one.  Use :func:`repair_to_efficient` to go from *dominating* to
    *efficient*.

    The test is always run on the *original* region ``D``: efficiency is a
    property of the problem, never of a truncated region.  ``region`` is
    exposed only so that a caller can deliberately test over a sub-region.
    """
    base = (region or problem.model).copy()
    n = base.n

    objective = [ZERO] * base.n
    constant = ZERO
    for k in range(problem.p):
        coeffs, const = problem.e_row(k, x_star)
        row = list(coeffs) + [ZERO] * (base.n - len(coeffs))
        base.add(row, GE, -const)                 # e_k(x) >= 0
        objective = [a + b for a, b in zip(objective, row)]
        constant += const

    res = solve_milp(base, objective)
    if res.status != OPTIMAL:
        # x* itself is feasible for the test, so this cannot happen unless x*
        # is outside D.
        raise ValueError("the efficiency test is infeasible: is x* in D?")

    theta = res.objective + constant
    if theta == 0:
        return EfficiencyTest(True, theta, None)
    return EfficiencyTest(False, theta, res.x[:n])


def repair_to_efficient(problem: MOILFP, x: Sequence[Fraction],
                        max_steps: int = 1000) -> List[Fraction]:
    """Follow the dominance chain from *x* until an efficient point is reached.

    Each step replaces the current point by one that dominates it, so the
    criterion vectors strictly increase along the chain; ``D`` being finite,
    the walk stops, and it stops exactly where the test returns ``theta = 0``.

    With linear criteria the chain has length at most one -- the first
    maximiser is already efficient -- so this costs a single extra test that
    immediately returns ``theta = 0``.  It is the fractional case that needs
    the walk, because there the maximiser of the test is only guaranteed to
    dominate.
    """
    current = list(x)
    for _ in range(max_steps):
        outcome = test_efficiency(problem, current)
        if outcome.efficient:
            return current
        current = outcome.witness
    raise RuntimeError("dominance chain did not terminate")


# --------------------------------------------------------------------------
# Bounds
# --------------------------------------------------------------------------
def lower_bounds(problem: MOILFP) -> List[Fraction]:
    """``M_k = min { numerator of Z_k over D }``, the paper's lower bounds.

    Reported for the linear case, where a criterion *is* its numerator and this
    is the ``M_i`` of equation (5); the paper's example gives ``(-3, -3)``.
    The cuts do not read it -- they need a bound on ``e_k``, which depends on
    the cut centre and is obtained in closed form by :func:`_e_lower_bound`
    from the extremes of :func:`criterion_extremes`.
    """
    bounds = []
    for k in range(problem.p):
        z = problem.criteria[k]
        res = solve_linear_milp(problem.model, list(z.U), minimize=True)
        if res.status != OPTIMAL:
            raise ValueError("D must be non-empty and bounded to compute the lower bounds")
        bounds.append(res.objective + z.alpha)
    return bounds


def criterion_extremes(problem: MOILFP) -> List[tuple]:
    """``(min N_k, min D_k, max D_k)`` over ``D``, for every criterion.

    Computed once per problem and cached on it: three integer programs per
    criterion, paid at the start rather than at every cut.
    """
    cached = getattr(problem, "_extremes", None)
    if cached is not None:
        return cached
    out = []
    for k in range(problem.p):
        z = problem.criteria[k]
        num_min = solve_linear_milp(problem.model, list(z.U), minimize=True)
        den_min = solve_linear_milp(problem.model, list(z.V), minimize=True)
        den_max = solve_linear_milp(problem.model, list(z.V))
        if OPTIMAL not in (num_min.status, den_min.status, den_max.status):
            raise ValueError("D must be non-empty and bounded")
        out.append((num_min.objective + z.alpha,
                    den_min.objective + z.beta,
                    den_max.objective + z.beta))
    problem._extremes = out
    return out


def _e_lower_bound(problem: MOILFP, k: int, x_hat: Sequence[Fraction]) -> Fraction:
    """A valid lower bound on ``e_k( . ; x_hat)`` over ``D``, in closed form.

    ``e_k(x) = D_k(x_hat) N_k(x) - N_k(x_hat) D_k(x)`` with ``D_k(x_hat) > 0``,
    so bounding the two terms separately bounds the whole::

        min e_k  >=  D_k(x_hat) * min N_k  -  N_k(x_hat) * ( max D_k  if N_k(x_hat) > 0
                                                             min D_k  otherwise )

    The three extremes come from :func:`criterion_extremes`, computed once for
    the problem, so a cut costs no optimisation at all to bound itself.

    Bounding this way rather than by a linear program over the *current* region
    matters for more than the LP itself: the extremes are integer minima over
    ``D``, which is the bound the linear case always used -- on linear criteria
    this expression collapses to ``min C_k x - C_k x_hat``, exactly equation
    (5)'s ``M_i`` shifted to the cut centre. Deriving the bound from a
    relaxation instead made the big-M looser, and on the instance with eleven
    cut iterations that cost a factor of ten in run time.
    """
    num_min, den_min, den_max = criterion_extremes(problem)[k]
    n_bar = problem.numerator(k, x_hat)
    d_bar = problem.denominator(k, x_hat)
    worst_den = den_max if n_bar > 0 else den_min
    return d_bar * num_min - n_bar * worst_den


# --------------------------------------------------------------------------
# Truncation
# --------------------------------------------------------------------------
def add_dominance_cut(region: Model, problem: MOILFP,
                      x_hat: Sequence[Fraction]) -> Model:
    """Remove from *region* every point ``x`` with ``Z(x) <= Z(x_hat)``.

    A point survives exactly when it beats ``x_hat`` on some criterion, which
    ``e_k`` turns into an integer linear disjunction::

        exists k :  e_k(x ; x_hat) >= 1

    modelled with one binary per criterion::

        e_k(x) >= 1 * y_k + M_k (1 - y_k) ,   k = 1..p
        sum_k y_k >= 1

    where ``M_k`` is a lower bound on ``e_k`` over the region, so that
    ``y_k = 0`` leaves the row vacuous and ``y_k = 1`` forces a *strict*
    improvement.  The ``>= 1`` is exact because ``e_k`` is integer-valued:
    no minimal step has to be estimated, which is what makes the transposition
    to ratios faithful rather than approximate.

    On linear criteria this is equation (5) of the paper verbatim, with
    ``e_k(x) = C_k x - C_k x_hat`` turning ``e_k >= 1`` into
    ``C_k x >= C_k x_hat + 1``.

    What the centre has to be, and what it does not
    -----------------------------------------------
    Validity needs only that ``x_hat`` is **feasible**.  Nothing above appeals
    to its efficiency: the block asserts that surviving points improve some
    criterion, and the centre's efficiency plays no part in that.

    Efficiency of the centre decides something else -- whether the slice the
    cut destroys held anything worth keeping.  Two regimes:

    * a **dominated** centre closes by itself.  No efficient point can share
      its criterion vector, since whatever dominates the centre would dominate
      that point too, so nothing efficient is removed and nothing has to be
      harvested first.
    * an **efficient** centre does not.  Every point of
      ``{x : Z(x) = Z(x_hat)}`` is efficient too and is about to be removed, so
      the caller must first establish that none of them beats the incumbent --
      which is precisely what :func:`best_with_same_criterion` is for.

    The algorithm always cuts on the efficient point handed over by the
    dominance repair, so it is always in the second regime and that harvest is
    a requirement rather than a refinement.  Cutting on the dominated maximiser
    instead would drop it -- and was measured at over 300x slower, because the
    efficient centre's cut is strictly the larger one and the dominated one has
    to grind the non-efficient band away point by point.

    (The two-regime reading, and the fractional form of this cut, are due to
    the ``claude/lnatawruh-8gyw92`` branch; the 300x figure is measured here.)
    """
    region = region.copy()
    y = region.add_variables(problem.p, integer=True)
    for k in range(problem.p):
        coeffs, const = problem.e_row(k, x_hat)
        padded = list(coeffs) + [ZERO] * (region.n - len(coeffs))
        m_k = _e_lower_bound(problem, k, x_hat)
        # -e_k(x) + (1 - M_k) y_k <= -M_k
        #
        # written in "<=" form on purpose: its slack column is a +1 unit vector
        # whenever -M_k >= 0, so the simplex's crash basis covers the row and
        # no phase-I artificial is needed for it -- p fewer artificials per cut
        # in every node of the branch & bound.
        row = [-c for c in padded]
        row[y[k]] = F(1) - m_k
        region.add(row, LE, const - m_k)
    # sum_k y_k >= 1
    #
    # The explicit "y_k <= 1" rows are redundant and left out: writing
    # K_k = 1 - M_k >= 1, a point satisfies the block iff some y_k >= 1, and
    # then e_k(x) >= 1; values y_k >= 2 only impose a stronger requirement, so
    # the union over the integer y is unchanged while the region loses p rows
    # per iteration.
    pick = [ZERO] * region.n
    for j in y:
        pick[j] = F(1)
    region.add(pick, GE, F(1))
    return region


#: the name the linear case is known by, kept so existing code reads unchanged
add_sylva_crema_cut = add_dominance_cut


def best_with_same_criterion(region: Model, problem: MOILFP,
                             x_tilde: Sequence[Fraction], phi,
                             cutoff=None,
                             denominator_positive: bool = False) -> MilpResult:
    """``Q(x~) = max { Phi(x) : x in D, Z(x) = Z(x~) }``.

    The slice is cut out by ``e_k(x ; x~) = 0`` for every ``k`` -- linear rows,
    exactly, because ``e_k`` vanishes precisely where the ratios agree.

    All its points share the non-dominated criterion vector ``Z(x~)``, hence
    they are all efficient as soon as ``x~`` is, and the cut is about to delete
    the whole slice.  This is the closure condition that a cut centred on an
    *efficient* point owes, and not an optimisation: see
    :func:`add_dominance_cut` for why the obligation exists only in that regime.

    The slice is never touched by the cuts already made, so it is the same set
    in the truncated region and in ``D`` itself: a previous cut would exclude
    it only if ``Z(x~) <= Z(x^s)`` for some stored efficient ``x^s``, which
    forces equality (both are non-dominated) and would mean ``x~`` had already
    been cut away -- impossible, since it dominates a point of the current
    region.  Solving over ``D`` keeps the binaries and rows of every
    accumulated cut out of this sub-problem.
    """
    sub = problem.model.copy()
    for k in range(problem.p):
        coeffs, const = problem.e_row(k, x_tilde)
        row = list(coeffs) + [ZERO] * (sub.n - len(coeffs))
        sub.add(row, EQ, -const)
    res = solve_fractional_milp(sub, phi, cutoff=cutoff,
                                denominator_positive=denominator_positive)
    if res.feasible:
        res.x = res.x[:problem.n]
    return res


# --------------------------------------------------------------------------
# Supported efficient points, cheaply: the weighted-sum scalarisation
# --------------------------------------------------------------------------
# For any strictly positive weight vector w, a maximiser of w'Z over D is
# efficient.  With *linear* criteria that maximisation is one ordinary integer
# program -- no cut, no binary, no efficiency test -- so each point costs a
# fraction of what step 1 of the algorithm costs.  Only *supported* efficient
# points are reachable this way, which loses nothing here: whatever comes back
# is efficient, and that is all a cut centre or a lower bound needs.

def has_linear_criteria(problem: MOILFP) -> bool:
    """True when every criterion is a ratio with a constant denominator.

    That is the paper's own case, and the only one where ``w'Z(x)`` is linear
    in ``x`` and the scalarisation below is an integer *linear* program.
    """
    return all(not any(z.V) for z in problem.criteria)


def weighted_sum_efficient(problem: MOILFP,
                           w: Sequence[Fraction]) -> Optional[List[Fraction]]:
    """An efficient point of ``D``: ``argmax { w'Z(x) : x in D }``, ``w > 0``.

    The weights must be strictly positive -- with a zero weight the maximiser
    is only *weakly* efficient and the guarantee is lost.  Returns ``None``
    when the program has no optimum, never a point whose efficiency is in
    doubt.

    Requires linear criteria (:func:`has_linear_criteria`); a sum of ratios is
    not a linear objective and this construction does not transfer to it.
    """
    if not has_linear_criteria(problem):
        raise ValueError(
            "the weighted-sum scalarisation needs linear criteria: with "
            "fractional ones w'Z is a sum of ratios, not a linear objective")
    if any(wk <= 0 for wk in w):
        raise ValueError("the weights must be strictly positive, or the "
                         "maximiser is only weakly efficient")
    n = problem.n
    # Z_k(x) = (U_k'x + alpha_k) / beta_k with beta_k > 0, so the scalarised
    # objective is sum_k (w_k / beta_k) * U_k'x up to an additive constant.
    coeffs = [ZERO] * n
    for k, z in enumerate(problem.criteria):
        scale = F(w[k]) / z.beta
        for j in range(n):
            coeffs[j] += scale * z.U[j]
    res = solve_linear_milp(problem.model, coeffs)
    return res.x[:n] if res.status == OPTIMAL else None


def spread_weights(p: int, spread: int = 4) -> List[List[int]]:
    """A small spread of strictly positive weights: all-ones, then one per
    criterion leaning on it.  ``p + 1`` vectors, so ``p + 1`` programs."""
    return ([[1] * p]
            + [[spread if i == k else 1 for i in range(p)] for k in range(p)])
