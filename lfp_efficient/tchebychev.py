"""The augmented weighted Tchebychev program, as a generator of efficient points.

It is the tool the surrounding literature reaches for -- Chaabane, Brahmi and
Ramdani (2012) optimise a linear function over an integer efficient set with
it, and Younsi-Abbaci and Moulai (2021) use it over the Pareto front -- and it
is strictly stronger than the weighted sum already in :mod:`efficiency`.

The program
-----------
With ``z*`` the ideal point, ``z*_k = max { Z_k(x) : x in D }``, weights
``w > 0`` and an augmentation ``rho > 0``::

    min   max_k  w_k ( z*_k - Z_k(x) )  +  rho * sum_k ( z*_k - Z_k(x) )
    s.t.  x in D

which linearises with one continuous variable::

    min   lambda + rho * sum_k ( z*_k - Z_k(x) )
    s.t.  lambda >= w_k ( z*_k - Z_k(x) )   for every k
          x in D

Why its optimum is efficient, for **any** ``w > 0`` and ``rho > 0``
-------------------------------------------------------------------
Let ``x*`` solve it and suppose some ``y in D`` dominated it, so
``Z(y) >= Z(x*)`` with one strict inequality.  Every deviation
``z*_k - Z_k(y)`` is then no larger than at ``x*``, so the ``max`` term does
not increase; and the sum **strictly** decreases, because one term strictly
does.  The objective at ``y`` would be strictly smaller, contradicting
optimality.  So no such ``y`` exists.

That is what the augmentation buys.  Without ``rho`` the plain Tchebychev
program can be minimised by a merely **weakly** efficient point, where the sum
argument above is unavailable.

Why it is stronger than a weighted sum
--------------------------------------
:func:`lfp_efficient.weighted_sum_efficient` can only reach *supported*
efficient points -- those on the convex hull of the criterion image.  Varying
``w`` here reaches **every** efficient point, unsupported ones included, which
is the reason the literature uses it to enumerate or sample the front.

Measured: a good generator, a bad seed
--------------------------------------
Over the 20 random instances of ``examples/tchebychev_study.py``, from the same
grid of 27 weight vectors, this program reached 112 of 175 efficient points
against the weighted sum's 66, and 29 of the 54 *unsupported* ones against 0;
all 540 optima it returned were efficient.  A wider sweep over 40 instances
agrees (216/369 against 135/369, 56 unsupported against 0, 1080 optima and no
inefficient one).  As a generator it does what the literature says it does.

As a seed for the box search it **loses**, over the 18 instances of the hybrid
study: 9.64 s with no seed, 7.97 s seeded by the Pareto local search (1.21x,
exactly optimal on 12 of 18), 13.50 s seeded from here (0.71x, exactly optimal
on 5 of 18).  Two structural reasons, neither an artefact of this code.  The
cost is ``p + 1`` integer programs on an enlarged model -- and it is not the
reference point: ``z*`` took 0.01-0.13 s of the seed against 0.13-0.85 s for the
scalarisations.  The quality is the real one: **this program never looks at
Phi**.  Its weights steer in criterion space relative to ``z*``, while the local
search is guided by ``Phi`` at every move.  A good spread of efficient points is
not a good ``Phi``.

So :func:`tchebychev_incumbent` is kept as the reference the comparison needs,
not wired into the default hybrid.

Linear criteria only
--------------------
``z*_k - Z_k(x)`` has to be linear in ``x`` for the rows above to be
constraints of an integer *linear* program.  With a ratio it is not, and the
function says so rather than returning something unfounded.  The references
are for linear criteria too.
"""

from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

from .efficiency import has_linear_criteria, spread_weights
from .milp import solve_linear_milp
from .model import LE, FractionalObjective, MOILFP
from .rational import F, ZERO
from .simplex import OPTIMAL


def ideal_point(problem: MOILFP) -> List[Fraction]:
    """``z*_k = max { Z_k(x) : x in D }`` for each criterion, cached.

    One integer program per criterion, paid once for the problem.  The point is
    generally not attained by any single feasible ``x`` -- that is precisely why
    it is a useful reference for the Tchebychev distance.
    """
    cached = getattr(problem, "_ideal", None)
    if cached is not None:
        return cached
    if not has_linear_criteria(problem):
        raise ValueError("the ideal point is computed here for linear criteria; "
                         "with ratios max Z_k is not a linear program")
    out = []
    for z in problem.criteria:
        res = solve_linear_milp(problem.model, list(z.U))
        if res.status != OPTIMAL:
            raise ValueError("D must be non-empty and bounded")
        out.append((res.objective + z.alpha) / z.beta)
    problem._ideal = out
    return out


def augmented_tchebychev_efficient(
        problem: MOILFP, weights: Sequence[Fraction],
        rho: Fraction = Fraction(1, 1000),
        ideal: Optional[Sequence[Fraction]] = None
) -> Optional[List[Fraction]]:
    """An efficient point of ``D``: the optimum of the program above.

    Sound for any strictly positive *weights* and any positive *rho* -- see the
    module docstring for the two-line argument.  Returns ``None`` only when the
    program has no optimum.

    A zero weight is refused: the ``max`` term then ignores that criterion, and
    while the augmentation still rules out dominated optima, the guarantee is
    stated here for ``w > 0`` and is not weakened silently.
    """
    if not has_linear_criteria(problem):
        raise ValueError(
            "the augmented Tchebychev program needs linear criteria: with "
            "ratios z*_k - Z_k(x) is not linear and the rows below are not "
            "constraints of an integer linear program")
    if any(F(w) <= 0 for w in weights):
        raise ValueError("the weights must be strictly positive")
    if F(rho) <= 0:
        raise ValueError("rho must be positive, or a weakly efficient point "
                         "can minimise the program")

    z_star = list(ideal) if ideal is not None else ideal_point(problem)
    n = problem.n
    model = problem.model.copy()
    (lam,) = model.add_variables(1, integer=False)

    # lambda >= w_k (z*_k - Z_k(x))  <=>  -lambda - (w_k/beta_k) U_k'x
    #                                      <= -w_k z*_k + (w_k/beta_k) alpha_k
    for k, z in enumerate(problem.criteria):
        scale = F(weights[k]) / z.beta
        row = [ZERO] * model.n
        for j in range(n):
            row[j] = -scale * z.U[j]
        row[lam] = F(-1)
        model.add(row, LE, -F(weights[k]) * z_star[k] + scale * z.alpha)

    # min lambda + rho * sum_k (z*_k - Z_k(x));  the constant part is dropped
    objective = [ZERO] * model.n
    objective[lam] = F(1)
    for z in problem.criteria:
        scale = F(rho) / z.beta
        for j in range(n):
            objective[j] -= scale * z.U[j]

    res = solve_linear_milp(model, objective, minimize=True)
    return res.x[:n] if res.status == OPTIMAL else None


def tchebychev_incumbent(problem: MOILFP, phi: FractionalObjective,
                         weights: Optional[Sequence[Sequence[Fraction]]] = None,
                         rho: Fraction = Fraction(1, 1000)
                         ) -> Optional[Tuple[List[Fraction], Fraction]]:
    """The best ``Phi`` among the efficient points a spread of weights reaches.

    Directly comparable with
    :func:`lfp_efficient.metaheuristic.metaheuristic_incumbent`: both return a
    point already known to be efficient and the value it attains, and both are
    there to seed the exact search.  The difference is where the guarantee
    comes from -- here from the program itself, there from an explicit
    verification after the fact.
    """
    if weights is None:
        weights = spread_weights(problem.p)
    best: Optional[Tuple[List[Fraction], Fraction]] = None
    for w in weights:
        point = augmented_tchebychev_efficient(problem, w, rho)
        if point is None:
            continue
        try:
            value = phi(point)
        except ZeroDivisionError:
            continue
        if best is None or value > best[1]:
            best = (point, value)
    return best
