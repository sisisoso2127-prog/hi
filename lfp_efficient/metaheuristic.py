"""A Pareto local search, and the exact--metaheuristic hybrid it feeds.

Why a heuristic incumbent is worth anything here -- and why it was not before
----------------------------------------------------------------------------
Handing the *paper's* method a good incumbent buys nothing.  Measured by giving
it the true optimum for free at iteration 1, all three reference instances
still took exactly as many iterations: what gets cut is decided by the
efficiency test on ``x_l``, not by ``Phi_opt``.

The box search of :mod:`lfp_efficient.criterion_space` is different in kind.  A
box whose bound fails to beat the incumbent is discarded **whole**, its integer
program never solved.  Measured with the true optimum handed over for free:
8.62s -> 6.05s over 18 instances, up to 56% fewer boxes solved.

So the value of a heuristic incumbent depends on which exact method it is
hybridised with -- nothing in one, a real mechanism in the other.  That 1.42x
is a *ceiling*: a real heuristic returns a value at most the optimum, and gets
proportionally less.

Where correctness lives
-----------------------
The incumbent must be a lower bound on ``max { Phi(x) : x efficient }``.  A
point that merely *looks* good is not enough: hand over the value of a
**dominated** point and the search may prune away the true optimum.

So nothing leaves this module unverified.  Every candidate the search proposes
is either confirmed efficient by the exact test of Theorem 1, or walked to an
efficient point by :func:`repair_to_efficient` -- and it is that point's value,
not the candidate's, that is returned.  The heuristic decides *where to look*;
it never decides what is true.

The search
----------
Pareto local search, in its usual form: the archive **is** the search.

* Seed the archive with a few random maximal feasible points -- increment
  random coordinates while feasibility holds, since efficient points live on
  the boundary.
* Repeatedly take an unexplored archive member, offer every unit neighbour to
  the archive, and mark whatever entered as unexplored in turn.
* Stop on an evaluation budget.

Restarting from fresh random points instead, which is what the first version
here did, throws the archive away between walks: measured on the heaviest
instance it left 4 archive points and an incumbent at 36% of the optimum, and
spending four times the effort did not fix it monotonically (0.360, 0.280,
0.458).

Exact arithmetic is not used inside the walk.  Nothing there is proved, the
model data is integral, and comparing two ratios needs no division:
``Phi(x) > Phi(y)`` is ``(Ux+a)(Vy+b) > (Uy+a)(Vx+b)`` once both denominators
are positive.  Integer arithmetic in the walk, exact rationals the moment a
candidate is handed to the exact machinery.
"""

import random
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

from .algorithm import Solution
from .criterion_space import optimize_in_criterion_space
from .efficiency import repair_to_efficient, test_efficiency
from .model import GE, LE, FractionalObjective, MOILFP, Model
from .rational import F, ZERO


class _IntegerModel:
    """The model and ``Phi`` as plain integers, for the walk only.

    The walk asks two questions millions of times -- is this point feasible,
    and is its ratio better -- and neither needs exact rational arithmetic to
    steer a heuristic.  Rows are scaled to integers once; ratios are compared
    by cross-multiplication.  Returns ``None`` from :func:`build` when the data
    is not integral, and the caller then falls back to the exact path.
    """

    __slots__ = ("n", "rows", "criteria", "U", "alpha", "V", "beta")

    def __init__(self, n, rows, criteria, U, alpha, V, beta):
        self.n, self.rows, self.criteria = n, rows, criteria
        self.U, self.alpha, self.V, self.beta = U, alpha, V, beta

    @staticmethod
    def build(problem: MOILFP, phi: FractionalObjective):
        def ints(values):
            out = []
            for v in values:
                f = F(v)
                if f.denominator != 1:
                    return None
                out.append(int(f))
            return out

        rows = []
        for c in problem.model.constraints:
            coeffs = ints(c.coeffs)
            rhs = ints([c.rhs])
            if coeffs is None or rhs is None:
                return None
            rows.append((coeffs, c.sense, rhs[0]))

        criteria = []
        for z in problem.criteria:
            num, den = ints(list(z.U) + [z.alpha]), ints(list(z.V) + [z.beta])
            if num is None or den is None:
                return None
            criteria.append((num, den))

        lifted = phi.lift(problem.n)
        U, V = ints(list(lifted.U) + [lifted.alpha]), ints(list(lifted.V) + [lifted.beta])
        if U is None or V is None:
            return None
        return _IntegerModel(problem.n, rows, criteria, U[:-1], U[-1], V[:-1], V[-1])

    def feasible(self, x) -> bool:
        for coeffs, sense, rhs in self.rows:
            lhs = 0
            for c, v in zip(coeffs, x):
                if c:
                    lhs += c * v
            if sense == LE:
                if lhs > rhs:
                    return False
            elif sense == GE:
                if lhs < rhs:
                    return False
            elif lhs != rhs:
                return False
        return True

    def criterion_vector(self, x):
        """``Z(x)`` as a tuple of ``(numerator, denominator)`` pairs."""
        out = []
        for num, den in self.criteria:
            a = num[-1] + sum(c * v for c, v in zip(num, x) if c)
            b = den[-1] + sum(c * v for c, v in zip(den, x) if c)
            out.append((a, b))
        return tuple(out)

    def phi_parts(self, x):
        a = self.alpha + sum(c * v for c, v in zip(self.U, x) if c)
        b = self.beta + sum(c * v for c, v in zip(self.V, x) if c)
        return a, b


def _ratio_ge(left, right) -> bool:
    """``left >= right`` for two ``(numerator, denominator)`` pairs.

    Written for positive denominators, which is the standing assumption of
    linear fractional programming and is what the criteria satisfy here; a
    non-positive one makes the comparison meaningless and is reported as False
    so the walk simply does not follow it.
    """
    (a, b), (c, d) = left, right
    if b <= 0 or d <= 0:
        return False
    return a * d >= c * b


class ParetoArchive:
    """Mutually non-dominated points, in the criterion space of *problem*.

    Criterion vectors are held as ``(numerator, denominator)`` integer pairs so
    the dominance test needs no division; ``points`` hands back ordinary
    rational vectors for the exact machinery downstream.
    """

    def __init__(self, problem: MOILFP):
        self.problem = problem
        self.entries: List[Tuple[tuple, List[int]]] = []

    def add(self, x: Sequence[int], z: tuple) -> bool:
        """Offer a point.  True when it entered: nothing here dominates it."""
        kept = []
        for other, point in self.entries:
            if _dominates(other, z):
                return False
            if not _dominates(z, other):
                kept.append((other, point))
        if any(p == list(x) for _, p in kept):
            return False
        kept.append((z, list(x)))
        self.entries = kept
        return True

    def points(self) -> List[List[Fraction]]:
        return [[F(v) for v in p] for _, p in self.entries]

    def __len__(self) -> int:
        return len(self.entries)


def _dominates(a: tuple, b: tuple) -> bool:
    """``a`` dominates ``b``: at least as good on every ratio, better on one."""
    better = False
    for u, v in zip(a, b):
        if not _ratio_ge(u, v):
            return False
        if not _ratio_ge(v, u):
            better = True
    return better


def random_maximal_point(model: Model, rng: random.Random,
                         integer_model=None) -> Optional[List[int]]:
    """A feasible point that cannot be increased on any single coordinate.

    Built by walking up from the origin one unit at a time along randomly
    chosen coordinates.  Efficient points sit on the boundary of ``D``, so a
    maximal point is a far better place to start than a random interior one --
    and the construction needs no solver.

    A coordinate that fails is set aside, but every coordinate is reconsidered
    as soon as some other one succeeds: with a negative coefficient anywhere in
    ``A`` a step up can **relax** a row, so a move that was infeasible a moment
    ago need not stay so.  Dropping coordinates permanently returned points
    that were not maximal at all -- on the paper's own instance it stopped at
    ``(3,2)`` with ``(3,3)`` feasible.
    """
    feasible = (integer_model.feasible if integer_model is not None
                else (lambda y: model.is_feasible([F(v) for v in y])))
    x = [0] * model.n
    if not feasible(x):
        return None
    free = list(range(model.n))
    rng.shuffle(free)
    while free:
        j = free[-1]
        x[j] += 1
        if feasible(x):
            free = list(range(model.n))     # a relaxed row may reopen others
            rng.shuffle(free)
        else:
            x[j] -= 1
            free.pop()
    return x


def _neighbours(x: Sequence[int], n: int):
    """Unit steps, then swaps: one coordinate down and another up."""
    for j in range(n):
        for delta in (1, -1):
            if x[j] + delta < 0:
                continue
            y = list(x)
            y[j] += delta
            yield y
    for j in range(n):
        if x[j] <= 0:
            continue
        for k in range(n):
            if k == j:
                continue
            y = list(x)
            y[j] -= 1
            y[k] += 1
            yield y


def pareto_local_search(problem: MOILFP, phi: FractionalObjective,
                        seeds: int = 8, budget: int = 4000,
                        seed: int = 0) -> ParetoArchive:
    """Fill an archive of non-dominated points.  Nothing here is proved.

    The archive drives the search: a member that has not been explored yet has
    its neighbourhood offered to the archive, and whatever enters becomes a
    member to explore in turn.  *budget* caps the number of neighbours
    evaluated, which is what bounds the run time.

    The neighbourhood has to include **swaps** -- one coordinate down, another
    up -- and not only unit steps.  From a maximal point a step up is
    infeasible by construction, and a step down lowers every criterion whose
    coefficients are non-negative, so the point it reaches is dominated and the
    archive refuses it.  With unit steps alone the walk therefore dies at once:
    measured, three archive points and an incumbent at 13% of the optimum.  A
    swap keeps the point on the boundary and is what moves along the frontier.
    """
    rng = random.Random(seed)
    archive = ParetoArchive(problem)
    fast = _IntegerModel.build(problem, phi)
    if fast is None:
        return archive                    # non-integral data: not our business

    frontier: List[List[int]] = []
    for _ in range(seeds):
        start = random_maximal_point(problem.model, rng, fast)
        if start is None:
            break
        if archive.add(start, fast.criterion_vector(start)):
            frontier.append(start)

    evaluated = 0
    while frontier and evaluated < budget:
        current = frontier.pop(rng.randrange(len(frontier)))
        for y in _neighbours(current, fast.n):
            evaluated += 1
            if evaluated > budget:
                break
            if not fast.feasible(y):
                continue
            if archive.add(y, fast.criterion_vector(y)):
                frontier.append(y)
    return archive


def metaheuristic_incumbent(problem: MOILFP, phi: FractionalObjective,
                            seeds: int = 8, budget: int = 4000,
                            candidates: int = 4,
                            seed: int = 0) -> Optional[Tuple[List[Fraction], Fraction]]:
    """A **verified efficient** point and its ``Phi``, or ``None``.

    The archive's best-scoring members are taken in order and put through the
    exact machinery: confirmed efficient, or walked to an efficient point whose
    value is returned in their place.  Only *candidates* of them are examined,
    since each costs one integer program and the point of the exercise is to be
    cheap next to the exact search that follows.
    """
    archive = pareto_local_search(problem, phi, seeds, budget, seed)
    if not len(archive):
        return None

    scored = []
    for point in archive.points():
        try:
            scored.append((phi(point), point))
        except ZeroDivisionError:
            continue
    scored.sort(key=lambda pair: pair[0], reverse=True)

    best: Optional[Tuple[List[Fraction], Fraction]] = None
    for _, point in scored[:candidates]:
        test = test_efficiency(problem, point)
        efficient = point if test.efficient else repair_to_efficient(problem,
                                                                    test.witness)
        try:
            value = phi(efficient)
        except ZeroDivisionError:
            continue
        if best is None or value > best[1]:
            best = (efficient, value)
    return best


def optimize_hybrid_metaheuristic(problem: MOILFP, phi: FractionalObjective,
                                  seeds: int = 8, budget: int = 4000,
                                  candidates: int = 4, seed: int = 0,
                                  time_budget: Optional[float] = None,
                                  verbose: bool = False) -> Solution:
    """Metaheuristic first, then the exact box search seeded with what it found.

    The answer is exactly the one the box search returns on its own -- same
    optimum, same proof -- because the only thing crossing from the heuristic
    is a lower bound attained at a point the exact test has certified.  What
    changes is how many boxes ever have to be solved.
    """
    found = metaheuristic_incumbent(problem, phi, seeds, budget,
                                    candidates, seed)
    if verbose:
        if found is None:
            print("--- metaheuristic found nothing usable")
        else:
            point, value = found
            print(f"--- metaheuristic: Phi = {value} at "
                  f"({', '.join(str(v) for v in point)}), verified efficient")

    if found is None:
        return optimize_in_criterion_space(problem, phi, time_budget=time_budget,
                                           verbose=verbose)
    point, value = found
    return optimize_in_criterion_space(problem, phi, time_budget=time_budget,
                                       incumbent=point, incumbent_value=value,
                                       explored_points=[point], verbose=verbose)
