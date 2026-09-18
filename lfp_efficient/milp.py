"""Exact branch & bound over the rational simplex.

Handles both objectives used by the algorithm:

* ``max c'x``                        -- lower bounds ``M_i``, efficiency test,
* ``max (U'x+alpha)/(V'x+beta)``     -- the main criterion ``Phi``.

Bounding is valid in both cases: the continuous relaxation of a maximisation
is an upper bound on its integer optimum, and for the fractional case the
Cambini-Martein simplex returns the true continuous optimum of the ratio
(the objective is pseudo-concave on ``{V'x + beta > 0}``).
"""

from dataclasses import dataclass, field
from fractions import Fraction
from math import ceil, floor
from typing import List, Optional, Sequence

from .model import EQ, GE, LE, Model
from .rational import F, ZERO, fmt, is_integral
from .simplex import (FractionalPricing, INFEASIBLE, LinearPricing, OPTIMAL,
                      SimplexResult, Tableau, UNBOUNDED, solve_standard_form)

#: returned when a *cutoff* was supplied and nothing beats it.  It means
#: "the region is not empty, but its optimum is <= cutoff" -- which is all the
#: caller asked about, and is distinct from a genuinely infeasible region.
CUTOFF = "cutoff"


@dataclass
class MilpResult:
    """Outcome of a branch & bound run."""

    status: str                                   # optimal / infeasible / unbounded
    x: Optional[List[Fraction]] = None            # optimal point, model variables only
    objective: Optional[Fraction] = None
    #: tableau of the LP relaxation of the node that produced the incumbent.
    #: Its basic solution *is* ``x`` (padded with slacks), which is what the
    #: edge exploration of Definition 2 needs.
    tableau: Optional[Tableau] = None
    n_structural: int = 0
    nodes: int = 0

    @property
    def feasible(self) -> bool:
        return self.status == OPTIMAL

    def __str__(self) -> str:
        if not self.feasible:
            return f"<{self.status}>"
        pt = "(" + ", ".join(fmt(v) for v in self.x) + ")"
        return f"{pt} -> {fmt(self.objective)}"


def _pricing_for(objective, n_cols: int):
    """Build the simplex pricing rule for a model objective, padded to ``n_cols``."""
    if isinstance(objective, (list, tuple)):                      # linear c
        c = list(objective) + [ZERO] * (n_cols - len(objective))
        return LinearPricing(c)
    obj = objective.lift(n_cols)                                  # fractional Phi
    return FractionalPricing(obj.U, obj.alpha, obj.V, obj.beta)


def solve_relaxation(model: Model, objective) -> SimplexResult:
    """Continuous relaxation of *model* (integrality simply ignored)."""
    A, b, n_struct = model.to_standard_form()
    if not A:                                                     # no constraint at all
        return SimplexResult(UNBOUNDED)
    n_cols = len(A[0])
    return solve_standard_form(A, b, _pricing_for(objective, n_cols), n_struct)


def solve_milp(model: Model, objective, max_nodes: int = 200_000,
               cutoff: Optional[Fraction] = None) -> MilpResult:
    """Depth-first branch & bound with incumbent pruning.

    *objective* is either a list of coefficients (linear) or a
    :class:`~lfp_efficient.model.FractionalObjective`.  Branching is done on
    the most fractional integer-constrained variable, using the usual dichotomy
    ``x_j <= floor(v)`` / ``x_j >= ceil(v)``.

    *cutoff* turns the run into the question "is there a feasible point with an
    objective **strictly greater** than this value, and if so which is best?".
    Nodes whose relaxation cannot beat it are pruned right away and the status
    ``CUTOFF`` is returned when none survives.  The main algorithm only ever
    needs to know whether the bound over the truncated region improves on the
    incumbent, so passing the incumbent as a cutoff prunes most of the tree in
    the late iterations -- exactly where the accumulated Sylva-Crema binaries
    would otherwise make each sub-problem expensive.
    """
    best = MilpResult(INFEASIBLE)
    nodes = 0
    any_feasible = False
    stack: List[Model] = [model]

    while stack:
        if nodes >= max_nodes:
            raise RuntimeError("branch & bound node limit reached: is the region bounded?")
        node = stack.pop()
        nodes += 1
        relax = solve_relaxation(node, objective)

        if relax.status == INFEASIBLE:
            continue
        if relax.status == UNBOUNDED:
            return MilpResult(UNBOUNDED, nodes=nodes)
        any_feasible = True

        # bound: the relaxation cannot beat the incumbent (or the cutoff) -> prune
        if best.feasible and relax.objective <= best.objective:
            continue
        if cutoff is not None and not best.feasible and relax.objective <= cutoff:
            continue

        x = relax.x[:node.n]
        # most-fractional branching: it splits the region more evenly than
        # taking the first fractional variable, and cuts the node count
        frac, worst = None, None
        for j in range(node.n):
            if node.integrality[j] and not is_integral(x[j]):
                gap = x[j] - floor(x[j])
                score = min(gap, 1 - gap)
                if worst is None or score > worst:
                    frac, worst = j, score

        if frac is None:                                          # integral -> incumbent
            if not best.feasible or relax.objective > best.objective:
                best = MilpResult(OPTIMAL, x, relax.objective,
                                  relax.tableau, relax.n_structural, nodes)
            continue

        value = x[frac]
        down = node.copy()
        down.add([F(1) if j == frac else ZERO for j in range(node.n)], LE, floor(value))
        up = node.copy()
        up.add([F(1) if j == frac else ZERO for j in range(node.n)], GE, ceil(value))
        stack.extend((down, up))                                  # DFS, "up" explored first

    best.nodes = nodes
    if not best.feasible and cutoff is not None and any_feasible:
        return MilpResult(CUTOFF, nodes=nodes)
    return best


def solve_linear_milp(model: Model, c: Sequence[Fraction], minimize: bool = False) -> MilpResult:
    """Convenience wrapper: ``max c'x`` (or ``min`` when *minimize* is set)."""
    coeffs = [F(v) for v in c]
    if minimize:
        res = solve_milp(model, [-v for v in coeffs])
        if res.feasible:
            res.objective = -res.objective
        return res
    return solve_milp(model, coeffs)


# --------------------------------------------------------------------------
# Linear fractional MILP with a denominator that may change sign
# --------------------------------------------------------------------------
def _integer_scale(values: Sequence[Fraction]) -> int:
    """Smallest ``s > 0`` such that ``s * v`` is an integer for every ``v``."""
    s = 1
    for v in values:
        d = v.denominator
        g = _gcd(s, d)
        s = s * d // g
    return s


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def solve_fractional_milp(model: Model, phi, max_nodes: int = 200_000,
                          cutoff: Optional[Fraction] = None) -> MilpResult:
    """``max (U'x+alpha)/(V'x+beta)`` over the integer points of *model*.

    Linear fractional programming always assumes ``V'x + beta > 0`` on the
    feasible set, because the ratio is pseudo-concave only there -- and the
    relaxation bound used by branch & bound is valid only under that same
    assumption.  The numerical example of the paper does *not* satisfy it:
    ``x = (0,0)`` belongs to ``D`` and gives ``5*0 + 0 - 1 = -1 < 0`` (this is
    precisely the point where the paper reads ``Phi_sup = Phi(0,0) = 1``).

    So the feasible set is split along the sign of the denominator::

        D+ = D and { s(V'x + beta) >= +1 }      Phi is a genuine LFP there
        D- = D and { s(V'x + beta) <= -1 }      Phi = (-U'x-alpha)/(-V'x-beta)

    where ``s`` scales ``(V, beta)`` to integers, so that for integer ``x`` the
    quantity ``s(V'x+beta)`` is an integer and the two branches cover every
    feasible point except those with a vanishing denominator (where ``Phi`` is
    undefined and which are therefore -- deliberately -- discarded).
    The answer is the better of the two branches.
    """
    phi = phi.lift(model.n)
    s = _integer_scale(list(phi.V) + [phi.beta])
    den_row = [s * v for v in phi.V]
    den_const = s * phi.beta

    best = MilpResult(INFEASIBLE)
    nodes = 0
    saw_cutoff = False
    for sign in (+1, -1):
        branch = model.copy()
        if sign > 0:
            #  s(V'x + beta) >= 1
            branch.add(den_row, GE, F(1) - den_const)
            objective = phi
        else:
            #  s(V'x + beta) <= -1, objective rewritten with a positive denominator
            branch.add(den_row, LE, F(-1) - den_const)
            objective = phi.__class__([-u for u in phi.U], [-v for v in phi.V],
                                      -phi.alpha, -phi.beta)
        best_so_far = best.objective if best.feasible else cutoff
        res = solve_milp(branch, objective, max_nodes=max_nodes, cutoff=best_so_far)
        nodes += res.nodes
        if res.status == UNBOUNDED:
            return MilpResult(UNBOUNDED, nodes=nodes)
        if res.status == CUTOFF:
            saw_cutoff = True
        if res.feasible and (not best.feasible or res.objective > best.objective):
            res.x = res.x[:model.n]
            best = res
    best.nodes = nodes
    if not best.feasible and saw_cutoff:
        return MilpResult(CUTOFF, nodes=nodes)
    return best
