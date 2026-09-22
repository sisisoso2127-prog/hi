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
from time import monotonic
from typing import List, Optional, Sequence

from .model import EQ, GE, LE, Model
from .rational import F, ZERO, fmt, is_integral
from .rational import dot
from .simplex import (FractionalPricing, INFEASIBLE, LinearPricing, OPTIMAL,
                      STALLED, SimplexResult, Tableau, UNBOUNDED,
                      add_bound_row, add_linear_row, restore_feasibility,
                      solve_standard_form)

#: returned when a time budget expired before the search finished.  The result
#: then carries no proven optimum, but its ``bound`` is still a valid upper
#: bound on one -- which is all an anytime caller needs.
INTERRUPTED = "interrupted"

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
    #: the root relaxation of this sub-problem: its tableau, pricing and the
    #: reference prices frozen at its vertex.  A sub-problem whose region is
    #: this one plus a few rows can start from it instead of paying a phase I
    #: of its own -- see :func:`solve_milp`'s *warm* argument.
    root: Optional[tuple] = None
    #: a valid upper bound on the true optimum.  Equal to ``objective`` when the
    #: search ran to completion; on an interrupted search it is the largest
    #: relaxation value still open, which no feasible point can exceed.
    bound: Optional[Fraction] = None

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


def _node_bound_model(model: Model, branches) -> Model:
    """Rebuild a node as a plain model -- the fallback when a warm start stalls."""
    node = model.copy()
    for j, bound, upper in branches:
        row = [F(1) if k == j else ZERO for k in range(model.n)]
        node.add(row, LE if upper else GE, bound)
    return node


class _Node:
    """A branch & bound node: its solved tableau and how it was reached."""

    __slots__ = ("tableau", "pricing", "value", "x", "branches", "z1", "z2")

    def __init__(self, tableau, pricing, value, x, branches):
        self.tableau, self.pricing = tableau, pricing
        self.value, self.x, self.branches = value, x, branches
        # Z_1, Z_2 at this (feasible) vertex, frozen for the children's
        # restoration ratio test
        if isinstance(pricing, FractionalPricing):
            full = tableau.solution()
            self.z1 = dot(pricing.U, full) + pricing.alpha
            self.z2 = dot(pricing.V, full) + pricing.beta
        else:
            self.z1 = self.z2 = ZERO


def _child(model: Model, objective, parent: _Node, j: int, bound: int, upper: bool):
    """Solve the child obtained by adding one bound row, warm from the parent."""
    branches = parent.branches + [(j, bound, upper)]
    tab = parent.tableau.clone()
    pricing = add_bound_row(tab, parent.pricing, j, F(bound), upper)

    status = restore_feasibility(tab, pricing, parent.z1, parent.z2)
    if status == INFEASIBLE:
        return None, branches
    if status == STALLED:                       # give up warm, solve from scratch
        cold = solve_relaxation(_node_bound_model(model, branches), objective)
        if cold.status == INFEASIBLE:
            return None, branches
        if cold.status == UNBOUNDED:
            return UNBOUNDED, branches
        return _Node(cold.tableau, cold.pricing, cold.objective,
                     cold.x[:model.n], branches), branches

    if tab.run(pricing) == UNBOUNDED:
        return UNBOUNDED, branches
    full = tab.solution()
    return _Node(tab, pricing, pricing.value(full), full[:model.n], branches), branches


def _reference_point(tab, pricing):
    """``(z1, z2)`` frozen at this tableau's vertex, for a restoration's ratio
    test.  Zero for a linear objective, which does not use them."""
    if isinstance(pricing, FractionalPricing):
        full = tab.solution()
        return dot(pricing.U, full) + pricing.alpha, dot(pricing.V, full) + pricing.beta
    return ZERO, ZERO


def warm_relaxation(model: Model, objective, warm):
    """The root relaxation of *model*, started from a parent's solved tableau.

    *warm* is ``(tableau, pricing, z1, z2, rows)``: a relaxation already solved
    for a **superset** region, and the ``coeffs . x <= rhs`` rows that cut it
    down to this one.  The parent's basis stays a basis once each new slack
    joins it, so only the new rows can be primal infeasible and a dual
    restoration fixes that -- the same argument branch & bound already uses for
    its own children, applied one level up.

    Falls back to the cold solve whenever the restoration stalls, so a warm
    start that does not work out costs time and never correctness.
    """
    parent, parent_pricing, z1, z2, rows = warm
    tab = parent.clone()
    pricing = parent_pricing
    for coeffs, rhs in rows:
        pricing = add_linear_row(tab, pricing, coeffs, rhs)

    status = restore_feasibility(tab, pricing, z1, z2)
    if status == INFEASIBLE:
        return SimplexResult(INFEASIBLE)
    if status == STALLED:
        return solve_relaxation(model, objective)
    if tab.run(pricing) == UNBOUNDED:
        return SimplexResult(UNBOUNDED, tableau=tab, n_structural=model.n,
                             pricing=pricing)
    x = tab.solution()
    return SimplexResult(OPTIMAL, x, pricing.value(x), tab, model.n, pricing)


def solve_milp(model: Model, objective, max_nodes: int = 200_000,
               cutoff: Optional[Fraction] = None,
               deadline: Optional[float] = None,
               warm=None) -> MilpResult:
    """Depth-first branch & bound, warm started from the parent basis.

    *objective* is either a list of coefficients (linear) or a
    :class:`~lfp_efficient.model.FractionalObjective`.  Branching is done on
    the most fractional integer-constrained variable, using the usual dichotomy
    ``x_j <= floor(v)`` / ``x_j >= ceil(v)``.

    Each child is obtained from its parent's optimal tableau by appending the
    branch row and re-optimising (see
    :func:`~lfp_efficient.simplex.restore_feasibility`), so only the root pays
    a phase I.  A child whose restoration stalls is solved from scratch
    instead, which keeps the result independent of the warm start.

    *cutoff* turns the run into the question "is there a feasible point with an
    objective **strictly greater** than this value, and if so which is best?".
    Nodes whose relaxation cannot beat it are pruned right away and the status
    ``CUTOFF`` is returned when none survives.  The main algorithm only ever
    needs to know whether the bound over the truncated region improves on the
    incumbent, so passing the incumbent as a cutoff prunes most of the tree in
    the late iterations -- exactly where the accumulated Sylva-Crema binaries
    would otherwise make each sub-problem expensive.

    *deadline* is a ``time.monotonic()`` instant past which the search stops and
    reports ``INTERRUPTED``.  Stopping early does not throw the work away: every
    feasible point either was found -- so it is at most the incumbent -- or lies
    under a node still on the stack, so it is at most that node's relaxation
    value.  The maximum of those, together with any cutoff, is a valid upper
    bound on the optimum and is returned in ``bound``.
    """
    root = (warm_relaxation(model, objective, warm) if warm is not None
            else solve_relaxation(model, objective))
    if root.status == INFEASIBLE:
        return MilpResult(INFEASIBLE, nodes=1)
    if root.status == UNBOUNDED:
        return MilpResult(UNBOUNDED, nodes=1)
    root_z1, root_z2 = _reference_point(root.tableau, root.pricing)
    root_start = (root.tableau, root.pricing, root_z1, root_z2)

    best = MilpResult(INFEASIBLE)
    nodes = 0
    any_feasible = True
    stack: List[_Node] = [_Node(root.tableau, root.pricing, root.objective,
                                root.x[:model.n], [])]

    while stack:
        if nodes >= max_nodes:
            raise RuntimeError("branch & bound node limit reached: is the region bounded?")
        if deadline is not None and monotonic() > deadline:
            open_bounds = [n.value for n in stack]
            if best.feasible:
                open_bounds.append(best.objective)
            if cutoff is not None:
                open_bounds.append(cutoff)
            return MilpResult(INTERRUPTED, best.x, best.objective, nodes=nodes,
                              root=root_start,
                              bound=max(open_bounds) if open_bounds else None)
        node = stack.pop()
        nodes += 1

        # bound: the relaxation cannot beat the incumbent (or the cutoff) -> prune
        if best.feasible and node.value <= best.objective:
            continue
        if cutoff is not None and not best.feasible and node.value <= cutoff:
            continue

        # most-fractional branching: it splits the region more evenly than
        # taking the first fractional variable, and cuts the node count
        frac, worst = None, None
        for j in range(model.n):
            if model.integrality[j] and not is_integral(node.x[j]):
                gap = node.x[j] - floor(node.x[j])
                score = min(gap, 1 - gap)
                if worst is None or score > worst:
                    frac, worst = j, score

        if frac is None:                                          # integral -> incumbent
            if not best.feasible or node.value > best.objective:
                best = MilpResult(OPTIMAL, node.x, node.value,
                                  node.tableau, model.n, nodes,
                                  bound=node.value)
            continue

        value = node.x[frac]
        for bound, upper in ((ceil(value), False), (floor(value), True)):
            child, _ = _child(model, objective, node, frac, bound, upper)
            if child is None:
                continue
            if child is UNBOUNDED:
                return MilpResult(UNBOUNDED, nodes=nodes)
            stack.append(child)                   # DFS, "down" explored first

    best.nodes = nodes
    best.root = root_start
    if not best.feasible and cutoff is not None and any_feasible:
        return MilpResult(CUTOFF, nodes=nodes, root=root_start, bound=cutoff)
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


def denominator_stays_positive(model: Model, phi) -> bool:
    """Is ``V'x + beta > 0`` on the whole *continuous* relaxation of *model*?

    The question is asked about the relaxation, not about the integer points:
    the simplex walks the vertices of the relaxation, so it is there that the
    ratio has to stay well defined.  When the answer is yes -- the usual case,
    and the case of every instance whose denominator has non-negative
    coefficients and a positive constant -- the sign split of
    :func:`solve_fractional_milp` is pure waste and gets skipped, which halves
    the work of every fractional sub-problem the algorithm solves.
    """
    phi = phi.lift(model.n)
    res = solve_relaxation(model, [-v for v in phi.V])
    if res.status != OPTIMAL:
        return False                      # empty or unbounded: stay on the safe side
    return -res.objective + phi.beta > 0


def solve_fractional_milp(model: Model, phi, max_nodes: int = 200_000,
                          cutoff: Optional[Fraction] = None,
                          denominator_positive: bool = False,
                          deadline: Optional[float] = None,
                          warm=None) -> MilpResult:
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

    *denominator_positive* records that the caller has already established
    ``V'x + beta > 0`` on the relaxation (see
    :func:`denominator_stays_positive`).  The split is then skipped entirely
    and the sign row is not even added, which removes one row and one whole
    branch & bound run from every call.
    """
    phi = phi.lift(model.n)
    if denominator_positive:
        res = solve_milp(model, phi, max_nodes=max_nodes, cutoff=cutoff,
                         deadline=deadline, warm=warm)
        if res.x:
            res.x = res.x[:model.n]
        return res

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
        res = solve_milp(branch, objective, max_nodes=max_nodes, cutoff=best_so_far,
                         deadline=deadline)
        nodes += res.nodes
        if res.status == INTERRUPTED:
            # the bound of a branch not finished still bounds the whole problem
            bounds = [b for b in (res.bound, best.bound) if b is not None]
            out = MilpResult(INTERRUPTED, res.x[:model.n] if res.x else None,
                             res.objective, nodes=nodes,
                             bound=max(bounds) if bounds else None)
            return out
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
