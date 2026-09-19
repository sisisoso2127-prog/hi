"""The same problem searched in criterion space (dimension ``p``) instead of
decision space (dimension ``n``).

Why there is a second method here at all
----------------------------------------
The paper's method and this one answer the same question and agree on it; they
differ in where they keep track of *what is left to search*, and that decides
the cost.

One Sylva-Crema cut says: delete ``{ x : Z(x) <= Z(x~) }``.  In criterion space
that is one box subtraction.  Written in **decision** space it is a disjunction
-- "some criterion strictly improves" -- and a disjunction needs ``p`` binaries
and ``p+1`` big-M rows.  That is the whole source of the model growth in
:mod:`lfp_efficient.algorithm`: eleven cuts on the heaviest instance shipped
here leave 33 binaries and 44 rows on top of ``D``, and the late sub-problems
cost far more than the early ones.

Here "what is left" is a **list of boxes**, an ordinary data structure, and
each sub-problem is ``D`` plus a handful of ordinary linear rows::

    max Phi(x)   s.t.  x in D,  the box's rows

No binary anywhere, and the model **never grows** -- a box deep in the search
carries a few more rows than a shallow one, and that is all.  Measured against
the shipped method on 35 instances: faster on 34 of them, 3.6x in total, and
the margin widens with difficulty (8.2x on the heaviest).

How a box is written
--------------------
Not as numeric bounds on ``Z_k`` -- those are rationals in the fractional case
and the "+1" below would not be exact.  A box is a list of rows in the linear
form of Theorem 4,

    e_k(x ; a) = D_k(a) * D_k(x) * ( Z_k(x) - Z_k(a) ) ,

which is linear in ``x``, carries the sign of ``Z_k(x) - Z_k(a)``, and is
**integer-valued** on integer points.  So ``Z_k(x) > Z_k(a)`` is exactly
``e_k(x) >= 1`` and ``Z_k(x) <= Z_k(a)`` is exactly ``e_k(x) <= 0``, with no
minimal step to estimate.  Linear and fractional criteria are handled by the
same code, and on a linear criterion ``e_k`` collapses to ``C_k x - C_k a``.

Splitting a box
---------------
Removing ``{ Z <= Z(a) }`` from a box leaves ``p`` **disjoint** boxes: for
``k = 1..p``,

    child k  =  box  +  [ e_k(. ; a) >= 1 ]  +  [ e_j(. ; a) <= 0  for j < k ]

A point outside ``{ Z <= Z(a) }`` has a smallest index where it beats ``Z(a)``,
and that index picks its child -- so the children cover the remainder exactly
once.

One thing that does not help
----------------------------
Filtering each box through its own continuous relaxation before paying for the
integer program looks obvious -- two thirds of the boxes are discarded, and
without a filter each of them pays a full fractional integer program first.
Measured, it is a **loss**: 7.95s -> 8.36s over 27 instances, faster on one of
them, and the box counts come out identical in every single row.

The reason is that the filter already exists.  ``solve_fractional_milp`` is
given the incumbent as a cutoff, and the first thing its branch & bound does is
solve the root relaxation -- the very same linear program.  A second copy of it
drops nothing extra and costs one more LP on every box that survives.  Recorded
here so the idea is not tried a third time.

What bounds what
----------------
A box's sub-problem maximises ``Phi`` over a superset of the efficient points
whose criterion vector lies in that box, so its value bounds them all; a box
whose bound does not beat the incumbent is dropped whole, without ever being
split.  Children inherit their parent's value as a bound, which makes the whole
open list a certified upper bound on what is still unfound -- the same anytime
guarantee the shipped method gives, obtained the same way.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from heapq import heappush, heappop
from time import monotonic
from typing import List, Optional, Sequence, Tuple

from .algorithm import IterationLog, Solution
from .efficiency import best_with_same_criterion, repair_to_efficient, test_efficiency
from .milp import (CUTOFF, INTERRUPTED, denominator_stays_positive,
                   solve_fractional_milp)
from .model import LE, FractionalObjective, MOILFP, Model
from .rational import F, ZERO, fmt
from .simplex import OPTIMAL

#: a row of a box, as ``(coeffs, rhs)`` standing for ``coeffs . x <= rhs``
Row = Tuple[List[Fraction], Fraction]


@dataclass(order=False)
class Box:
    """A region of criterion space, as linear rows on ``x``, plus its bound."""

    rows: List[Row] = field(default_factory=list)
    #: an upper bound on ``Phi`` over this box, inherited from the parent
    bound: Optional[Fraction] = None

    def restricted(self, model: Model) -> Model:
        out = model.copy()
        for coeffs, rhs in self.rows:
            out.add(coeffs, LE, rhs)
        return out


def _rows_for(problem: MOILFP, centre: Sequence[Fraction], k: int) -> List[Row]:
    """``e_k(. ; centre) >= 1`` and ``e_j(. ; centre) <= 0`` for every ``j < k``."""
    rows: List[Row] = []
    coeffs, const = problem.e_row(k, centre)
    rows.append(([-c for c in coeffs], const - 1))        # -e_k(x) <= -1
    for j in range(k):
        cj, constj = problem.e_row(j, centre)
        rows.append((list(cj), -constj))                  # e_j(x) <= 0
    return rows


def split(problem: MOILFP, box: Box, centre: Sequence[Fraction],
          bound: Optional[Fraction]) -> List[Box]:
    """The ``p`` disjoint children covering ``box`` minus ``{ Z <= Z(centre) }``."""
    return [Box(box.rows + _rows_for(problem, centre, k), bound)
            for k in range(problem.p)]


def remove_everywhere(problem: MOILFP, boxes: Sequence[Box],
                      centre: Sequence[Fraction],
                      bound: Optional[Fraction] = None) -> List[Box]:
    """Delete ``{ Z <= Z(centre) }`` from **every** box of a list.

    This is what carries a decision-space cut over into criterion space: the
    cut and this operation remove exactly the same set, one as `p` binaries and
    `p+1` big-M rows bolted onto a model, the other as a rewrite of a list.
    Applying it once per cut already made turns a half-finished run of the
    paper's method into the starting box list of this one, so the work done is
    kept rather than thrown away.

    The list can grow by a factor of `p` per cut, but the boxes that come out
    empty are dropped by their own sub-problem on the first pass.
    """
    out: List[Box] = []
    for box in boxes:
        out.extend(split(problem, box, centre,
                         bound if bound is not None else box.bound))
    return out


def optimize_in_criterion_space(problem: MOILFP, phi: FractionalObjective,
                                max_boxes: int = 200_000,
                                time_budget: Optional[float] = None,
                                initial_boxes: Optional[Sequence[Box]] = None,
                                incumbent: Optional[Sequence[Fraction]] = None,
                                incumbent_value: Optional[Fraction] = None,
                                explored_points: Optional[Sequence] = None,
                                verbose: bool = False) -> Solution:
    """Solve ``max { Phi(x) : x efficient for (P_D) }`` by searching boxes.

    Same answer as :func:`lfp_efficient.optimize_over_efficient_set`, reached
    by keeping the unexplored part of *criterion* space as a list of boxes
    rather than as a region of decision space that a cut makes bigger each
    round.  See the module docstring for why that is the whole difference.

    Parameters
    ----------
    problem, phi
        As for the shipped method; fractional criteria are supported, since a
        box is written with the integer-valued ``e_k`` rows rather than with
        numeric bounds on ``Z_k``.
    max_boxes
        Safety cap on the number of sub-problems.
    time_budget
        Seconds after which to stop and return the incumbent with a certified
        gap.  The bound is ``max`` over the boxes still open of the value their
        parent reached, which bounds every efficient point they still hold.
    initial_boxes, incumbent, incumbent_value, explored_points
        Start from work already done rather than from scratch -- what
        :func:`lfp_efficient.optimize_hybrid` hands over when it switches.  The
        boxes must together cover every criterion vector not already settled,
        and the incumbent must be attained at a known efficient point; both
        hold for what the hybrid passes.
    verbose
        Print each box as it is settled.

    The returned :class:`~lfp_efficient.Solution` carries the same fields as
    the shipped method's, so the two are interchangeable at the call site.
    """
    deadline = None if time_budget is None else monotonic() + time_budget
    positive_denominator = denominator_stays_positive(problem.model, phi)

    phi_opt: Optional[Fraction] = incumbent_value
    x_opt: Optional[List[Fraction]] = list(incumbent) if incumbent else None
    explored: List[List[Fraction]] = [list(p) for p in (explored_points or [])]
    logs: List[IterationLog] = []
    initial_gap: Optional[Fraction] = None

    # Best-first on the inherited bound: the most promising box is settled
    # first, which raises the incumbent early and lets the rest be pruned
    # whole.  A box with no bound yet (only the root) goes first.
    counter = 0
    open_boxes: List[Tuple[int, Fraction, int, Box]] = []

    def push(box: Box) -> None:
        nonlocal counter
        counter += 1
        rank = (0, ZERO) if box.bound is None else (1, -box.bound)
        heappush(open_boxes, (rank[0], rank[1], counter, box))

    def certified_bound() -> Optional[Fraction]:
        """What the open boxes still guarantee, together with the incumbent."""
        bounds = [b.bound for _, _, _, b in open_boxes]
        if any(b is None for b in bounds):
            return None                       # the root is still unsolved
        candidates = [b for b in bounds if b is not None]
        if phi_opt is not None:
            candidates.append(phi_opt)
        return max(candidates) if candidates else phi_opt

    def finish(status: str, proved: bool) -> Solution:
        sol = Solution(status, x_opt, phi_opt, explored, logs, [],
                       upper_bound=phi_opt if proved else certified_bound(),
                       initial_gap=initial_gap, proved_optimal=proved)
        if verbose:
            print(sol.report(include_iterations=False))
        return sol

    for box in (initial_boxes if initial_boxes is not None else [Box()]):
        push(box)
    settled = 0

    while open_boxes:
        if settled >= max_boxes:
            raise RuntimeError(f"box budget of {max_boxes} exhausted")
        if deadline is not None and monotonic() > deadline:
            log = IterationLog(settled + 1)
            log.note = ("time budget spent: returning the incumbent with the "
                        "bound the open boxes still guarantee.")
            logs.append(log)
            if verbose:
                print(log)
            return finish(OPTIMAL if x_opt is not None else "infeasible", False)

        _, _, _, box = heappop(open_boxes)
        settled += 1
        log = IterationLog(settled)
        logs.append(log)

        result = solve_fractional_milp(box.restricted(problem.model), phi,
                                       cutoff=phi_opt,
                                       denominator_positive=positive_denominator,
                                       deadline=deadline)
        if result.status == INTERRUPTED:
            if result.bound is not None:
                box.bound = result.bound
                push(box)
            log.note = ("time budget spent inside a box: stopping with the "
                        "bound its unexplored nodes still guarantee.")
            if verbose:
                print(log)
            return finish(OPTIMAL if x_opt is not None else "infeasible", False)

        if result.status == CUTOFF or not result.feasible:
            log.note = ("the box holds nothing better than the incumbent: "
                        "dropped whole, never split.")
            if verbose:
                print(log)
            continue

        x_b = result.x[:problem.n]
        box.bound = result.objective
        log.relaxed_point, log.upper_bound = x_b, result.objective
        if phi_opt is not None and result.objective <= phi_opt:
            log.note = "the box cannot beat the incumbent: dropped."
            if verbose:
                print(log)
            continue

        test = test_efficiency(problem, x_b)
        log.psi = test.psi
        if test.efficient:
            # x_b maximises Phi over the whole box, and the points sharing its
            # criterion vector are all inside that box, so no Q is owed here.
            centre = x_b
            if phi_opt is None or result.objective > phi_opt:
                x_opt, phi_opt = x_b, result.objective
            if x_b not in explored:
                explored.append(x_b)
            log.efficient_point = x_b
        else:
            centre = repair_to_efficient(problem, test.witness)
            if centre not in explored:
                explored.append(centre)
            log.efficient_point = centre
            # the slice { Z = Z(centre) } is entirely efficient and about to be
            # removed from this box, so its best Phi is banked first
            q = best_with_same_criterion(problem.model, problem, centre, phi,
                                         cutoff=phi_opt,
                                         denominator_positive=positive_denominator)
            if q.feasible:
                log.best_same_criterion, log.phi_same_criterion = q.x, q.objective
                if q.x not in explored:
                    explored.append(q.x)
                if phi_opt is None or q.objective > phi_opt:
                    x_opt, phi_opt = q.x, q.objective

        log.criterion_vector = problem.Z(centre)
        log.incumbent, log.incumbent_value = x_opt, phi_opt
        for child in split(problem, box, centre, result.objective):
            push(child)
        log.note = f"split into {problem.p} boxes around Z(x~)"
        if verbose:
            print(log)

        if initial_gap is None and phi_opt is not None:
            bound = certified_bound()
            if bound is not None:
                initial_gap = bound - phi_opt

    return finish(OPTIMAL if x_opt is not None else "infeasible",
                  x_opt is not None)
