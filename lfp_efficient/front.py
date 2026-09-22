"""The **complete** non-dominated set, enumerated in criterion space.

:func:`lfp_efficient.efficient_subset` delivers a certified *subset* and never
claims completeness.  This module delivers the whole front and proves it --
which is a different thing, and the honest answer to "give me the efficient
solutions" when the asker means *all* of them.

:func:`lfp_efficient.enumerate_nondominated` already did this, but in decision
space: it truncates ``D`` with Sylva-Crema cuts, so every cut adds ``p``
binaries and ``p+1`` big-M rows and each sub-problem is strictly larger than
the last.  That is the cost profile this whole package exists to avoid, and the
same substitution works here: keep what is left in a **list of boxes**, and the
truncation is a box subtraction with no binary anywhere and no model growth.

The loop
--------
Start from ``R^p``.  Pop a box, look for a feasible point of ``D`` inside it,
and if there is none drop the box.  Otherwise turn that point into an efficient
one, record its criterion vector, and split the box around it
(Proposition 1 of the note).  Stop when the list is empty.

Why it terminates.  The probe returns ``x`` in the box, and the efficient
point ``a`` it is repaired to dominates it, so ``Z(x) <= Z(a)``: the split
removes at least ``Z(x)`` from that box.  The unexplored region therefore
shrinks on every pass, and the non-dominated set of a bounded integer program
is finite.

Why it is complete.  A non-dominated vector ``v`` leaves the region only
through a split around a centre ``a`` with ``v <= Z(a)``.  If ``v != Z(a)``
that says ``a`` dominates ``v``, which no non-dominated vector admits.  So
``v`` can only leave at the moment it is recorded -- and since the region is
exhausted, every ``v`` is recorded exactly once.

Vectors are not solutions
-------------------------
``Phi`` is a function of ``x`` and **not** of ``Z(x)``, so one non-dominated
vector can carry several efficient points with different ``Phi``.  Enumerating
the front therefore does not by itself answer ``(P_E)``.  When *phi* is given,
each vector is paired with the point maximising ``Phi`` on its slice -- which
is the pairing a decision maker actually wants, and which the plain vector
enumeration of :mod:`lfp_efficient.enumeration` does not provide.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from heapq import heappop, heappush
from time import monotonic
from typing import List, Optional, Tuple

from .criterion_space import Box, split
from .efficiency import (best_with_same_criterion, efficient_dominator,
                         test_efficiency)
from .milp import denominator_stays_positive, solve_linear_milp
from .model import FractionalObjective, MOILFP
from .rational import F, ZERO
from .simplex import OPTIMAL


@dataclass
class Front:
    """The complete non-dominated set, and one efficient point for each vector."""

    vectors: List[List[Fraction]] = field(default_factory=list)
    points: List[List[Fraction]] = field(default_factory=list)
    #: ``Phi`` at each point, when a ``Phi`` was given; the point is then the
    #: best one on its slice, so this is the most a decision maker can get
    #: while standing on that vector of the front
    values: List[Fraction] = field(default_factory=list)
    #: boxes settled -- the unit of work, as in the optimisation search
    boxes: int = 0
    #: ``True`` when the list emptied, so the front is provably complete
    complete: bool = False

    def __len__(self) -> int:
        return len(self.vectors)

    def report(self) -> str:
        head = (f"{len(self)} non-dominated vectors"
                + (" -- the complete front, proved"
                   if self.complete else " -- INCOMPLETE (budget spent)"))
        lines = [head, f"  {self.boxes} boxes settled"]
        if self.values:
            best = max(range(len(self.values)), key=lambda i: self.values[i])
            lines.append(f"  best Phi on the front: {self.values[best]} at "
                         f"({', '.join(str(c) for c in self.points[best])})")
        return "\n".join(lines)


def enumerate_front(problem: MOILFP, phi: Optional[FractionalObjective] = None,
                    max_boxes: int = 200_000,
                    time_budget: Optional[float] = None) -> Front:
    """Enumerate the whole non-dominated set, in criterion space.

    With *phi*, each vector is paired with the point maximising ``Phi`` on its
    slice, at the cost of one extra integer program per vector.  Without it,
    the efficient point recorded is whichever the repair landed on.

    The result carries ``complete``: ``True`` only when the box list emptied,
    which is the proof.  A run stopped by *max_boxes* or *time_budget* returns
    what it has with ``complete = False`` rather than a front it cannot
    support.
    """
    deadline = None if time_budget is None else monotonic() + time_budget
    positive = (denominator_stays_positive(problem.model, phi)
                if phi is not None else False)

    # any strictly positive direction pulls a point out of the box; the sum of
    # the criteria numerators tends to land on an efficient one straight away,
    # which saves the repair below more often than not
    probe = [ZERO] * problem.n
    for z in problem.criteria:
        for j in range(problem.n):
            probe[j] += F(z.U[j])

    front = Front()
    counter = 0
    open_boxes: List[Tuple[int, Box]] = []
    heappush(open_boxes, (0, Box()))
    seen = set()

    while open_boxes:
        if front.boxes >= max_boxes:
            return front
        if deadline is not None and monotonic() > deadline:
            return front

        _, box = heappop(open_boxes)
        front.boxes += 1

        model = box.restricted(problem.model)
        direction = probe + [ZERO] * (model.n - problem.n)
        found = solve_linear_milp(model, direction)
        if found.status != OPTIMAL:
            continue                            # the box holds nothing: drop it

        x = found.x[:problem.n]
        outcome = test_efficiency(problem, x)
        centre = x if outcome.efficient else efficient_dominator(problem, outcome)

        key = tuple(problem.Z(centre))
        if key not in seen:
            seen.add(key)
            front.vectors.append(list(key))
            if phi is not None:
                best = best_with_same_criterion(problem.model, problem, centre,
                                                phi,
                                                denominator_positive=positive)
                if best.feasible:
                    front.points.append(list(best.x[:problem.n]))
                    front.values.append(best.objective)
                else:                           # Phi undefined on the slice
                    front.points.append(list(centre))
            else:
                front.points.append(list(centre))

        for child in split(problem, box, centre, None):
            counter += 1
            heappush(open_boxes, (counter, child))

    front.complete = True
    return front
