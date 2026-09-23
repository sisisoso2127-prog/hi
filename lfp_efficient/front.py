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

Why the probe maximises a direction instead of just finding a point
--------------------------------------------------------------------
The probe only needs *some* feasible point of the box, so maximising a linear
direction looks like paid-for waste: a zero objective would stop at the first
integer point found.  Measured, it is the opposite, and by a lot.

Profiling the decision-space enumeration on an instance that takes 137 s put
**100%** of the time in the probe, over 31 calls and 14983 branch & bound
nodes, with the model growing from 5 columns and 7 rows to 95 and 127 -- which
is exactly ``p`` binaries and ``p+1`` rows per cut, so that part is structural
and expected.  Replacing the direction by a zero objective:

    n=4   9 cuts,  658 nodes, 1.35s   ->  9 cuts, 1397 nodes,  5.33s
    n=5   9 cuts,  372 nodes, 0.79s   ->  9 cuts, 2105 nodes, 10.15s

The **cut counts are identical**: the probe's direction does not change how
many vectors there are, or the order they come out in.  What explodes is the
branch & bound inside each probe, because a zero objective gives it nothing to
prune with -- every node's relaxation is worth 0, so no bound can discard
anything until an integer point has been stumbled upon.

So the direction is load-bearing rather than decorative, and this module uses
the same one.  Recorded because it is a plausible-looking optimisation that
makes things four to thirteen times worse, and because it was checked while
suspecting the *comparison* was unfair -- it was not.

The front is not the set of efficient points
--------------------------------------------
One non-dominated vector can be attained by several efficient points, and this
module returns **one point per vector**.  So the front is complete while the
set of points it carries need not be.

On random instances with generic criterion coefficients the distinction never
shows up: over 32 such instances, 476 efficient points sat on 476 distinct
vectors and every slice was a singleton.  That is a property of the sample, not
of the problem -- generic coefficients make ties improbable.  Make a criterion
ignore a variable and the ties appear at once::

    D : x1,x2,x3 <= 2,  x1 + x2 <= 3
    Z = (x1, x2)                       # x3 does not enter Z
    Phi = (x1 + x2 + 3*x3) / (x1 + x2 + 1)

Here ``|E(P_D)| = 6`` sits on **2** vectors, and this module returns 2 points of
the 6.  The slice ``Z = (1,2)`` alone holds ``(1,2,0)``, ``(1,2,1)`` and
``(1,2,2)``, with ``Phi`` equal to ``3/4``, ``6/5`` and ``3/2``.

And that is exactly why ``Q`` is solved rather than skipped: it keeps the
**best** point of each slice.  On the instance above the repair lands on
``Phi = 3/4`` and ``Q`` moves it to ``3/2`` -- so the largest value on the front
is still the optimum of ``(P_E)``, which it would not be otherwise.  Skipping
``Q`` would have halved the answer while leaving the front itself correct, and
leaving the output looking entirely reasonable.

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
from typing import List, Optional, Sequence, Tuple

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
    #: one efficient point per vector -- the best on its slice when a ``Phi``
    #: was given.  **Not** every efficient point: a slice can hold several, and
    #: only one is kept.  See the module docstring for a worked case.
    points: List[List[Fraction]] = field(default_factory=list)
    #: ``Phi`` at each point, when a ``Phi`` was given; the point is then the
    #: best one on its slice, so this is the most a decision maker can get
    #: while standing on that vector of the front
    values: List[Fraction] = field(default_factory=list)
    #: boxes settled -- the unit of work, as in the optimisation search
    boxes: int = 0
    #: ``True`` when the list emptied, so the **front** is provably complete.
    #: That is completeness of the non-dominated *vectors*, not of the
    #: efficient *points* -- see the module docstring.
    complete: bool = False
    #: integer programs spent probing boxes -- the work a seed can remove
    probes: int = 0
    #: vectors recorded from the seeds, at no integer program at all
    seeded: int = 0

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


def in_box(problem: MOILFP, box: Box, x) -> bool:
    """Does *x* satisfy every row of *box*?

    A box is a conjunction of ``coeffs' x <= rhs`` rows on the model's own
    variables, so this is a scan and not a program.  It is what lets a seed be
    placed in the list without solving anything.
    """
    return all(sum(c * xi for c, xi in zip(coeffs, x[:problem.n])) <= rhs
               for coeffs, rhs in box.rows)


def pre_split(problem: MOILFP, seeds) -> Tuple[List[Box], List[List[Fraction]]]:
    """Split the root box around known efficient points, paying no programs.

    The split of Section~3 is *disjoint* and covers the box minus
    ``{ Z <= Z(centre) }``, so a point whose vector has not been removed lies
    in exactly one child.  Placing a seed is therefore a scan of the list for
    the box that holds it, followed by the same split the search would have
    done -- with the probe, the efficiency test and the repair all skipped,
    because the seed is already known to be efficient.

    A seed whose vector has been removed by an earlier one is dominated by it
    and is dropped: two seeds can be distinct points of the same slice, and a
    generator makes no promise that they are not.

    The seeds **must be efficient**.  Completeness survives an inefficient
    centre -- the argument only needs it to lie in ``D`` -- but the recorded
    vector would not be on the front, and the output would be wrong rather
    than merely slow.
    """
    boxes = [Box()]
    recorded: List[List[Fraction]] = []
    seen = set()
    for a in seeds:
        key = tuple(problem.Z(a))
        if key in seen:
            continue
        index = next((i for i, b in enumerate(boxes) if in_box(problem, b, a)),
                     None)
        if index is None:               # already removed: dominated by a seed
            continue
        seen.add(key)
        recorded.append([F(c) for c in a])
        box = boxes.pop(index)
        boxes.extend(split(problem, box, a, None))
    return boxes, recorded


def enumerate_front(problem: MOILFP, phi: Optional[FractionalObjective] = None,
                    max_boxes: int = 200_000,
                    time_budget: Optional[float] = None,
                    seeds: Optional[List[Sequence[Fraction]]] = None) -> Front:
    """Enumerate the whole non-dominated set, in criterion space.

    With *phi*, each vector is paired with the point maximising ``Phi`` on its
    slice, at the cost of one extra integer program per vector.  Without it,
    the efficient point recorded is whichever the repair landed on.

    The result carries ``complete``: ``True`` only when the box list emptied,
    which is the proof.  A run stopped by *max_boxes* or *time_budget* returns
    what it has with ``complete = False`` rather than a front it cannot
    support.

    *seeds* are efficient points already known -- from a generator, from an
    earlier run, from the decision maker.  Each is placed in the box list by
    :func:`pre_split`, which records its vector and splits around it **without
    solving anything**: no probe, no efficiency test, no repair.  The answer is
    unchanged; what changes is how much of it had to be found.  Seeds that are
    not efficient make the output wrong, not slow, so they must come from a
    source that guarantees it.
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
    seen = set()

    def record(centre):
        """Bank one vector and, when a ``Phi`` was given, the best point on it.

        The same for a seeded vector as for a found one: the ``Q`` program is
        what makes the largest value on the front the optimum of ``(P_E)``, so
        a seed must not skip it.  What a seed skips is the probe, the
        efficiency test and the repair -- never this.
        """
        key = tuple(problem.Z(centre))
        if key in seen:
            return
        seen.add(key)
        front.vectors.append(list(key))
        if phi is None:
            front.points.append(list(centre))
            return
        best = best_with_same_criterion(problem.model, problem, centre, phi,
                                        denominator_positive=positive)
        if best.feasible:
            front.points.append(list(best.x[:problem.n]))
            front.values.append(best.objective)
        else:                                   # Phi undefined on the slice
            front.points.append(list(centre))

    if seeds:
        boxes, recorded = pre_split(problem, seeds)
        for a in recorded:
            record(a)
            front.seeded += 1
        for box in boxes:
            counter += 1
            heappush(open_boxes, (counter, box))
    else:
        heappush(open_boxes, (0, Box()))

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
        front.probes += 1
        if found.status != OPTIMAL:
            continue                            # the box holds nothing: drop it

        x = found.x[:problem.n]
        outcome = test_efficiency(problem, x)
        centre = x if outcome.efficient else efficient_dominator(problem, outcome)

        record(centre)

        for child in split(problem, box, centre, None):
            counter += 1
            heappush(open_boxes, (counter, child))

    front.complete = True
    return front
