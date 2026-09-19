"""A hybrid: start where the paper starts, finish where the boxes are cheaper.

Neither search dominates the other, and the measurements say exactly where the
line falls.

* When the maximiser of ``Phi`` over ``D`` happens to be efficient, the paper's
  method settles everything in **one** iteration, and no box search beats that.
* Past that point the region carries `p` binaries and `p+1` rows for every cut
  made, the sub-problems grow, and the box search wins by a margin that widens
  with the difficulty and with the number of criteria (measured, up to 103x at
  `p = 5`).

So: run the paper's loop, and if it has not closed after *switch_after*
iterations, hand the unfinished work to the box search.

What this is and is not worth
-----------------------------
Measured, because it decides how the option should be read.  On 24 instances,
against the better of the two pure methods on each:

    switch_after = 3     13.14s   within 15% of the better on  4 of 24
    switch_after = 1      8.47s   within 15% of the better on 23 of 24
    pure box search       8.11s
    the paper's method  268.61s

Choosing 3 without measuring was simply wrong: it pays three expensive
iterations where one suffices.  At 1 the hybrid **ties** the pure box search --
8.47s against 8.11s here, and 1.47s against 1.54s on a family built so that
``Phi`` is easy and the paper's method closes in one iteration.  Two
differences of about 4%, pointing opposite ways: noise.

So the honest reading is that the criterion-space search is what matters, and
the hybrid at ``switch_after = 1`` neither helps nor hurts beside it.  It earns
its place as insurance for the one-iteration case, not as a third method that
beats both.  The paper's method alone is 33x slower over the same set.

What "hand over" means, and why it is not a restart
---------------------------------------------------
Two things carry across, and together they make the switch nearly free.

**The cuts.** A Sylva-Crema cut on `x~` removes ``{ x : Z(x) <= Z(x~) }``.
Deleting that same set from a list of boxes is
:func:`lfp_efficient.criterion_space.remove_everywhere`, so applying it once
per cut already made reproduces, as a box list, exactly the region the first
phase had reached.  Nothing that was proved is proved again.

**The incumbent.** ``Phi_opt`` is attained at a known efficient point, so it
is a valid lower bound whatever search finds the rest; passed as the starting
incumbent it prunes boxes from the very first one.

The certified upper bound is the one thing that does *not* carry across, and
deliberately: the second phase computes its own from the boxes it holds, which
is the honest bound for the region it is actually searching.

Why the switch cannot break the proof
-------------------------------------
Both phases certify the same way -- a lower bound attained at an efficient
point, and an upper bound over everything not yet settled.  The handover
preserves both: the boxes cover every criterion vector the first phase had not
disposed of, and the incumbent is attained.  So the answer the hybrid proves
optimal is optimal for the same reason either method's would be, and
``tests/`` checks it against both of them and against the independent scan.
"""

from fractions import Fraction
from time import monotonic
from typing import List, Optional

from .algorithm import Solution, optimize_over_efficient_set
from .criterion_space import Box, optimize_in_criterion_space, remove_everywhere
from .model import FractionalObjective, MOILFP


def optimize_hybrid(problem: MOILFP, phi: FractionalObjective,
                    switch_after: int = 1,
                    time_budget: Optional[float] = None,
                    verbose: bool = False) -> Solution:
    """Solve ``max { Phi(x) : x efficient for (P_D) }``, switching once.

    Parameters
    ----------
    switch_after
        Iterations of the paper's method to run before handing over.  ``0``
        makes this the pure box search, and a number past the longest run makes
        it the pure paper method, so the two are the ends of one dial.  The
        default is **1**, measured: one iteration wins the case where the
        maximiser of ``Phi`` is already efficient, and its cut and incumbent
        are inherited by the second phase rather than wasted.  Three was the
        first default and is measurably worse -- see the module docstring.
    time_budget
        Shared across both phases; whatever is left when the first ends is what
        the second gets.  The answer stays certified either way.
    verbose
        Print both phases' traces, and the handover between them.

    Returns the usual :class:`~lfp_efficient.Solution`.  Its ``iterations``
    hold the first phase's logs followed by the second's, so the trace reads
    straight through.
    """
    if switch_after < 0:
        raise ValueError("switch_after must be >= 0")
    deadline = None if time_budget is None else monotonic() + time_budget

    if switch_after == 0:
        return optimize_in_criterion_space(problem, phi, verbose=verbose,
                                           time_budget=time_budget)

    first = optimize_over_efficient_set(problem, phi, stop_after=switch_after,
                                        time_budget=time_budget, verbose=verbose)
    if first.proved_optimal or first.value is None:
        return first                      # the easy case: it closed on its own

    remaining = None if deadline is None else max(deadline - monotonic(), 0.0)
    if remaining is not None and remaining <= 0:
        return first                      # the budget went on the first phase

    # ---- the handover -----------------------------------------------------
    # Replay each cut as a box-list rewrite.  The centres are in the trace:
    # every iteration that cut recorded the efficient point it cut around.
    boxes: List[Box] = [Box()]
    centres = [it.efficient_point for it in first.iterations
               if it.efficient_point is not None and it.note.startswith("cut off")]
    for centre in centres:
        boxes = remove_everywhere(problem, boxes, centre)
    if verbose:
        print(f"--- handover " + "-" * 40)
        print(f"  {len(centres)} cut(s) replayed as {len(boxes)} box(es); "
              f"incumbent Phi_opt = {first.value} carried over")

    second = optimize_in_criterion_space(
        problem, phi, time_budget=remaining, initial_boxes=boxes,
        incumbent=first.x, incumbent_value=first.value,
        explored_points=first.explored, verbose=verbose)

    second.iterations = first.iterations + second.iterations
    return second
