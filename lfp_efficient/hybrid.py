"""A hybrid: start where the paper starts, finish where the boxes are cheaper.

Neither search dominates the other, and the measurements say exactly where the
line falls.

* The paper's method settles an easy instance in two or three iterations, and
  on those it is the faster of the two -- the region has barely grown, and the
  criterion-space search pays for boxes it did not need.
* Past that point the region carries `p` binaries and `p+1` rows for every cut
  made, the sub-problems grow, and the box search wins by a margin that widens
  with the difficulty and with the number of criteria (measured, up to 103x at
  `p = 5`).

So: run the paper's loop, and if it has not closed after *switch_after*
iterations, hand the unfinished work to the box search.

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
                    switch_after: int = 3,
                    time_budget: Optional[float] = None,
                    verbose: bool = False) -> Solution:
    """Solve ``max { Phi(x) : x efficient for (P_D) }``, switching once.

    Parameters
    ----------
    switch_after
        Iterations of the paper's method to run before handing over.  ``0``
        makes this the pure box search, and a number past the longest run makes
        it the pure paper method, so the two are the ends of one dial.
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
