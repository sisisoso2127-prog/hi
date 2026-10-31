"""Queries over ``E(P_D)`` once it is explicit -- and what that widens.

The decision-space literature on optimising a function over an integer
efficient set is built around *one* decision function at a time, and around a
function whose structure the method can exploit.  A branch-and-cut for a linear
fractional ``Phi`` needs ``Phi`` linear fractional: the cuts, the bounds and the
Dinkelbach-style sub-problems are all derived from the ratio form.  Change
``Phi`` to a sum of two ratios and the method does not degrade, it does not
apply; ask for two decision functions at once and the literature answers with a
separate paper and a separate branch-and-bound.

The criterion-space route of this package reaches ``E(P_D)`` *itself*
(:mod:`lfp_efficient.complete`), and that changes what the decision function has
to be.  A finite, explicit, certified set admits **any computable** ``f``:

    the decision-space methods require structure in ``f`` and nothing of
    ``E(P_D)``; this route requires ``E(P_D)`` and nothing of ``f``.

That is a trade, not a free lunch, and the honest form of it is stated in
:func:`maximize_over_efficient_set`.  But it is a trade in a direction the
fractional literature has not taken, and it is what makes the combination worth
more than the sum of its two halves: criterion-space search was already known
for *linear* decision functions, and the fractional problem was already known in
*decision* space; neither alone gives the class below.

What this module adds over scanning ``result.points`` by hand is the
hypothesis.  Every function here refuses a :class:`~lfp_efficient.complete.CompleteSet`
that is not proved complete.  A budget-truncated set is still a set of genuine
efficient points, and a maximum over it is still a number -- just not the answer
to the question asked, and nothing downstream could tell the difference.  That
is the eleventh finding of the audit in a new place, so it is refused here
before it can be made.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, List, Optional, Sequence

from .complete import CompleteSet

__all__ = ["OverEfficientSet", "NondominatedOverEfficientSet",
           "maximize_over_efficient_set", "nondominated_over_efficient_set"]


def _assert_complete(result: CompleteSet, what: str) -> None:
    """Refuse a set that was never proved to be all of ``E(P_D)``."""
    if not isinstance(result, CompleteSet):
        raise TypeError(f"{what} needs a CompleteSet, got {type(result).__name__}")
    if not result.complete:
        raise ValueError(
            f"the efficient set handed in is not proved complete, so {what} "
            "would be taken over a subset of E(P_D) and would look exactly "
            "like the answer.  Re-run complete_efficient_set without a budget, "
            "or accept a bound rather than an answer")
    if not result.points:
        raise ValueError("the efficient set handed in is empty")


@dataclass
class OverEfficientSet:
    """The exact optimum of one function over ``E(P_D)``."""

    x: List[Fraction]
    value: object
    #: how many efficient points were ranked, and how many left ``f`` undefined
    scanned: int = 0
    undefined: int = 0
    #: every efficient point attaining the optimum, not merely the first
    argoptima: List[List[Fraction]] = field(default_factory=list)

    def __str__(self) -> str:
        pt = ", ".join(str(c) for c in self.x)
        tail = (f", {len(self.argoptima)} points attain it"
                if len(self.argoptima) > 1 else "")
        return (f"{self.value} at ({pt}) over {self.scanned} efficient points"
                f"{tail}")


def maximize_over_efficient_set(result: CompleteSet,
                                f: Callable[[Sequence[Fraction]], object],
                                *, minimize: bool = False) -> OverEfficientSet:
    """Exact optimum of **any** computable ``f`` over ``E(P_D)``.

    ``f`` is called on efficient points and nothing else is asked of it: no
    ratio form, no linearity, no monotonicity, no derivative.  It may be a sum
    of ratios, a maximum of ratios, a product, a piecewise definition, or a
    black box, and the answer is still exact, because the optimum of a function
    over a finite explicit set is decided by evaluation and not by structure.

    The contrast is the point of this module.  A decision-space branch-and-cut
    for a linear fractional objective derives its cuts and its bounds from the
    ratio form; given a sum of two ratios it does not lose efficiency, it stops
    applying.  Here that case costs exactly what the single-ratio case costs.

    The trade, stated plainly: that method never builds ``E(P_D)`` and so is
    unaffected by its size, while this one builds it and so is bounded by it.
    Neither dominates.  They fail on different instances -- on a structureless
    ``f``, and on an ``E(P_D)`` too large to enumerate -- and that is why the
    two are worth reporting side by side rather than ranked.

    A point where ``f`` raises :exc:`ZeroDivisionError` or
    :exc:`ArithmeticError` is not a candidate (``Phi`` is undefined there); such
    points are counted in ``undefined`` rather than passed over silently, since
    an ``f`` undefined on *every* efficient point is a modelling error and the
    caller should be able to see it.

    Ties are kept: ``argoptima`` holds every efficient point attaining the
    optimum, which the fractional literature's single-answer interface cannot
    report and which matters here because ``Phi`` is a function of ``x`` and not
    of ``Z(x)``.
    """
    _assert_complete(result, "this optimisation")

    best_x, best_v, ties, undefined = None, None, [], 0
    for x in result.points:
        try:
            v = f(x)
        except (ZeroDivisionError, ArithmeticError):
            undefined += 1
            continue
        if best_v is None:
            best_x, best_v, ties = list(x), v, [list(x)]
            continue
        better = (v < best_v) if minimize else (v > best_v)
        if better:
            best_x, best_v, ties = list(x), v, [list(x)]
        elif v == best_v:
            ties.append(list(x))

    if best_v is None:
        raise ValueError(
            f"the objective is undefined at all {undefined} efficient points")
    return OverEfficientSet(x=best_x, value=best_v, scanned=len(result.points),
                            undefined=undefined, argoptima=ties)


@dataclass
class NondominatedOverEfficientSet:
    """The non-dominated subset of ``E(P_D)`` for several decision functions."""

    points: List[List[Fraction]] = field(default_factory=list)
    values: List[List[object]] = field(default_factory=list)
    scanned: int = 0
    undefined: int = 0

    def __len__(self) -> int:
        return len(self.points)


def nondominated_over_efficient_set(
        result: CompleteSet,
        fs: Sequence[Callable[[Sequence[Fraction]], object]],
        *, minimize: Optional[Sequence[bool]] = None
) -> NondominatedOverEfficientSet:
    """Several decision functions over ``E(P_D)`` at once, exactly.

    With two linear fractional functions this is the problem of Chaiblaine,
    Moulaï and Cherfaoui, which in decision space needs its own branch and
    bound with cutting planes.  Here it is a dominance filter over a set already
    in hand, and the cost does not depend on how many functions are asked for or
    on what they are: ``m`` ratios, one ratio and one quadratic, or ten
    black boxes all cost the same scan.

    *minimize* gives a sense per function; the default maximises all of them.
    A point where any ``f`` is undefined is dropped and counted, because a
    partially defined vector cannot be compared.
    """
    _assert_complete(result, "this non-dominance filter")
    if not fs:
        raise ValueError("no decision functions were given")
    senses = list(minimize) if minimize is not None else [False] * len(fs)
    if len(senses) != len(fs):
        raise ValueError(f"{len(fs)} functions but {len(senses)} senses")

    pts, vals, undefined = [], [], 0
    for x in result.points:
        try:
            v = [f(x) for f in fs]
        except (ZeroDivisionError, ArithmeticError):
            undefined += 1
            continue
        pts.append(list(x))
        # orient every component so that "larger is better"
        vals.append([-v[k] if senses[k] else v[k] for k in range(len(fs))])

    keep_pts, keep_vals = [], []
    for i, vi in enumerate(vals):
        dominated = False
        for j, vj in enumerate(vals):
            if i == j:
                continue
            if (all(b >= a for a, b in zip(vi, vj))
                    and any(b > a for a, b in zip(vi, vj))):
                dominated = True
                break
        if not dominated:
            keep_pts.append(pts[i])
            # report the values the caller asked for, not the oriented ones
            keep_vals.append([-c if senses[k] else c
                              for k, c in enumerate(vi)])

    return NondominatedOverEfficientSet(points=keep_pts, values=keep_vals,
                                        scanned=len(result.points),
                                        undefined=undefined)
