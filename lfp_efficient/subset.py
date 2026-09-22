"""A **subset** of the efficient set, delivered rather than left as a by-product.

The rest of this package answers one question -- which efficient point maximises
``Phi`` -- and proves the answer.  This module answers the other one a decision
maker usually has: *show me efficient solutions, with their ``Phi``, so I can
look at them.*  That is the stated expected result of the work this package
implements, and it is the one thing the machinery could already produce but
never exported.

Nothing here is new computation.  Two sources already exist:

**The exact search's own trail.**  To cut on a point, the box search must first
prove it efficient, so every point it cuts on is certified -- and by the end it
has accumulated a set of them for free, as a by-product of proving optimality.
Measured over 25 instances: 51% of ``E(P_D)``, at no cost beyond the search.

**The Pareto archive.**  The local search of :mod:`lfp_efficient.metaheuristic`
walks the efficient set guided by ``Phi`` and keeps a mutually non-dominated
archive.  Measured on the same instances: 98--100% of ``E(P_D)``, and the walk
itself costs hundredths of a second.

What is *not* here, and why
---------------------------
The augmented Tchebychev generator is not a third source, although it is
exported and certified by construction.  Measured: it reaches 39% of
``E(P_D)`` on its own and adds **nothing at all** on top of the archive --
every point it found was already there.  Adding it would cost 0.60 s against
the archive's 0.07 s and deliver no extra solution, so it is left out of the
default and remains available for anyone who wants it deliberately.

Certification is the cost
-------------------------
The archive filters by dominance among the points it has *seen*, which is not
the exact test of Theorem 1: over a wide sweep about one member in 2000 turns
out not to be efficient.  Certifying the whole archive is therefore one integer
program per member, and that -- not the walk -- is where the time goes: 0.25 s
against 0.03 s at ``n = 4``, 0.62 s against 0.11 s at ``n = 5``.  It is on by
default, because an *uncertified* "efficient set" is a claim and not a result;
``certify=False`` is offered for anyone who wants the archive cheaply and knows
what they are holding.

What this is not
----------------
**It is not the whole efficient set, and completeness is never claimed.**  The
coverage figures above are measured against exhaustive enumeration on instances
small enough to enumerate; on instances where ``E(P_D)`` cannot be enumerated
there is no way to know what fraction was found, and this module does not
pretend otherwise.

The trail of the exact search is also a **biased** sample rather than a
representative one: it is exactly the points the search had to cut on, so it
concentrates where ``Phi`` is large.  For a decision maker ranking solutions by
``Phi`` that bias is the useful direction, but it is a bias and is reported as
one.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

from .criterion_space import optimize_in_criterion_space
from .efficiency import test_efficiency
from .metaheuristic import metaheuristic_incumbent, pareto_local_search
from .model import FractionalObjective, MOILFP


@dataclass
class EfficientSubset:
    """Efficient solutions and their ``Phi``, best first.

    ``points[i]`` and ``values[i]`` go together.  ``optimum`` is the proved
    answer to ``(P_E)`` when the exact search ran, and is then also the first
    entry of ``points``.
    """

    points: List[List[Fraction]] = field(default_factory=list)
    values: List[Fraction] = field(default_factory=list)
    optimum: Optional[Tuple[List[Fraction], Fraction]] = None
    proved_optimal: bool = False
    #: every member passed the exact efficiency test of Theorem 1
    certified: bool = True
    #: how many members each source was the **first** to find.  A point
    #: both sources reach is attributed to whichever ran first, so this
    #: is an attribution of novelty, not a count of what each can reach
    sources: Dict[str, int] = field(default_factory=dict)
    #: archive members dropped because the exact test rejected them
    rejected: int = 0

    def __len__(self) -> int:
        return len(self.points)

    def best(self, k: int = 1) -> List[Tuple[List[Fraction], Fraction]]:
        """The *k* members with the largest ``Phi``."""
        return list(zip(self.points, self.values))[:k]

    def report(self) -> str:
        lines = [f"{len(self)} efficient solutions"
                 + (" (every one certified)" if self.certified
                    else " (NOT certified: archive membership only)")]
        if self.optimum is not None:
            x, v = self.optimum
            proof = "proved optimal" if self.proved_optimal else "best found"
            lines.append(f"  Phi* = {v} at ({', '.join(str(c) for c in x)})"
                         f"  [{proof}]")
        for name, count in self.sources.items():
            lines.append(f"  {count:4d} from {name}")
        if self.rejected:
            lines.append(f"  {self.rejected} archive member(s) rejected by the "
                         f"exact test")
        lines.append("  this is a SUBSET of E(P_D); completeness is not claimed")
        return "\n".join(lines)


def efficient_subset(problem: MOILFP, phi: FractionalObjective, *,
                     exact: bool = True, metaheuristic: bool = True,
                     certify: bool = True, seeds: int = 8, budget: int = 4000,
                     seed: int = 0,
                     time_budget: Optional[float] = None) -> EfficientSubset:
    """Generate a certified subset of ``E(P_D)``, with ``Phi`` on each member.

    Parameters
    ----------
    exact
        Run the criterion-space search.  Costs a full solve and returns the
        proved optimum along with the points it cut on.  With *metaheuristic*
        also set, the search is seeded from the archive, so the two share work
        rather than duplicating it.
    metaheuristic
        Run the Pareto local search and take its archive.  This is what makes
        the subset wide; on its own it is nearly free.
    certify
        Put every archive member through the exact test and keep only those
        that pass.  One integer program each -- the dominant cost -- and the
        difference between a result and a claim.  The exact search's own points
        are certified whatever this is set to, since it proved them to cut
        on them.
    """
    if not (exact or metaheuristic):
        raise ValueError("ask for at least one source: exact, metaheuristic, "
                         "or both")

    found: List[List[Fraction]] = []
    seen = set()
    sources: Dict[str, int] = {}
    rejected = 0

    def add(point, source):
        key = tuple(point)
        if key in seen:
            return
        seen.add(key)
        found.append(list(point))
        sources[source] = sources.get(source, 0) + 1

    incumbent = None
    if metaheuristic:
        archive = pareto_local_search(problem, phi, seeds, budget, seed)
        for point in archive.points():
            if certify and not test_efficiency(problem, list(point)).efficient:
                rejected += 1
                continue
            add(point, "the Pareto archive")
        if exact:
            incumbent = metaheuristic_incumbent(problem, phi, seeds, budget,
                                                seed=seed)

    solution = None
    if exact:
        if incumbent is not None:
            solution = optimize_in_criterion_space(
                problem, phi, time_budget=time_budget, incumbent=incumbent[0],
                incumbent_value=incumbent[1], explored_points=[incumbent[0]])
        else:
            solution = optimize_in_criterion_space(problem, phi,
                                                   time_budget=time_budget)
        for point in solution.explored:
            add(point, "the exact search")

    scored = []
    for point in found:
        try:
            scored.append((phi(point), point))
        except ZeroDivisionError:          # Phi undefined there; not a solution
            continue
    scored.sort(key=lambda pair: pair[0], reverse=True)

    out = EfficientSubset(
        points=[point for _, point in scored],
        values=[value for value, _ in scored],
        certified=certify or not metaheuristic,
        sources=sources,
        rejected=rejected)
    if solution is not None and solution.x is not None:
        out.optimum = (list(solution.x), solution.value)
        out.proved_optimal = solution.proved_optimal
    return out
