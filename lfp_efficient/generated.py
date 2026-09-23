"""Seeding the **front enumeration** from a generator: the hybrid that fits.

Three hybridizations were measured before this one, and the seeding ones were
attached to the *optimisation* search.  There they feed one mechanism: the
bound.  A search that already knows a good value prunes boxes whose bound
cannot beat it, so a seed is worth what the bound can do with it -- and the
Tchebychev seed lost, because it steers in criterion space relative to the
ideal point and never looks at ``Phi``.

**The front enumeration has no bound and no incumbent.**  It must exhaust every
box, so nothing about it can be pruned by knowing a good value, and a seed of
the earlier kind is worth exactly nothing.  What it does have is a different
mechanism: every box costs one probe, one efficiency test and sometimes a
repair, all spent to find *a vector*.  A vector already known needs none of
them.  Seeding here is therefore not "start from a better value" but "do not
pay to discover what you already know".

Why this generator, and why it needs no verification
----------------------------------------------------
The augmented weighted Tchebychev program returns a point that is efficient
**by construction**, for any strictly positive weights and any positive
augmentation -- the two-line argument is in :mod:`lfp_efficient.tchebychev`.
That matters here more than anywhere else in this package.  The Pareto archive
of :mod:`lfp_efficient.metaheuristic` also produces efficient points, but only
after an explicit verification, and that verification *is* the cost of that
hybrid.  A generator whose output needs no checking is the one that can pay.

And what the Tchebychev program is good at is exactly what is wanted: a spread
of efficient points, unsupported ones included, which no positive weighted sum
reaches.  As a seed for ``max Phi`` that spread was beside the point.  For
enumerating the front the spread *is* the point.

What it cannot change
---------------------
The answer, and the number of vectors in it.  A seed removes the cost of
*finding* a vector, never the vector itself, so the front that comes out is the
same front and it is still proved complete the same way.  It cannot reduce the
box count below what the splits require either: each recorded vector splits its
box into ``p`` children whether it was found or given.

So the arithmetic of the trade is simple and worth stating before measuring
it: seeding with ``k`` vectors removes ``k`` probes and ``k`` efficiency tests
and adds ``p`` programs for the ideal point plus one per weight vector.  It
pays when a scalarisation is cheaper than a probe-and-test, and when the
weights actually land on distinct vectors of the front.

Measured: it loses, and the arithmetic says it had to
------------------------------------------------------
It does not pay.  Over 120 instances at ``n = 5..7`` and ``p = 3, 4``, against
the same enumeration without seeds:

===================  ============  ==============  ==========  =========
instance             weight grid   seeded          programs    time
===================  ============  ==============  ==========  =========
``n=5  p=3``         narrow        75% of front    1.06x       1.44x
``n=5  p=3``         wide          **100%**        1.48x       **2.61x**
``n=6  p=3``         narrow        43%             1.10x       1.18x
``n=6  p=3``         wide          75%             1.28x       1.84x
``n=6  p=4``         narrow        33%             1.11x       1.25x
``n=6  p=4``         wide          71%             1.21x       1.80x
``n=7  p=3``         narrow        38%             1.03x       1.16x
``n=7  p=3``         wide          64%             1.10x       1.55x
===================  ============  ==============  ==========  =========

Every answer identical, every front still proved complete -- and every ratio
above 1, which is the wrong direction.  The decisive row is the second: the
wide grid seeds **the entire front** and is the **worst** of the eight.  More
seeding is more loss, which is not what a useful seed does.

The reason is one measurement, and it is structural rather than an artefact.
Seeding ``k`` distinct vectors saves at most ``k`` probes; producing them costs
at least ``k`` scalarisations, plus ``p`` for the ideal point, plus one for
every weight vector that lands on a vector already seen.  And a scalarisation
is not cheaper than a probe:

    one probe          median   9.48 ms,   0 b&b nodes
    one scalarisation  median  20.90 ms,  21 b&b nodes
    ratio              median   2.51x, cheaper on only 7 of 40

The probe maximises a linear direction over ``D`` plus a few ordinary rows and
usually needs no branching at all.  The scalarisation minimises a *maximum*,
which costs a continuous variable and ``p`` extra rows, and it branches.  Even
crediting the seed with the efficiency test it also removes -- $7.68$ ms, so
$17.2$ ms saved against $20.9$ ms spent -- a single seed is already a losing
trade before the ideal point and the duplicate weights are counted.

And the saving is not even guaranteed.  Seeding was expected to remove exactly
one probe per seed, and in the median it does -- over 152 instances the median
saving is $+1.00$ probe per seed, which is the arithmetic above coming out
right.  But the pre-split imposes a box structure the search would not have
chosen, and on **5 of those 152** the seeded run spent *more* probes than the
plain one, up to four more: a box the plain run would have absorbed into a
larger one now exists separately and has to be probed to prove it empty.  So
the upper bound on the gain is one probe per seed, and even that is not a
floor.

So the hypothesis this module was written to test is refused: *the front is
where a generator should pay, because there the spread is the point*.  The
spread is indeed the point, and the generator does supply it -- it seeds 100%
of the front when asked.  It still loses, because what the enumeration needs is
not a better spread but a cheaper way to produce one, and a Tchebychev program
is a dearer way to find an efficient point than the probe the search already
has.

What is kept, and why
---------------------
The code stays, and ``seeds`` stays on
:func:`~lfp_efficient.front.enumerate_front`, for two reasons that do not
depend on this verdict.  Seeds from *any* source are handled soundly, so a
decision maker's known-good points, or a front from an earlier run on a
tightened model, cost nothing to reuse.  And the mechanism it exercises --
recording a vector and splitting around it without solving anything -- is the
one the exact--exact hybrid already uses in the other direction.

This is the fourth hybridization to lose, and the pattern across all four is
now the same sentence: a hybridization pays only where the exact method has a
mechanism the first phase can feed **more cheaply than the exact method feeds
it itself**.  The earlier three failed the first half of that test.  This one
passes it -- the mechanism is real and the seed reaches it -- and fails the
second.
"""
from dataclasses import dataclass, field
from fractions import Fraction
from time import monotonic
from typing import List, Optional, Sequence

from .efficiency import has_linear_criteria, spread_weights
from .front import Front, enumerate_front
from .model import FractionalObjective, MOILFP
from .tchebychev import augmented_tchebychev_efficient, ideal_point

F = Fraction

__all__ = ["GeneratedSeeds", "generate_seeds", "generated_front"]


@dataclass
class GeneratedSeeds:
    """Efficient points a weight sweep reached, and what they cost."""

    points: List[List[Fraction]] = field(default_factory=list)
    #: distinct criterion vectors among them -- what the seeding can actually use
    vectors: List[List[Fraction]] = field(default_factory=list)
    #: scalarisations solved, including the ``p`` for the ideal point
    programs: int = 0
    seconds: float = 0.0

    def __len__(self) -> int:
        return len(self.vectors)


def generate_seeds(problem: MOILFP,
                   weights: Optional[Sequence[Sequence[Fraction]]] = None,
                   rho: Fraction = F(1, 1000)) -> GeneratedSeeds:
    """Run the Tchebychev program over a spread of weights and keep the optima.

    Every point returned is efficient by the program's own guarantee, so none
    of them is verified here -- which is the whole economy of this hybrid.
    Duplicate criterion vectors are dropped: several weight vectors commonly
    land on the same efficient point, and a repeat is worth nothing to the
    enumeration.
    """
    if not has_linear_criteria(problem):
        raise ValueError(
            "the Tchebychev generator is for linear criteria; with ratios "
            "z*_k - Z_k(x) is not linear and the program is not an integer "
            "linear program")
    started = monotonic()
    result = GeneratedSeeds()
    ideal = ideal_point(problem)
    result.programs += problem.p                     # the ideal point
    seen = set()
    for w in (weights if weights is not None else spread_weights(problem.p)):
        point = augmented_tchebychev_efficient(problem, w, rho, ideal)
        result.programs += 1
        if point is None:
            continue
        key = tuple(problem.Z(point))
        if key in seen:
            continue
        seen.add(key)
        result.points.append(list(point))
        result.vectors.append(list(key))
    result.seconds = monotonic() - started
    return result


def generated_front(problem: MOILFP,
                    phi: Optional[FractionalObjective] = None,
                    weights: Optional[Sequence[Sequence[Fraction]]] = None,
                    rho: Fraction = F(1, 1000),
                    max_boxes: int = 200_000,
                    time_budget: Optional[float] = None):
    """Generate a spread of efficient points, then enumerate the front from it.

    Returns ``(front, seeds)``.  The front is the same front
    :func:`~lfp_efficient.front.enumerate_front` returns without seeds, proved
    complete by the same argument -- ``front.seeded`` says how much of it was
    handed over rather than discovered, and ``front.probes`` how many integer
    programs the search still had to spend.
    """
    seeds = generate_seeds(problem, weights, rho)
    remaining = (None if time_budget is None
                 else max(0.0, time_budget - seeds.seconds))
    front = enumerate_front(problem, phi, max_boxes=max_boxes,
                            time_budget=remaining, seeds=seeds.points)
    return front, seeds
