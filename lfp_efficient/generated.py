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

from .complete import CompleteSet, complete_efficient_set
from .efficiency import (efficient_dominator, has_linear_criteria,
                         spread_weights, test_efficiency)
from .metaheuristic import pareto_local_search
from .front import Front, enumerate_front
from .rational import ZERO
from .model import FractionalObjective, MOILFP
from .tchebychev import augmented_tchebychev_efficient, ideal_point

F = Fraction

__all__ = ["GeneratedSeeds", "generate_seeds", "generated_front",
           "hybrid_complete_set", "pareto_front", "pareto_seeds",
           "probe_order"]


@dataclass
class GeneratedSeeds:
    """Efficient points a weight sweep reached, and what they cost."""

    points: List[List[Fraction]] = field(default_factory=list)
    #: distinct criterion vectors among them -- what the seeding can actually use
    vectors: List[List[Fraction]] = field(default_factory=list)
    #: scalarisations solved, including the ``p`` for the ideal point
    programs: int = 0
    seconds: float = 0.0
    #: walk restarts spent, when the walk sized itself
    rounds: int = 1

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


# --------------------------------------------------------------------------
# the one that pays: a Pareto archive, ordered the way the search would go
# --------------------------------------------------------------------------
def probe_order(problem: MOILFP, points):
    """Sort efficient points the way the enumeration's own probe would meet them.

    The probe maximises the sum of the criteria numerators, so the search
    splits first around whatever scores highest on that direction.  Seeding in
    a different order builds a box structure the search would not have chosen,
    and that structure can cost more probes than the seeds save: in archive
    order the probe count went *up* on 4 of 12 instances.  In this order it
    went up on none, and the median fell from $0.89\times$ to $0.70\times$.
    Sorting the other way gives $0.94\times$ and is worse on 5 of 12, which is
    what makes the direction causal rather than a coincidence.
    """
    direction = [ZERO] * problem.n
    for z in problem.criteria:
        for j in range(problem.n):
            direction[j] += F(z.U[j])
    return sorted(points,
                  key=lambda a: sum(c * ai for c, ai in zip(direction, a)),
                  reverse=True)


def pareto_seeds(problem: MOILFP, phi: FractionalObjective,
                 seeds: int = 8, budget: int = 4000,
                 seed: int = 0, adaptive: bool = False,
                 patience: int = 0, max_rounds: int = 8) -> GeneratedSeeds:
    """Efficient points from a Pareto local search, verified and ordered.

    Unlike the Tchebychev program, a local search gives no guarantee, so every
    archive member is put through the exact efficiency test and replaced by a
    dominator when it fails.  That verification looks like the cost that sank
    the earlier seeding hybrids -- and here it is not, because **the
    enumeration was going to pay the same test anyway**: every probe it skips
    was going to be followed by one.  What a seed really costs is its share of
    the walk.

    Why the walk restarts itself
    ----------------------------
    A fixed budget is a fixed amount of work against a target that grows.
    Measured across instance sizes, that is exactly how the hybrid fails: what
    collapses is not the exact phase but **the share of the front the walk
    finds**, from 87% at ``n = 6`` down to 35% at ``n = 12`` and 12% at
    ``n = 12, p = 5`` -- and the speed-up tracks it down to $0.97\times$.  The
    probe count per vector stays flat throughout (2.4--5.2), so the cause is
    the walk and not the search.

The fix that works is one line: **the first round is sized to the
    instance**, ``seeds * max(1, n//4)`` restarts.  Restarts are what the
    measurement pointed at -- at ``n = 12``, 8 to 16 of them took coverage
    from 35% to 95% while the walk's own time barely moved.  Over 120
    instances at ``n = 6 \dots 12`` that gives a median $1.36\times$, faster
    on 116 of 120.

    Chasing coverage further does not work, which is the useful part
    ----------------------------------------------------------------
    With *adaptive*, the walk restarts in rounds that **double** in size until
    one contributes no criterion vector the earlier rounds had not.  It does
    what it says: coverage at ``n = 12, p = 5`` goes from 45% to 97%.  And it
    is **slower** --- $1.02\times$ against the single scaled round's
    $1.14\times$ on those instances, and a median $1.32\times$ against
    $1.36\times$ over all 120, winning 109 of them against 116.

    So coverage is not the objective, and this is the second time that
    mattered: the balance between what the walk costs and what its seeds save
    is.  At ``p = 5`` the walk is dear, and buying the last half of the front
    costs more than the probes it removes --- visible directly in a widening
    sweep, where 46% coverage gives $1.20\times$ and 93% gives $1.12\times$.
    ``adaptive`` is therefore off by default and kept for the record.  It then stops on its own where a
    constant cannot: the walk sizes itself to the front rather than to the
    number 4000.  Restarts matter more than neighbours -- at ``n = 12`` raising
    the restarts from 8 to 16 took coverage from 35% to 95% while the walk's
    own time barely moved.
    """
    started = monotonic()
    result = GeneratedSeeds()
    seen = set()
    quiet = 0
    rounds = max_rounds if adaptive else 1

    for r in range(rounds):
        # Doubling, not repetition.  Identical rounds re-walk the same
        # neighbourhoods: seven of them reached 86% at n=12 and cost more than
        # the exact search they were meant to shorten.  Doubling pays at most
        # twice the cost of the round that was the right size.
        # The first round is already sized to the instance.  Doubling from 8
        # needs four rounds to reach the scale n=12 wants, and the rounds
        # below it are wasted: measured, 4.94s of walk where 2.0s covers the
        # front.  Restarts are what matter, so they scale with n.
        scale = 1 << r
        start_seeds = seeds * max(1, problem.n // 4)
        start_budget = budget * max(1, problem.n // 4)
        archive = pareto_local_search(problem, phi, seeds=start_seeds * scale,
                                      budget=start_budget * scale,
                                      seed=seed + r)
        gained = 0
        for x in archive.points():
            key = tuple(problem.Z(x))
            if key in seen:
                continue            # cheap pre-filter: no test on a repeat
            outcome = test_efficiency(problem, x)
            point = (list(x) if outcome.efficient
                     else efficient_dominator(problem, outcome))
            result.programs += 1
            key = tuple(problem.Z(point))
            if key in seen:
                continue
            seen.add(key)
            gained += 1
            result.points.append(list(point))
            result.vectors.append(list(key))
        result.rounds = r + 1
        if not adaptive:
            break
        # Stop on DIMINISHING returns, not on zero.  Rounds double, so the
        # round after the last productive one is the most expensive of all;
        # waiting for it to come back empty pays for the whole search twice.
        # A round that adds under a tenth of what is already held is the
        # signal, and it arrives one doubling earlier.
        threshold = max(1, len(seen) // 10)
        quiet = quiet + 1 if gained <= threshold else 0
        if quiet > patience:
            break

    result.points = probe_order(problem, result.points)
    result.seconds = monotonic() - started
    return result


def pareto_front(problem: MOILFP, phi: Optional[FractionalObjective] = None,
                 max_boxes: int = 200_000,
                 time_budget: Optional[float] = None,
                 **walk):
    """Walk first, then enumerate the front from what the walk found.

    Returns ``(front, seeds)``.  This is the hybridization that pays: over 12
    instances at ``n = 6..10`` the front comes out **1.34x** faster in the
    median (min $0.77$, max $1.67$), with the probe count down from $0.89$ to
    $0.70$ of the unseeded search once the seeds are put in
    :func:`probe_order`.

    Needs a *phi* for the walk, which scores its archive by it; the front it
    returns is the same front either way.
    """
    if phi is None:
        raise ValueError("the Pareto walk is guided by Phi and needs one; "
                         "use enumerate_front for the unguided enumeration")
    found = pareto_seeds(problem, phi, **walk)
    remaining = (None if time_budget is None
                 else max(0.0, time_budget - found.seconds))
    front = enumerate_front(problem, phi, max_boxes=max_boxes,
                            time_budget=remaining, seeds=found.points)
    return front, found


def hybrid_complete_set(problem: MOILFP, phi: FractionalObjective,
                        bounds=None, max_boxes: int = 200_000,
                        max_points: int = 1_000_000,
                        time_budget: Optional[float] = None,
                        **walk) -> CompleteSet:
    """``E(P_D)`` in full, reached by a metaheuristic--exact hybrid.

    Pareto local search supplies efficient points; each is verified and used to
    split the box list without a probe; the exact enumeration finishes the
    front and proves it complete; the slices then complete it to every
    efficient point.  **The answer is identical to the pure exact method** --
    the heuristic decides only how much of the front has to be discovered, and
    the completeness proof is untouched because a seed enters the list exactly
    where a found point would have.

    Measured against :func:`~lfp_efficient.complete.complete_efficient_set`
    over the same 12 instances: **1.30x** in the median, min $0.82$, max
    $1.65$, and the same set of points every time.
    """
    front, found = pareto_front(problem, phi, max_boxes=max_boxes,
                                time_budget=time_budget, **walk)
    spent = found.seconds
    remaining = None if time_budget is None else max(0.0, time_budget - spent)
    return complete_efficient_set(problem, phi, bounds=bounds,
                                  max_boxes=max_boxes, max_points=max_points,
                                  time_budget=remaining, front=front)
