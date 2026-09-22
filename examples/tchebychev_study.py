"""The augmented weighted Tchebychev program, measured against what is here.

The scalarisation of Chaabane, Brahmi and Ramdani (2012) and of Younsi-Abbaci
and Moulai (2021) is the reference this package has to be compared with, since
the surrounding literature generates efficient points with it.  This script
runs that comparison and prints the three numbers the README quotes.

1. **Reach.**  How much of ``E(P_D)`` each scalarisation touches from the same
   grid of weights, and in particular how many *unsupported* efficient points
   -- the ones no strictly positive weighted sum can ever maximise at.
2. **Seed.**  What each generator is worth as a starting incumbent for the
   criterion-space box search, in total time and in how often the seed is
   already optimal.
3. **Coverage.**  The Tchebychev grid against the Pareto archive as generators
   of a subset of the efficient set, on time and on completeness.

The headline is a split verdict, and both halves matter: as a generator the
Tchebychev program does what the literature says (it reaches unsupported points
the weighted sum cannot, and every optimum it returns is efficient), while as a
seed for this objective it is slower than not seeding at all -- because its
weights steer in criterion space and never look at ``Phi``.

Run with ``python examples/tchebychev_study.py`` (a couple of minutes).
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lfp_efficient import (FractionalObjective, LE, MOILP, Model,
                           augmented_tchebychev_efficient,
                           enumerate_efficient_set, metaheuristic_incumbent,
                           optimize_in_criterion_space, pareto_local_search,
                           spread_weights, tchebychev_incumbent,
                           test_efficiency, weighted_sum_efficient)

GRID = [[a, b, c] for a in (1, 2, 5) for b in (1, 2, 5) for c in (1, 2, 5)]


def instance(seed, n, p=3, ub=3, tightness=0.4):
    """A small bounded instance: a box, a few loose rows, and a ratio ``Phi``."""
    rng = random.Random(seed)
    model = Model(n)
    for j in range(n):
        row = [0] * n
        row[j] = 1
        model.add(row, LE, ub)
    for _ in range(2):
        row = [rng.randint(1, 4) for _ in range(n)]
        model.add(row, LE, max(1, int(sum(row) * ub * tightness)))
    criteria = [[rng.randint(-3, 5) for _ in range(n)] for _ in range(p)]
    phi = FractionalObjective([-rng.randint(1, 5) for _ in range(n)],
                              [rng.randint(1, 3) for _ in range(n)],
                              rng.randint(20, 40), rng.randint(4, 8))
    return MOILP(model, criteria), phi, ub


def supported_points(problem, efficient):
    """The efficient points some strictly positive weighted sum maximises at.

    Found by scanning a fine grid rather than by an LP: the instances here are
    small, and a scan cannot claim a point is unsupported for a subtle reason.
    """
    values = {x: tuple(z(list(x)) for z in problem.criteria) for x in efficient}
    out = set()
    for w1 in range(1, 25):
        for w2 in range(1, 25):
            for w3 in range(1, 25):
                top = max(w1 * values[x][0] + w2 * values[x][1] + w3 * values[x][2]
                          for x in efficient)
                out.update(x for x in efficient
                           if w1 * values[x][0] + w2 * values[x][1]
                           + w3 * values[x][2] == top)
    return out


def reach_study(trials=20):
    print("\n1. Reach: what each scalarisation touches from the same 27 weights")
    print("-" * 70)
    total = reached_t = reached_w = 0
    unsup_total = unsup_t = unsup_w = 0
    programs = inefficient = 0
    for seed in range(trials):
        problem, _, ub = instance(seed, 4)
        efficient = [tuple(x) for x in
                     enumerate_efficient_set(problem, [ub] * problem.n).efficient]
        if not efficient:
            continue
        unsupported = set(efficient) - supported_points(problem, efficient)

        got_t, got_w = set(), set()
        for w in GRID:
            point = augmented_tchebychev_efficient(problem, w)
            if point is not None:
                programs += 1
                inefficient += not test_efficiency(problem, point).efficient
                got_t.add(tuple(point))
            other = weighted_sum_efficient(problem, w)
            if other is not None:
                got_w.add(tuple(other))

        total += len(efficient)
        reached_t += len(got_t)
        reached_w += len(got_w)
        unsup_total += len(unsupported)
        unsup_t += len(got_t & unsupported)
        unsup_w += len(got_w & unsupported)

    print(f"   {total} efficient points over {trials} instances, "
          f"{unsup_total} of them unsupported")
    print(f"   weighted sum         : {reached_w:4d} reached, "
          f"{unsup_w:3d} unsupported")
    print(f"   augmented Tchebychev : {reached_t:4d} reached, "
          f"{unsup_t:3d} unsupported")
    print(f"   {programs} Tchebychev programs, {inefficient} optima not efficient")


def seed_study():
    print("\n2. Seed: what each generator is worth to the box search")
    print("-" * 70)
    print(f"   {'instance':10s} | {'no seed':>15s} | {'Pareto search':>15s} | "
          f"{'Tchebychev':>15s}")
    bare = meta = tcheb = 0.0
    cost_m = cost_t = 0.0
    hit_m = hit_t = 0
    count = 0
    # tight instances at n = 6, 7 are solved in hundredths of a second, and a
    # seed cannot pay for itself in a search that short.  The looser ones below
    # are where the question has an answer.
    for seed in range(6):
        for n, tightness in ((6, 0.3), (7, 0.3), (8, 0.45), (9, 0.45)):
            problem, phi, _ = instance(seed, n, tightness=tightness)
            count += 1

            t = time.time()
            exact = optimize_in_criterion_space(problem, phi)
            bare_elapsed = time.time() - t
            bare += bare_elapsed

            row = []
            for name, find in (("m", metaheuristic_incumbent),
                               ("t", tchebychev_incumbent)):
                t = time.time()
                found = find(problem, phi)
                cost = time.time() - t
                t = time.time()
                if found is None:
                    run = optimize_in_criterion_space(problem, phi)
                else:
                    run = optimize_in_criterion_space(
                        problem, phi, incumbent=found[0],
                        incumbent_value=found[1], explored_points=[found[0]])
                elapsed = time.time() - t + cost
                assert run.value == exact.value and run.proved_optimal
                if name == "m":
                    meta += elapsed
                    cost_m += cost
                    hit_m += found is not None and found[1] == exact.value
                else:
                    tcheb += elapsed
                    cost_t += cost
                    hit_t += found is not None and found[1] == exact.value
                row.append(elapsed)

            print(f"   n={n} s{seed}     | {len(exact.iterations):4d} bx "
                  f"{bare_elapsed:6.2f}s | {row[0]:15.2f}s | {row[1]:14.2f}s",
                  flush=True)

    print(f"   totals: no seed {bare:.2f}s | Pareto search {meta:.2f}s "
          f"({bare / meta:.2f}x) | Tchebychev {tcheb:.2f}s ({bare / tcheb:.2f}x)")
    print(f"   seed cost: Pareto search {cost_m:.2f}s | Tchebychev {cost_t:.2f}s")
    print(f"   exactly optimal: Pareto search {hit_m}/{count} | "
          f"Tchebychev {hit_t}/{count}")
    print("\n   Two things to read off, and they are different things.  The")
    print("   ordering is stable: the Tchebychev seed loses to not seeding at")
    print("   all, on every family tried, while the Pareto seed ties or wins.")
    print("   But seed QUALITY is not seed VALUE -- the Pareto seed is exactly")
    print("   optimal on most of these and still buys little, because the box")
    print("   search here is not bound-limited.  On the heavier family of the")
    print("   hybrid study (README) the same seed is worth 1.21x.")


def coverage_study(trials=20):
    print("\n3. Coverage: the two as generators of a subset of E(P_D)")
    print("-" * 70)
    total = got_t = got_m = 0
    time_t = time_m = 0.0
    uncertified = 0
    for seed in range(trials):
        problem, phi, ub = instance(seed, 4)
        efficient = {tuple(x) for x in
                     enumerate_efficient_set(problem, [ub] * problem.n).efficient}
        if not efficient:
            continue

        t = time.time()
        reached = {tuple(x) for w in GRID
                   if (x := augmented_tchebychev_efficient(problem, w))}
        time_t += time.time() - t

        t = time.time()
        archive = pareto_local_search(problem, phi)
        time_m += time.time() - t
        members = {tuple(x) for x in archive.points()}
        uncertified += sum(1 for x in members
                           if not test_efficiency(problem, list(x)).efficient)

        total += len(efficient)
        got_t += len(reached)
        got_m += len(members)

    print(f"   augmented Tchebychev : {got_t:4d}/{total} = {got_t / total:3.0%} "
          f"in {time_t:5.2f}s, every optimum efficient by construction")
    print(f"   Pareto archive       : {got_m:4d}/{total} = {got_m / total:3.0%} "
          f"in {time_m:5.2f}s, {uncertified} of {got_m} member(s) NOT efficient")
    print("   (the archive filters by dominance among the points it has seen,")
    print("    which is not the exact test -- over a wider sweep the miss rate")
    print("    came out around 1 in 2000, rare but not zero)")
    print("\n   Cheap, near-complete, uncertified against expensive, partial,")
    print("   certified.  That trade is why metaheuristic_incumbent puts the")
    print("   archive's best through the exact test before anything crosses.")


def main():
    print(__doc__.split("\n\n")[0])
    reach_study()
    seed_study()
    coverage_study()
    return 0


if __name__ == "__main__":
    sys.exit(main())
