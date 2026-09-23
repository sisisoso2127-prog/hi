#!/usr/bin/env python3
"""Five structurally different instance families, and the hybrid on all of them.

Every figure in this project comes from one pseudo-random generator: four
dense inequality rows, coefficients in $1..4$, tightness $0.30$, criteria in
$1..5$, $ub = 3$.  The standing limitation has been that the *magnitudes*
belong to that family even where the *directions* are explained by measured
mechanisms.  This removes the limitation rather than restating it.

The families are chosen to move the quantities the mechanisms depend on:

``mixed``        the original -- dense rows with mixed criterion signs.
``knapsack``     every constraint coefficient positive and few rows, so the
                 feasible set is a corner and the criteria compete only
                 through capacity.  The structure most of the applied
                 literature actually uses.
``conflicting``  criteria with opposing signs, which makes the non-dominated
                 set large.  Front size is what drives the enumeration, so
                 this is the family the hybrid should like most.
``aligned``      criteria positively correlated, which makes the front small.
                 The family the hybrid should like least: little to discover.
``loose``        the original with tightness $0.60$ instead of $0.30$, so
                 $|\\Dset|$ grows by an order of magnitude while $|F|$ does not.

If the hybrid's advantage were an artefact of the original generator, these
would not agree.  Run with ``python studies/families.py``.
"""
import argparse
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lfp_efficient import FractionalObjective, LE, MOILP, Model
from lfp_efficient.front import enumerate_front
from lfp_efficient.generated import pareto_front


def _phi(rng, n):
    return FractionalObjective([-rng.randint(1, 5) for _ in range(n)],
                               [rng.randint(1, 3) for _ in range(n)],
                               rng.randint(20, 45), rng.randint(4, 10))


def mixed(seed, n, p, ub=3, tightness=0.30, rows=4):
    rng = random.Random(seed * 7919 + n * 101 + p)
    model = Model(n)
    for j in range(n):
        unit = [0] * n
        unit[j] = 1
        model.add(unit, LE, ub)
    for _ in range(rows):
        row = [rng.randint(1, 4) for _ in range(n)]
        model.add(row, LE, max(1, int(sum(row) * ub * tightness)))
    return (MOILP(model, [[rng.randint(1, 5) for _ in range(n)]
                          for _ in range(p)]), _phi(rng, n))


def loose(seed, n, p):
    return mixed(seed, n, p, tightness=0.60)


def knapsack(seed, n, p, ub=3):
    """Two capacity rows, all-positive data: the applied shape."""
    rng = random.Random(seed * 104729 + n * 31 + p)
    model = Model(n)
    for j in range(n):
        unit = [0] * n
        unit[j] = 1
        model.add(unit, LE, ub)
    for _ in range(2):
        weights = [rng.randint(3, 20) for _ in range(n)]
        model.add(weights, LE, int(sum(weights) * ub * 0.35))
    return (MOILP(model, [[rng.randint(1, 20) for _ in range(n)]
                          for _ in range(p)]), _phi(rng, n))


def conflicting(seed, n, p, ub=3):
    """Criteria pulling against each other: a large non-dominated set."""
    rng = random.Random(seed * 15485863 + n * 17 + p)
    model = Model(n)
    for j in range(n):
        unit = [0] * n
        unit[j] = 1
        model.add(unit, LE, ub)
    for _ in range(3):
        row = [rng.randint(1, 4) for _ in range(n)]
        model.add(row, LE, max(1, int(sum(row) * ub * 0.40)))
    criteria = []
    for k in range(p):
        # each criterion rewards its own half of the variables and penalises
        # the other, so improving one costs another
        sign = [1 if (j + k) % p == 0 else -1 for j in range(n)]
        criteria.append([sign[j] * rng.randint(1, 5) for j in range(n)])
    return MOILP(model, criteria), _phi(rng, n)


def aligned(seed, n, p, ub=3):
    """Criteria that mostly agree: a small non-dominated set."""
    rng = random.Random(seed * 32452843 + n * 13 + p)
    model = Model(n)
    for j in range(n):
        unit = [0] * n
        unit[j] = 1
        model.add(unit, LE, ub)
    for _ in range(3):
        row = [rng.randint(1, 4) for _ in range(n)]
        model.add(row, LE, max(1, int(sum(row) * ub * 0.35)))
    base = [rng.randint(3, 9) for _ in range(n)]
    criteria = [[max(1, b + rng.randint(-3, 3)) for b in base]
                for _ in range(p)]
    return MOILP(model, criteria), _phi(rng, n)


#: each family with the sizes that keep it solvable and informative.  The
#: front sizes differ by two orders of magnitude between them, which is the
#: point: |F| is what drives the enumeration, so a family that fixes it fixes
#: the answer.
FAMILIES = {
    "mixed":       (mixed,       [(7, 3), (8, 4)]),
    "loose":       (loose,       [(7, 3), (8, 4)]),
    "knapsack":    (knapsack,    [(7, 3), (8, 4)]),
    "conflicting": (conflicting, [(6, 3), (7, 3)]),
    "aligned":     (aligned,     [(7, 3), (8, 4)]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=15)
    parser.add_argument("--budget", type=float, default=300.0)
    args = parser.parse_args()

    print(f"{args.instances} instances per cell, {args.budget:.0f}s budget\n")
    print(f"{'family':>12} {'n':>3} {'p':>2} | {'|F|':>6} {'|D| probes':>11} | "
          f"{'cover':>6} | {'exact':>9} {'hybrid':>9} {'ratio':>7} {'wins':>6}")
    print("-" * 86)
    overall = {}
    for name, (build, sizes) in FAMILIES.items():
        pooled = []
        for n, p in sizes:
            ratios, cover, fronts, probes, ta_s, tb_s = [], [], [], [], [], []
            censored = 0
            for seed in range(args.instances):
                problem, phi = build(seed, n, p)
                t = time.perf_counter()
                a = enumerate_front(problem, phi, time_budget=args.budget)
                ta = time.perf_counter() - t
                if not a.complete:
                    censored += 1
                    continue
                t = time.perf_counter()
                b, found = pareto_front(problem, phi, time_budget=args.budget)
                tb = time.perf_counter() - t
                assert ({tuple(v) for v in a.vectors}
                        == {tuple(v) for v in b.vectors}), f"{name} {seed}"
                assert max(a.values) == max(b.values), f"{name} {seed}"
                ratios.append(ta / tb)
                cover.append(b.seeded / max(1, len(a.vectors)))
                fronts.append(len(a.vectors))
                probes.append(a.probes)
                ta_s.append(ta)
                tb_s.append(tb)
            if not ratios:
                print(f"{name:>12} {n:>3} {p:>2} | all censored")
                continue
            pooled += ratios
            note = f" [{censored} cens]" if censored else ""
            print(f"{name:>12} {n:>3} {p:>2} | "
                  f"{statistics.median(fronts):>6.0f} "
                  f"{statistics.median(probes):>11.0f} | "
                  f"{statistics.median(cover):>5.0%} | "
                  f"{statistics.median(ta_s):>8.3f}s "
                  f"{statistics.median(tb_s):>8.3f}s "
                  f"{statistics.median(ratios):>6.2f}x "
                  f"{sum(1 for r in ratios if r > 1):>3}/{len(ratios):<3}{note}")
            sys.stdout.flush()
        overall[name] = pooled

    print("\nper family, pooled:")
    for name, rs in overall.items():
        if not rs:
            continue
        q = statistics.quantiles(rs, n=4) if len(rs) >= 4 else [min(rs), 0, max(rs)]
        print(f"  {name:<12} median {statistics.median(rs):.2f}x "
              f"[{q[0]:.2f}, {q[2]:.2f}]  worst {min(rs):.2f}x  "
              f"faster on {sum(1 for r in rs if r > 1)}/{len(rs)}")
    every = [r for rs in overall.values() for r in rs]
    print(f"\n  all {len(every)} instances: median "
          f"{statistics.median(every):.2f}x, faster on "
          f"{sum(1 for r in every if r > 1)}")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
