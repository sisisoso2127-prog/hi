"""Larger instances, each cross-checked against an independent exact reference.

The numerical illustration of the paper has 11 feasible points and 7 efficient
ones -- small enough that any method works.  This script runs the algorithm on
instances up to ``n = 10`` variables, ``p = 3`` criteria and tens of thousands
of feasible points, and verifies every answer with
``best_over_efficient_set_by_scan``: sort the feasible points by decreasing
``Phi`` and return the first one that survives a dominance test.  That
reference shares no code path with the algorithm -- no cut, no efficiency LP,
no simplex -- so an agreement between the two is a real check.

Two regimes are covered, and they behave very differently:

* **easy** -- the maximiser of ``Phi`` over ``D`` happens to be efficient, so
  the very first efficiency test settles the problem in one iteration;
* **hard** -- the criteria reward large ``x`` while ``Phi`` rewards small
  ``x``, so the maximiser of ``Phi`` over ``D`` is dominated and the algorithm
  has to cut its way through several non-dominated vectors.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lfp_efficient import (FractionalObjective, LE, MOILP, Model,
                           best_over_efficient_set_by_scan,
                           enumerate_efficient_set,
                           optimize_over_efficient_set)
from lfp_efficient.rational import fmt


def boxed(n, ub, rows, criteria, U, V, alpha, beta):
    model = Model(n)
    for j in range(n):
        unit = [0] * n
        unit[j] = 1
        model.add(unit, LE, ub)
    for coeffs, rhs in rows:
        model.add(coeffs, LE, rhs)
    return MOILP(model, criteria), FractionalObjective(U, V, alpha, beta), [ub] * n


def instance_medium(n):
    """Mixed-sign criteria, three resource constraints, x_j in 0..3."""
    return boxed(
        n, 3,
        [([2, 3, 1, 4, 2, 1][:n], 14),
         ([1, 2, 3, 1, 1, 2][:n], 12),
         ([3, 1, 2, 1, 3, 1][:n], 15)],
        [[3, 1, -2, 1, -1, 2][:n],
         [-1, 2, 3, -1, 2, -1][:n],
         [1, -1, 1, 2, 1, 1][:n]],
        U=[2, 3, 1, 4, 1, 2][:n], V=[1] * n, alpha=1, beta=2)


def instance_large():
    """n = 10, four resource constraints, tens of thousands of feasible points."""
    return boxed(
        10, 3,
        [([3, 2, 4, 1, 2, 3, 1, 2, 2, 3], 22),
         ([1, 4, 2, 3, 1, 2, 3, 1, 4, 1], 20),
         ([2, 1, 3, 2, 4, 1, 2, 3, 1, 2], 21),
         ([4, 1, 1, 3, 2, 2, 1, 4, 2, 1], 23)],
        [[4, 3, -1, 2, 1, -2, 3, 1, 2, -1],
         [-2, 1, 4, 1, -1, 3, 1, 2, -3, 2],
         [1, -1, 2, 3, 2, 1, -2, 4, 1, 3]],
        U=[5, 4, 3, 6, 2, 4, 3, 5, 4, 6], V=[2, 1, 2, 1, 3, 1, 2, 1, 1, 2],
        alpha=2, beta=3)


def instance_hard():
    """n = 10 where the maximiser of Phi over D is *dominated*.

    Every criterion rewards large ``x`` while ``Phi`` has an all-negative
    numerator, so its unconstrained-looking optimum sits near the origin, far
    from the efficient frontier: the algorithm must generate several
    non-dominated vectors before the bound closes.
    """
    return boxed(
        10, 3,
        [([2, 2, 3, 4, 1, 1, 4, 3, 2, 2], 25),
         ([4, 4, 2, 2, 2, 4, 1, 1, 2, 1], 22),
         ([1, 3, 4, 4, 4, 4, 4, 2, 3, 1], 18),
         ([2, 4, 2, 3, 4, 3, 4, 4, 3, 4], 21)],
        [[3, 1, 3, 5, 2, 3, 5, 5, 5, 1],
         [2, 5, 3, 3, 1, 1, 4, 4, 1, 3],
         [1, 4, 2, 1, 3, 4, 4, 1, 1, 5]],
        U=[-5, -1, -4, -5, -3, -5, -3, -5, -2, -1], V=[2, 1, 1, 1, 3, 3, 1, 1, 2, 2],
        alpha=39, beta=6)


def instance_hardest():
    """The heaviest of the family: 11 cut iterations before the bound closes."""
    return boxed(
        10, 3,
        [([3, 2, 4, 1, 1, 1, 3, 1, 2, 1], 19),
         ([4, 4, 1, 2, 1, 4, 1, 1, 2, 1], 24),
         ([1, 2, 1, 2, 3, 4, 2, 1, 3, 2], 19),
         ([2, 3, 1, 1, 1, 2, 4, 4, 3, 4], 25)],
        [[3, 3, 2, 2, 2, 1, 5, 3, 5, 4],
         [3, 4, 3, 5, 1, 1, 5, 4, 2, 3],
         [2, 4, 4, 1, 1, 5, 5, 3, 3, 3]],
        U=[-5, -4, -5, -4, -1, -1, -3, -4, -1, -1], V=[3, 3, 2, 3, 3, 3, 2, 2, 3, 2],
        alpha=31, beta=4)


CASES = [
    ("medium  n=4", lambda: instance_medium(4), True),
    ("medium  n=5", lambda: instance_medium(5), True),
    ("medium  n=6", lambda: instance_medium(6), True),
    ("large   n=10 (easy Phi)", instance_large, False),
    ("hard    n=10 (dominated Phi optimum)", instance_hard, False),
    ("hardest n=10 (11 cut iterations)", instance_hardest, False),
]


def main():
    print(f"{'instance':38s} {'|D|':>7s} {'Phi_opt':>10s} {'iter':>5s} "
          f"{'gen':>4s} {'algo':>9s} {'check':>9s}  ok")
    print("-" * 96)

    for name, builder, count_efficient in CASES:
        problem, phi, bounds = builder()

        t = time.time()
        solution = optimize_over_efficient_set(problem, phi)
        t_algo = time.time() - t

        t = time.time()
        ref_x, ref_value, n_feasible, n_tested = \
            best_over_efficient_set_by_scan(problem, phi, bounds)
        t_check = time.time() - t

        ok = solution.value == ref_value
        print(f"{name:38s} {n_feasible:7d} {fmt(solution.value):>10s} "
              f"{len(solution.iterations):5d} {len(solution.explored):4d} "
              f"{t_algo:8.2f}s {t_check:8.2f}s  {'yes' if ok else 'NO'}")
        if not ok:
            raise SystemExit(f"MISMATCH on {name}: "
                             f"{fmt(solution.value)} vs {fmt(ref_value)}")

        # the reported point must itself be efficient and realise the value
        assert phi(solution.x) == solution.value
        if count_efficient:
            total = len(enumerate_efficient_set(problem, bounds).efficient)
            print(f"{'':38s} -> generated {len(solution.explored)} of the "
                  f"{total} efficient points ({100 * len(solution.explored) / total:.0f}%), "
                  f"reference needed {n_tested} dominance test(s)")

    print("-" * 96)
    print("every instance verified against the independent scan reference.")


if __name__ == "__main__":
    main()
