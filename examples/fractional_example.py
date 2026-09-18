"""The fractional generalisation: every criterion is a ratio (MOILFP).

The paper optimises a fractional ``Phi`` over the efficient set of a program
whose criteria are **linear**.  Here the criteria are ratios too::

    (MOILFP)  "max"  Z_k(x) = (c_k'x + a_k) / (d_k'x + b_k),  k = 1..p
              s.t.   x in D = { x in Z^n_+ : A x <~ b }

    (P_E)      max   Phi(x) = (U'x + alpha) / (V'x + beta)
               s.t.  x in E(MOILFP)

Nothing in the method changes shape.  What changes is that every question about
the criteria is asked through the linear form

    e_k(x ; a) = D_k(a) * D_k(x) * ( Z_k(x) - Z_k(a) )

which is linear in ``x`` and carries the sign of ``Z_k(x) - Z_k(a)``.  With
integer data it is integer-valued, so "strictly better on criterion k" is the
exact condition ``e_k(x) >= 1`` -- the "+1" of the linear cut, transposed to
ratios without estimating a minimal step.

One thing does change, and it is not cosmetic: with linear criteria the
maximiser of the efficiency test is itself efficient (Ecker & Kouada), while
with ratios it only *dominates*, because the k-th term of the test's objective
carries a factor ``D_k(x)`` that varies from point to point.  The dominance
chain has to be walked.  ``tests/`` measures the difference: every linear chain
is one step, while a measurable share of fractional ones are longer.
"""

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lfp_efficient import (FractionalObjective, LE, MOILFP, Model,
                           optimize_over_efficient_set, repair_to_efficient,
                           test_efficiency)
from lfp_efficient.rational import F, fmt


def build():
    # Chosen so that the dominance chain is genuinely longer than one step --
    # the behaviour that separates ratios from linear criteria.
    model = (Model(2)
             .add([1, 0], LE, 6)
             .add([0, 1], LE, 6)
             .add([1, 1], LE, 6)
             .add([2, 1], LE, 11))
    problem = MOILFP(model, [
        FractionalObjective([-1, 0], [2, 2], 2, 3),     # (-x1 + 2)/(2x1 + 2x2 + 3)
        FractionalObjective([4, 2], [1, 2], 0, 2),      # (4x1 + 2x2)/(x1 + 2x2 + 2)
    ])
    phi = FractionalObjective([1, 1], [1, 2], 0, 3)     # (x1 + x2)/(x1 + 2x2 + 3)
    return problem, phi, [6, 6]


def point(x):
    return "(" + ", ".join(fmt(v) for v in x) + ")"


def main():
    problem, phi, bounds = build()
    print("criteria, all of them ratios:")
    for k, z in enumerate(problem.criteria, start=1):
        print(f"    Z{k}(x) = {z}")
    print(f"    Phi(x) = {phi}\n")

    feasible = [[F(a), F(b)] for a, b in product(*(range(u + 1) for u in bounds))
                if problem.model.is_feasible([F(a), F(b)])]
    efficient = [x for x in feasible
                 if not any(problem.dominates(y, x) for y in feasible)]

    print(f"|D| = {len(feasible)} feasible points, |E| = {len(efficient)} efficient")
    print("  E(MOILFP) =", ", ".join(point(x) for x in efficient))

    # the exact test must reproduce that classification point by point
    misread = [x for x in feasible
               if test_efficiency(problem, x).efficient != (x in efficient)]
    print(f"  the exact test agrees with the definition on all {len(feasible)} points: "
          f"{'yes' if not misread else 'NO -- ' + str(misread)}")

    # and the chain has to be walked: show a point that needs more than one step
    for x in feasible:
        if x in efficient:
            continue
        first = test_efficiency(problem, x).witness
        if first not in efficient:
            print(f"\n  a chain the linear case never produces:")
            print(f"    {point(x)} is dominated; the test returns {point(first)},")
            print(f"    which is itself still dominated, and the walk ends at "
                  f"{point(repair_to_efficient(problem, x))}.")
            print(f"    With linear criteria the first answer is always already "
                  f"efficient (Ecker & Kouada);")
            print(f"    with ratios the test's objective is weighted by D_k(x), "
                  f"so it need not be.")
            break

    best = max(efficient, key=phi)
    print(f"\n  brute force : max Phi over E = {fmt(phi(best))} at {point(best)}")

    solution = optimize_over_efficient_set(problem, phi)
    print(f"  algorithm   : Phi_opt = {fmt(solution.value)} at {point(solution.x)}"
          f"   [{len(solution.iterations)} iteration(s), "
          f"{'proved optimal' if solution.proved_optimal else 'not proved'}]")

    assert solution.value == phi(best), "the algorithm missed the optimum"
    print("\nalgorithm and the definition agree on a fully fractional instance.")


if __name__ == "__main__":
    main()
