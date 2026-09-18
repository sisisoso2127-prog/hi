"""The numerical illustration of section 4 of the paper.

    max Z1 = x1 - 2 x2
    max Z2 = -x1 + 4 x2
    s.t.   -2 x1 +   x2 <= 0
            6 x1 +   x2 <= 21
           -2 x1 + 4 x2 <= 6
           x1, x2 >= 0 integer

    main criterion   Phi(x) = (x1 + x2 - 1) / (5 x1 + x2 - 1)

D holds 11 feasible points, 7 of which are efficient:
    E(P_D) = {(2,0), (2,1), (2,2), (3,0), (3,1), (3,2), (3,3)}
and the expected answer is X_opt = (3,3) with Phi_opt = 5/17.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lfp_efficient import (FractionalObjective, LE, MOILP, Model,
                           enumerate_efficient_set, optimize_over_efficient_set)
from lfp_efficient.rational import fmt


def build():
    D = (Model(2)
         .add([-2, 1], LE, 0)
         .add([6, 1], LE, 21)
         .add([-2, 4], LE, 6))
    problem = MOILP(D, [[1, -2], [-1, 4]])
    phi = FractionalObjective(U=[1, 1], V=[5, 1], alpha=-1, beta=-1)
    return problem, phi


def main():
    problem, phi = build()
    print(f"main criterion  Phi(x) = {phi}\n")

    solution = optimize_over_efficient_set(problem, phi, verbose=True)

    print()
    reference = enumerate_efficient_set(problem, bounds=[4, 4])
    print("brute force check over the whole box 0..4 x 0..4")
    print("  feasible points  :", len(reference.feasible))
    print("  efficient points :", ", ".join(
        "(" + ", ".join(fmt(v) for v in p) + ")" for p in reference.efficient))
    best_x, best_phi = reference.best(phi)
    print(f"  max Phi over E(P_D) = {fmt(best_phi)} at "
          f"({', '.join(fmt(v) for v in best_x)})")
    assert best_phi == solution.value, "the algorithm missed the optimum!"
    print("\nalgorithm and exhaustive enumeration agree.")
    print(f"the algorithm generated {len(solution.explored)} efficient points "
          f"out of {len(reference.efficient)}.")


if __name__ == "__main__":
    main()
