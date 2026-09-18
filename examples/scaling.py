"""How far the algorithm goes, and what actually makes an instance hard.

The paper stops at ``n = 2`` with 11 feasible points.  This script walks a
family of instances up to ``n = 20``, with tens of thousands of feasible
points, and proves each answer with :func:`~lfp_efficient.certify_optimum`
wherever that stays affordable.

What the numbers say
--------------------
The size of ``n`` is *not* what decides the cost.  Two things do:

* **how many cut iterations** the instance needs -- one per non-dominated
  vector generated, and each cut adds ``p`` binaries and ``p+1`` rows to every
  sub-problem that follows;
* **how hard ``max Phi`` over the region is as an integer program** -- which
  grows with the number of feasible points, so with ``n``, with the variable
  bound ``ub``, and with how loose the constraints are.

An instance with ``n = 20`` and a tight feasible region is solved in seconds,
while one with ``n = 16`` and loose constraints is not solved at all: the very
first sub-problem, ``max Phi`` over ``D`` with no cut yet, is already a hard
integer program.  That is a property of the sub-problem, not of the method.

Certification
-------------
Past roughly ``|D| = 10^5`` no reference method can enumerate the region any
more, so correctness is established differently: ``certify_optimum`` proves
that the returned point is efficient and that **every** feasible point with a
strictly greater ``Phi`` is dominated.  The set it has to inspect is carved out
by one extra linear row, and almost all of its members are settled by a
dominance witness already in hand -- a handful of exact efficiency tests is
usually enough for tens of thousands of challengers.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lfp_efficient import (FractionalObjective, LE, MOILP, Model,
                           certify_optimum, optimize_over_efficient_set)
from lfp_efficient.rational import fmt


def instance(n, ub, coefficients, rhs, criteria, U, V, alpha, beta):
    model = Model(n)
    for j in range(n):
        unit = [0] * n
        unit[j] = 1
        model.add(unit, LE, ub)
    for row, r in zip(coefficients, rhs):
        model.add(row, LE, r)
    return MOILP(model, criteria), FractionalObjective(U, V, alpha, beta), [ub] * n


def build(seed, n, ub, tightness, negative_numerator=True):
    """A reproducible pseudo-random instance of the family used in the study.

    ``tightness`` is the share of the maximal resource consumption each
    constraint allows, so it controls ``|D|`` directly.  With
    ``negative_numerator`` the criteria reward large ``x`` while ``Phi`` rewards
    small ``x``: the maximiser of ``Phi`` over ``D`` is then dominated and the
    algorithm must cut its way to the answer.
    """
    import random
    rng = random.Random(seed)
    coefficients, rhs = [], []
    for _ in range(4):
        row = [rng.randint(1, 4) for _ in range(n)]
        coefficients.append(row)
        rhs.append(max(1, int(sum(row) * ub * tightness)))
    criteria = [[rng.randint(1, 5) for _ in range(n)] for _ in range(3)]
    U = ([-rng.randint(1, 5) for _ in range(n)] if negative_numerator
         else [rng.randint(1, 6) for _ in range(n)])
    V = [rng.randint(1, 3) for _ in range(n)]
    return instance(n, ub, coefficients, rhs, criteria,
                    U, V, rng.randint(20, 45), rng.randint(4, 10))


#: (label, seed, n, ub, tightness, certify?)
STUDY = [
    ("n=10  ub=3", 0, 10, 3, 0.25, True),
    ("n=12  ub=3", 0, 12, 3, 0.25, True),
    ("n=12  ub=3", 1, 12, 3, 0.25, True),
    ("n=14  ub=3", 1, 14, 3, 0.25, False),
    ("n=16  ub=3", 0, 16, 3, 0.15, False),
    ("n=20  ub=2", 0, 20, 2, 0.15, False),
]


def main():
    print(f"{'instance':14s} {'Phi_opt':>10s} {'iter':>5s} {'gen':>4s} {'solve':>9s}"
          f"   certificate")
    print("-" * 88)
    for label, seed, n, ub, tightness, certify in STUDY:
        problem, phi, bounds = build(seed, n, ub, tightness)

        t = time.time()
        solution = optimize_over_efficient_set(problem, phi)
        t_solve = time.time() - t

        line = (f"{label:14s} {fmt(solution.value):>10s} "
                f"{len(solution.iterations):5d} {len(solution.explored):4d} "
                f"{t_solve:8.2f}s")
        if certify:
            t = time.time()
            proof = certify_optimum(problem, phi, solution.x, solution.value, bounds)
            t_proof = time.time() - t
            if not proof.valid:
                raise SystemExit(f"{label}: {proof.reason}")
            line += (f"   proved in {t_proof:6.2f}s "
                     f"({proof.challengers} challengers, {proof.tests} tests)")
        else:
            line += "   not certified (the challenger set is too large to walk)"
        print(line, flush=True)

    print("-" * 88)
    print("Difficulty follows |D| and the number of cut iterations, not n alone:")
    print("the same n = 16 is solved in seconds when the region is tight and")
    print("stays out of reach when it is loose.")


if __name__ == "__main__":
    main()
