"""
Diagnostics for hybrid_validation.py.

The validation harness reports PASS on both cases, but the PASS does not
establish what it claims. This script reproduces three issues.

Run:  python3 diagnostics.py
"""

import time

import numpy as np


# Load hybrid_validation.py without executing its module-level MAIN block.
_SRC = open("hybrid_validation.py").read().split("# MAIN - COLAB VERSION")[0]
_HV = {}
exec(compile(_SRC, "hybrid_validation.py", "exec"), _HV)

Search = _HV["Search"]
feasible_solutions = _HV["feasible_solutions"]
objectives = _HV["objectives"]
exact_pareto = _HV["exact_pareto"]
dominates = _HV["dominates"]
N_value = _HV["N_value"]
D_value = _HV["D_value"]


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ------------------------------------------------------------------
# 1. The benchmark's Pareto front is a single point.
# ------------------------------------------------------------------

def issue_1_degenerate_front():
    rule("1. The benchmark has a singleton Pareto front")

    print("F(x) = (2*x1 + 2*x2, x1 + 4*x2)   over   x1 + x2 <= n, x >= 0")
    print()
    print("With s = x1 + x2:  f1 = 2s depends only on s, and f2 = 4s - 3*x1.")
    print("So for every s, x1 = 0 is weakly better in f1 and strictly better in")
    print("f2; and both objectives grow with s. Hence (0, n) dominates the whole")
    print("feasible set and the Pareto front is always exactly one point.")
    print()

    for n in (4, 8, 20, 40):
        front = exact_pareto(n)
        pts = [tuple(int(v) for v in x) for x, _ in front]
        print(f"  n = {n:3d}   |X| = {len(feasible_solutions(n)):5d}"
              f"   |Pareto| = {len(front)}   front = {pts}")

    print()
    print("Recall and precision of 1.000000 therefore say nothing about the")
    print("search: any method that finds (0, n) scores a perfect front.")


# ------------------------------------------------------------------
# 2. The search enumerates the entire feasible set.
# ------------------------------------------------------------------

def issue_2_exhaustive_search():
    rule("2. Search.run() is exhaustive enumeration")

    print("The main loop breaks only when raw_candidates() is empty, and every")
    print("batch member is passed to the oracle. So the oracle count equals the")
    print("size of the feasible set by construction, for any n.")
    print()
    print(f"{'n':>5} {'|X|':>7} {'oracle calls':>13} {'visited':>9}"
          f" {'iterations':>11}   equal?")

    for n in (2, 4, 8, 20, 40, 60):
        s = Search(n)
        s.run()
        total = len(feasible_solutions(n))
        print(f"{n:>5} {total:>7} {s.oracle_calls:>13} {len(s.visited):>9}"
              f" {s.iterations:>11}   {s.oracle_calls == total}")

    print()
    print("Consequences:")
    print("  * 'Oracle calls' is not a sampling budget; there is no saving over")
    print("    brute force.")
    print("  * The ranking, adaptive-K and diversification machinery only")
    print("    reorders the enumeration, it never truncates it.")
    print()
    print("Ranking makes that enumeration more expensive than brute force,")
    print("because frontier_score() is recomputed for every unvisited point on")
    print("every iteration:")
    print()

    for n in (20, 40, 80):
        t0 = time.perf_counter()
        Search(n).run()
        t_search = time.perf_counter() - t0

        t0 = time.perf_counter()
        exact_pareto(n)
        t_brute = time.perf_counter() - t0

        print(f"  n = {n:3d}   Search.run() = {t_search:7.3f}s"
              f"   exact_pareto() = {t_brute:7.3f}s")

    print()
    print("Two pieces of that machinery are also inert:")
    print("  * rank(): points are appended in raw-score order and the loop")
    print("    breaks at k, so selection is plain top-k. The 0.25 * d diversity")
    print("    bonus is added after a point is already selected, and only")
    print("    reorders a batch whose members all get evaluated anyway.")
    print("  * run(): on a successful batch, K = max(3, min(K, 24)) leaves K")
    print("    unchanged for every reachable K, so K never decreases.")


# ------------------------------------------------------------------
# 3. Restricting Dinkelbach to the Pareto set is not generally valid.
# ------------------------------------------------------------------

def issue_3_ratio_over_pareto_set():
    rule("3. The ratio optimum need not lie on the Pareto front")

    print("N(f) = 2*f1 + f2       is increasing in f1 and in f2")
    print("D(f) = 0.5*f1 - 0.1*f2 + 1   is increasing in f1, decreasing in f2")
    print()
    print("So N/D is increasing in f2 but not monotone in f1: a point can be")
    print("dominated and still have a strictly better ratio. Running Dinkelbach")
    print("over the Pareto archive instead of the feasible set is therefore an")
    print("assumption, not an identity.")
    print()
    print("It happens to hold for this benchmark (the front is one point, and")
    print("it wins). It fails as soon as the objectives genuinely conflict.")
    print()

    import random

    rng = random.Random(0)
    n = 6
    bad = 0
    tested = 0
    shown = False

    for _ in range(2000):
        c = [rng.randint(1, 6) for _ in range(4)]
        pts = [
            (
                x,
                np.array(
                    [
                        c[0] * x[0] + c[1] * x[1],
                        c[2] * x[0] + c[3] * x[1],
                    ],
                    dtype=float,
                ),
            )
            for x in feasible_solutions(n)
        ]

        # Keep the Dinkelbach transform well posed.
        if any(D_value(f) <= 0.0 for _, f in pts):
            continue

        tested += 1

        front = [
            (x, f)
            for x, f in pts
            if not any(dominates(g, f) for _, g in pts)
        ]

        best_all = max(pts, key=lambda p: N_value(p[1]) / D_value(p[1]))
        best_front = max(front, key=lambda p: N_value(p[1]) / D_value(p[1]))

        phi_all = N_value(best_all[1]) / D_value(best_all[1])
        phi_front = N_value(best_front[1]) / D_value(best_front[1])

        if phi_all - phi_front > 1e-9:
            bad += 1

            if not shown:
                shown = True
                print(f"  counterexample: f1 = {c[0]}*x1 + {c[1]}*x2,"
                      f"  f2 = {c[2]}*x1 + {c[3]}*x2,  n = {n}")
                print(f"    argmax over feasible set : x = {best_all[0]}"
                      f"  F = {best_all[1]}  Phi = {phi_all:.10f}")
                print(f"    argmax over Pareto front : x = {best_front[0]}"
                      f"  F = {best_front[1]}  Phi = {phi_front:.10f}")
                print()

    print(f"  linear models sampled (D > 0 everywhere) : {tested}")
    print(f"  models where the Pareto restriction loses : {bad}")

    print()
    print("Note in the benchmark's favour: for F(x) = (2x1+2x2, x1+4x2),")
    print("D(F(x)) = 0.9*x1 + 0.6*x2 + 1 > 0 on the whole feasible set, so the")
    print("Dinkelbach transform itself is well posed here.")


if __name__ == "__main__":
    issue_1_degenerate_front()
    issue_2_exhaustive_search()
    issue_3_ratio_over_pareto_set()
    print()
