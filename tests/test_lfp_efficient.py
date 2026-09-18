"""Tests: the paper's example, the building blocks, and randomised validation.

Run with ``python3 tests/test_lfp_efficient.py`` (no pytest required) or with
``python3 -m pytest tests/``.
"""

import random
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lfp_efficient import (FractionalObjective, LE, MOILP, Model,
                           add_sylva_crema_cut, alternative_optima_columns,
                           best_with_same_criterion, enumerate_efficient_set,
                           lower_bounds, optimize_over_efficient_set,
                           solve_fractional_milp, solve_linear_milp,
                           solve_relaxation, test_efficiency)
from lfp_efficient.rational import F, fmt


def paper_problem():
    D = (Model(2)
         .add([-2, 1], LE, 0)
         .add([6, 1], LE, 21)
         .add([-2, 4], LE, 6))
    return MOILP(D, [[1, -2], [-1, 4]]), FractionalObjective([1, 1], [5, 1], -1, -1)


# --------------------------------------------------------------------------
def test_feasible_and_efficient_sets_of_the_paper():
    """Section 4: D has 11 feasible points, 7 of them efficient."""
    problem, _ = paper_problem()
    enum = enumerate_efficient_set(problem, bounds=[5, 5])
    assert len(enum.feasible) == 11
    expected = {(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3)}
    assert {tuple(int(v) for v in x) for x in enum.efficient} == expected


def test_lower_bounds_match_the_paper():
    """The paper computes M_1 = M_2 = -3."""
    problem, _ = paper_problem()
    assert lower_bounds(problem) == [F(-3), F(-3)]


def test_relaxed_problem_optimum_is_the_origin():
    """Step 0: max Phi over D is attained at x = (0,0) with Phi = 1.

    (0,0) is the single feasible point with a negative denominator, which is
    why the sign-split of ``solve_fractional_milp`` is needed to find it.
    """
    problem, phi = paper_problem()
    res = solve_fractional_milp(problem.model, phi)
    assert [int(v) for v in res.x] == [0, 0]
    assert res.objective == F(1)


def test_efficiency_test_theorem_1():
    """psi* = 0 exactly on the efficient points, and the witness is efficient."""
    problem, _ = paper_problem()
    enum = enumerate_efficient_set(problem, bounds=[5, 5])
    efficient = {tuple(x) for x in enum.efficient}
    for x in enum.feasible:
        res = test_efficiency(problem, x)
        assert res.efficient == (tuple(x) in efficient), x
        if not res.efficient:
            assert res.psi > 0
            assert tuple(res.witness) in efficient          # Ecker & Kouada
            assert problem.dominates(res.witness, x)


def test_origin_is_not_efficient_with_psi_2():
    """Iteration 1 of the paper: the test at (0,0) has optimal value 2."""
    problem, _ = paper_problem()
    res = test_efficiency(problem, [F(0), F(0)])
    assert not res.efficient and res.psi == F(2)


def test_sylva_crema_cut_removes_exactly_the_dominated_slice():
    """After cutting on x^s, the surviving points are those improving some C_i."""
    problem, _ = paper_problem()
    M = lower_bounds(problem)
    enum = enumerate_efficient_set(problem, bounds=[5, 5])
    x_s = [F(2), F(1)]
    region = add_sylva_crema_cut(problem.model, problem, x_s, M)
    c_s = problem.C(x_s)
    for x in enum.feasible:
        survives = any(a > b for a, b in zip(problem.C(x), c_s))
        # a point survives iff it can be completed with binaries y making it
        # feasible for the cut region
        padded = x + [F(0)] * (region.n - len(x))
        ok = False
        for mask in range(1 << problem.p):
            y = [F((mask >> i) & 1) for i in range(problem.p)]
            if region.is_feasible(x + y):
                ok = True
                break
        assert ok == survives, (x, survives, ok)


def test_Q_subproblem_picks_the_best_of_a_criterion_slice():
    """Q(x~) maximises Phi among the points sharing the vector C x~."""
    problem, phi = paper_problem()
    res = best_with_same_criterion(problem.model, problem, [F(2), F(1)], phi)
    assert res.feasible
    assert problem.C(res.x) == problem.C([F(2), F(1)])
    enum = enumerate_efficient_set(problem, bounds=[5, 5])
    slice_ = [x for x in enum.feasible
              if problem.C(x) == problem.C([F(2), F(1)])]
    assert res.objective == max(phi(x) for x in slice_)


def test_reduced_gradient_is_non_positive_at_an_optimum():
    """Theorem 3: x is optimal for the relaxed LFP iff gamma_j <= 0 for all j."""
    from lfp_efficient.edges import reduced_gradient
    problem, phi = paper_problem()
    # restrict to the branch where the denominator is positive so that the
    # continuous LFP is well posed
    region = problem.model.copy().add([5, 1], LE, 100).add([-5, -1], LE, -2)
    relax = solve_relaxation(region, phi)
    gamma = reduced_gradient(relax.tableau, phi)
    assert all(g <= 0 for g in gamma)
    assert all(gamma[j] == 0 for j in relax.tableau.basis)   # basic columns


def test_paper_example_end_to_end():
    """The headline result: X_opt = (3,3), Phi_opt = 5/17."""
    problem, phi = paper_problem()
    sol = optimize_over_efficient_set(problem, phi)
    assert [int(v) for v in sol.x] == [3, 3]
    assert sol.value == Fraction(5, 17)
    # the promise of the method: E(P_D) is never enumerated
    assert len(sol.explored) < 7


def test_edge_exploration_does_not_change_the_optimum():
    problem, phi = paper_problem()
    a = optimize_over_efficient_set(problem, phi, use_edge_exploration=True)
    b = optimize_over_efficient_set(problem, phi, use_edge_exploration=False)
    assert a.value == b.value


def test_minimisation_by_sign_flip():
    """min Phi = -max(-Phi): flipping U and alpha flips the problem."""
    problem, _ = paper_problem()
    neg = FractionalObjective([-1, -1], [5, 1], 1, -1)
    sol = optimize_over_efficient_set(problem, neg)
    enum = enumerate_efficient_set(problem, bounds=[5, 5])
    assert sol.value == max(neg(x) for x in enum.efficient)


# --------------------------------------------------------------------------
def random_instance(rng):
    """A small random bi- or tri-objective integer program on a bounded box."""
    n = rng.choice([2, 3])
    p = rng.choice([2, 3])
    bound = rng.choice([3, 4])
    model = Model(n)
    for j in range(n):                                   # keep D bounded
        row = [0] * n
        row[j] = 1
        model.add(row, LE, bound)
    for _ in range(rng.randint(1, 3)):
        coeffs = [rng.randint(-3, 4) for _ in range(n)]
        model.add(coeffs, LE, rng.randint(2, 12))
    criteria = [[rng.randint(-3, 4) for _ in range(n)] for _ in range(p)]
    # a denominator that stays away from 0 on the box, to keep Phi well defined
    V = [rng.randint(1, 3) for _ in range(n)]
    beta = rng.randint(1, 5)
    U = [rng.randint(-4, 5) for _ in range(n)]
    alpha = rng.randint(-4, 5)
    return (MOILP(model, criteria),
            FractionalObjective(U, V, alpha, beta),
            [bound] * n)


def test_random_instances_against_exhaustive_enumeration(trials=60, seed=20250918):
    """The real safety net: 60 random instances, algorithm vs. brute force."""
    rng = random.Random(seed)
    checked = 0
    for _ in range(trials):
        problem, phi, bounds = random_instance(rng)
        enum = enumerate_efficient_set(problem, bounds)
        if not enum.efficient:
            continue
        expected_x, expected = enum.best(phi)
        sol = optimize_over_efficient_set(problem, phi)
        assert sol.value == expected, (
            f"mismatch: got {fmt(sol.value)} expected {fmt(expected)}")
        # the reported point must itself be efficient and attain the value
        assert tuple(sol.x) in {tuple(x) for x in enum.efficient}
        assert phi(sol.x) == sol.value
        checked += 1
    assert checked >= 40, f"only {checked} usable instances"
    return checked


# --------------------------------------------------------------------------
def main():
    # only the functions defined here -- ``test_efficiency`` imported from the
    # library is a building block, not a test case
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)
             and getattr(v, "__module__", None) == "__main__"]
    failures = 0
    for fn in tests:
        try:
            extra = fn()
            suffix = f" ({extra} instances)" if isinstance(extra, int) else ""
            print(f"  PASS  {fn.__name__}{suffix}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:                          # noqa: BLE001
            failures += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
