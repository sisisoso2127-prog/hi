"""Tests: the paper's example, the building blocks, and randomised validation.

Run with ``python3 tests/test_lfp_efficient.py`` (no pytest required) or with
``python3 -m pytest tests/``.
"""

import random
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lfp_efficient import (FractionalObjective, LE, MOILFP, MOILP, Model,
                           add_dominance_cut, alternative_optima_columns,
                           clean_tableau_at, edge_direction, max_step_in,
                           random_maximal_point,
                           spread_weights, weighted_sum_efficient,
                           best_over_efficient_set_by_scan, certify_optimum,
                           best_with_same_criterion, enumerate_efficient_set,
                           enumerate_nondominated, lower_bounds,
                           maximize_by_full_enumeration,
                           metaheuristic_incumbent, optimize_hybrid,
                           optimize_hybrid_metaheuristic,
                           optimize_in_criterion_space, pareto_local_search,
                           Box,
                           optimize_over_efficient_set,
                           solve_fractional_milp, solve_linear_milp,
                           solve_relaxation, test_efficiency,
                           augmented_tchebychev_efficient, ideal_point,
                           tchebychev_incumbent, efficient_dominator,
                           repair_to_efficient)
from lfp_efficient.rational import F, fmt
from lfp_efficient.criterion_space import _rows_for
from lfp_efficient.milp import solve_relaxation, warm_relaxation
from lfp_efficient.simplex import (INFEASIBLE, OPTIMAL as LP_OPTIMAL, STALLED,
                                   add_linear_row, restore_feasibility)


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
    enum = enumerate_efficient_set(problem, bounds=[5, 5])
    x_s = [F(2), F(1)]
    region = add_dominance_cut(problem.model, problem, x_s)
    c_s = problem.Z(x_s)
    for x in enum.feasible:
        survives = any(a > b for a, b in zip(problem.Z(x), c_s))
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


def test_clean_tableau_reproduces_the_point_it_was_built_at():
    """The basis of Definition 2 is read at x_l, in the region -- nowhere else.

    Built on a region that already carries a Sylva-Crema cut, so the tableau
    has the cut's binaries and slacks in it, and on a point that is a vertex of
    that region.
    """
    problem, phi = paper_problem()
    region = add_dominance_cut(problem.model.copy(), problem, [F(2), F(1)])
    relaxed = solve_fractional_milp(region, phi)
    assert relaxed.feasible
    tableau = clean_tableau_at(region, relaxed.x)
    assert tableau is not None
    assert tableau.solution()[:region.n] == list(relaxed.x)
    assert len(tableau.basis) == tableau.m == len(set(tableau.basis))
    assert all(v >= 0 for v in tableau.xb)
    # a basis of the region, with none of the bound rows branch & bound needed
    assert tableau.m <= len(region.constraints)


def test_a_non_vertex_has_no_basis():
    """A point in the middle of a face is not a basic solution: no edges there."""
    model = Model(2).add([1, 0], LE, 4).add([0, 1], LE, 4).add([1, 1], LE, 4)
    assert clean_tableau_at(model, [F(1), F(1)]) is None      # strictly interior
    assert clean_tableau_at(model, [F(0), F(0)]) is not None  # a vertex


def test_max_step_in_matches_a_scan_of_the_region():
    """theta0 read from D, checked against walking the ray one unit at a time."""
    problem, _ = paper_problem()
    checked = 0
    for x in ([F(2), F(0)], [F(3), F(0)], [F(2), F(2)]):
        for d in ([F(1), F(0)], [F(0), F(1)], [F(1), F(1)], [F(-1), F(1)]):
            top = max_step_in(problem.model, x, d)
            scan = 0
            while problem.model.is_feasible([a + (scan + 1) * b
                                             for a, b in zip(x, d)]):
                scan += 1
            assert top == scan, (x, d, top, scan)
            checked += 1
    return f"{checked} (point, direction) pairs"


def test_the_edge_step_fires_and_saves_an_iteration():
    """A recorded instance where an edge of Gamma_l carries the optimum.

    The step is rare -- measured, it fires on about one random instance in
    eighty -- so the one case that does fire is pinned here: the candidate must
    be efficient, feasible in D, and score exactly the round's upper bound,
    which is what licenses stopping on it.
    """
    model = Model(5)
    for j in range(5):
        row = [0] * 5
        row[j] = 1
        model.add(row, LE, 3)
    model.add([2, 2, 3, 4, 1], LE, 9)
    model.add([1, 4, 3, 2, 2], LE, 9)
    model.add([4, 4, 4, 2, 2], LE, 12)
    problem = MOILP(model, [[2, 5, 4, 1, 1], [2, 5, 1, 3, 1], [3, 4, 5, 4, 4]])
    phi = FractionalObjective([-4, -5, -4, -2, -3], [1, 1, 1, 2, 1], 28, 7)

    on = optimize_over_efficient_set(problem, phi)
    off = optimize_over_efficient_set(problem, phi, use_edge_exploration=False)
    assert on.value == off.value
    assert len(on.iterations) < len(off.iterations)

    fired = [it for it in on.iterations if it.edge_candidate is not None]
    assert len(fired) == 1
    it = fired[0]
    c = it.edge_candidate
    assert problem.model.is_feasible(c.x)
    assert test_efficiency(problem, c.x).efficient
    assert c.phi == it.upper_bound == on.value
    return f"saved {len(off.iterations) - len(on.iterations)} iteration(s)"


def test_an_edge_of_gamma_keeps_phi_constant():
    """gamma_j = 0 means the ratio does not move along the edge.

    That is the whole licence for the step: a point found there scores exactly
    Phi(x_l), which is the round's upper bound, so being efficient makes it
    globally optimal.
    """
    model = Model(2).add([1, 0], LE, 6).add([0, 1], LE, 6).add([1, 1], LE, 4)
    phi = FractionalObjective([1, 1], [2, 2], 1, 3)     # depends on x1 + x2 only
    base = [F(4), F(0)]
    tableau = clean_tableau_at(model, base)
    assert tableau is not None

    columns = alternative_optima_columns(tableau, phi)
    assert columns, "the edge along x1 + x2 = 4 has a vanishing reduced gradient"
    walked = 0
    for j in columns:
        d = edge_direction(tableau, j, 2)
        if not any(d):
            continue
        for theta in range(1, max_step_in(model, base, d) + 1):
            point = [a + theta * b for a, b in zip(base, d)]
            assert model.is_feasible(point), (j, theta)
            assert phi(point) == phi(base), (j, theta)
            walked += 1
    assert walked == 4, walked        # (3,1), (2,2), (1,3), (0,4)
    return f"{walked} point(s) along the edge, Phi constant on all of them"


def test_every_weighted_sum_maximiser_is_efficient():
    """The theorem the batch option rests on, checked against Definition 1.

    For w > 0 a maximiser of w'Z over D is efficient -- which is why a batch
    centre needs no efficiency test of its own.
    """
    rng = random.Random(20240919)
    checked = 0
    for _ in range(12):
        problem, phi, bounds = random_instance(rng)
        enum = enumerate_efficient_set(problem, bounds=bounds)
        if not enum.efficient:
            continue
        truth = {tuple(x) for x in enum.efficient}
        for w in spread_weights(problem.p) + [[rng.randint(1, 6)
                                               for _ in range(problem.p)]]:
            point = weighted_sum_efficient(problem, w)
            if point is None:
                continue
            assert tuple(point) in truth, (w, point)
            checked += 1
    assert checked > 20, checked
    return f"{checked} scalarisations, every maximiser efficient"


def test_a_zero_weight_is_refused():
    """With w_k = 0 the maximiser is only weakly efficient: the guarantee is
    gone, so the function refuses rather than returning a doubtful point."""
    problem, _ = paper_problem()
    try:
        weighted_sum_efficient(problem, [1, 0])
    except ValueError as exc:
        assert "strictly positive" in str(exc)
    else:
        raise AssertionError("a zero weight was accepted")


def test_every_augmented_tchebychev_optimum_is_efficient():
    """The guarantee the generator rests on, checked against Definition 1.

    For any ``w > 0`` and ``rho > 0`` the optimum of the augmented program is
    efficient -- the augmentation is what rules out the merely *weakly*
    efficient optima the plain Tchebychev program admits.
    """
    rng = random.Random(20250922)
    checked = 0
    for _ in range(12):
        problem, phi, bounds = random_instance(rng)
        enum = enumerate_efficient_set(problem, bounds=bounds)
        if not enum.efficient:
            continue
        truth = {tuple(x) for x in enum.efficient}
        for w in spread_weights(problem.p) + [[rng.randint(1, 6)
                                               for _ in range(problem.p)]]:
            point = augmented_tchebychev_efficient(problem, w)
            if point is None:
                continue
            assert tuple(point) in truth, (w, point)
            checked += 1
    assert checked > 20, checked
    return f"{checked} programs, every optimum efficient"


def test_tchebychev_reaches_efficient_points_no_weighted_sum_can():
    """What the augmented program buys over :func:`weighted_sum_efficient`.

    A weighted sum can only maximise at a *supported* efficient point -- one on
    the convex hull of the criterion image.  On the paper's instance three of
    the seven efficient points are unsupported, and the Tchebychev program
    reaches all three while no positive weighted sum reaches any.
    """
    problem, _ = paper_problem()
    efficient = [tuple(x) for x in
                 enumerate_efficient_set(problem, bounds=[6, 12]).efficient]
    values = {x: tuple(z(list(x)) for z in problem.criteria) for x in efficient}

    supported = set()
    for w1 in range(1, 40):
        for w2 in range(1, 40):
            top = max(w1 * values[x][0] + w2 * values[x][1] for x in efficient)
            supported.update(x for x in efficient
                             if w1 * values[x][0] + w2 * values[x][1] == top)
    unsupported = set(efficient) - supported
    assert unsupported, "the instance was meant to have unsupported points"

    reached = set()
    for w1 in range(1, 8):
        for w2 in range(1, 8):
            point = augmented_tchebychev_efficient(problem, [w1, w2])
            if point is not None:
                reached.add(tuple(point))
    assert unsupported <= reached, sorted(unsupported - reached)
    return (f"{len(unsupported)} unsupported of {len(efficient)} efficient, "
            f"all reached; no weighted sum reaches any")


def test_the_ideal_point_dominates_every_efficient_point():
    """``z*`` is an upper bound on each criterion, and generally attained by no
    single feasible point -- which is what makes it a reference to move away
    from rather than a solution."""
    problem, _ = paper_problem()
    z_star = ideal_point(problem)
    efficient = enumerate_efficient_set(problem, bounds=[6, 12]).efficient
    attained = 0
    for x in efficient:
        z = [c(list(x)) for c in problem.criteria]
        assert all(z[k] <= z_star[k] for k in range(problem.p)), (x, z, z_star)
        attained += all(z[k] == z_star[k] for k in range(problem.p))
    assert attained == 0, "z* turned out to be feasible on this instance"
    return f"z* = {tuple(z_star)}, above all {len(efficient)} efficient points"


def test_tchebychev_refuses_what_would_void_its_guarantee():
    """A zero weight drops a criterion from the ``max`` term and a zero ``rho``
    readmits weakly efficient optima; neither is accepted silently."""
    problem, phi = paper_problem()
    for weights, rho, expected in [([1, 0], Fraction(1, 1000), "strictly positive"),
                                   ([1, 1], Fraction(0), "rho must be positive"),
                                   ([1, 1], Fraction(-1), "rho must be positive")]:
        try:
            augmented_tchebychev_efficient(problem, weights, rho)
        except ValueError as exc:
            assert expected in str(exc), (weights, rho, exc)
        else:
            raise AssertionError(f"accepted weights={weights} rho={rho}")

    # and the incumbent it feeds is a genuine efficient point, not a claim
    point, value = tchebychev_incumbent(problem, phi)
    assert test_efficiency(problem, point).efficient, point
    assert value == phi(point)
    return "zero weight, zero rho and negative rho all refused"


def test_tchebychev_is_refused_on_fractional_criteria():
    """``z*_k - Z_k(x)`` with a ratio is not linear, so the rows of the
    program would not be constraints of an integer linear program."""
    model = Model(2).add([1, 0], LE, 4).add([0, 1], LE, 4).add([1, 1], LE, 5)
    problem = MOILFP(model, [FractionalObjective([1, 0], [1, 1], 1, 2),
                             FractionalObjective([0, 1], [1, 0], 0, 3)])
    for call in (lambda: ideal_point(problem),
                 lambda: augmented_tchebychev_efficient(problem, [1, 1])):
        try:
            call()
        except ValueError as exc:
            assert "linear" in str(exc), exc
        else:
            raise AssertionError("fractional criteria were accepted")
    return "refused, with the reason stated"


def test_the_search_stops_once_the_best_open_bound_is_hopeless():
    """The heap is ordered by inherited bound descending, so the first box whose
    bound the incumbent already beats proves every remaining box worthless.

    Pinned as a mechanism, not as a timing: the run must end with boxes still
    open and unsolved, and still be proved optimal with the same value the
    exhaustive scan gives.
    """
    rng = random.Random(20250922)
    stopped_early = 0
    checked = 0
    for _ in range(14):
        problem, phi, bounds = random_instance(rng)
        best_x, best_value, _, _ = best_over_efficient_set_by_scan(
            problem, phi, bounds)
        if best_x is None:
            continue
        solution = optimize_in_criterion_space(problem, phi)
        assert solution.proved_optimal, "the early exit must keep the proof"
        assert solution.value == best_value, (solution.value, best_value)
        checked += 1
        last = solution.iterations[-1]
        if last.note and "dropped unsolved" in last.note:
            stopped_early += 1
    assert checked >= 10, checked
    assert stopped_early > 0, "the exit never fired on any instance"
    return f"{stopped_early} of {checked} runs ended on a hopeless best bound"


def test_the_early_exit_never_changes_the_answer_when_seeded():
    """Seeding raises the incumbent, which makes the exit fire sooner.  The
    answer must not move -- that is what Proposition 2 of the note requires of
    any seed, and the exit must not weaken it."""
    rng = random.Random(4242)
    agreed = 0
    for _ in range(10):
        problem, phi, bounds = random_instance(rng)
        plain = optimize_in_criterion_space(problem, phi)
        seeded = optimize_hybrid_metaheuristic(problem, phi)
        if plain.x is None:
            continue
        assert plain.proved_optimal and seeded.proved_optimal
        assert plain.value == seeded.value, (plain.value, seeded.value)
        agreed += 1
    assert agreed >= 7, agreed
    return f"{agreed} instances, seeded and unseeded agree and both proved"


def test_a_failed_tests_witness_is_already_efficient_on_linear_criteria():
    """Ecker & Kouada, checked rather than assumed.

    With linear criteria the test maximises a strictly positive weighted sum of
    ``Z`` over ``{ Z(y) >= Z(x*) }``; anything dominating a maximiser lies in
    that region and scores strictly higher, so no maximiser is dominated.
    :func:`efficient_dominator` relies on this to skip an integer program.
    """
    rng = random.Random(90210)
    checked = 0
    for _ in range(14):
        problem, _, bounds = random_instance(rng)
        for _ in range(10):
            point = [F(rng.randint(0, b)) for b in bounds]
            if not problem.model.is_feasible(point):
                continue
            outcome = test_efficiency(problem, point)
            if outcome.efficient:
                continue
            witness = outcome.witness
            assert test_efficiency(problem, witness).efficient, witness
            assert efficient_dominator(problem, outcome) == list(witness)
            assert problem.dominates(witness, point), (witness, point)
            checked += 1
    assert checked > 15, checked
    return f"{checked} witnesses, every one efficient and dominating"


def test_the_dominator_shortcut_agrees_with_the_walk_it_replaces():
    """The shortcut must return what ``repair_to_efficient`` would, or the
    search would cut on a different centre and stop being the same method."""
    rng = random.Random(1337)
    agreed = 0
    for _ in range(12):
        problem, _, bounds = random_instance(rng)
        for _ in range(8):
            point = [F(rng.randint(0, b)) for b in bounds]
            if not problem.model.is_feasible(point):
                continue
            outcome = test_efficiency(problem, point)
            if outcome.efficient:
                continue
            assert (efficient_dominator(problem, outcome)
                    == repair_to_efficient(problem, outcome.witness))
            agreed += 1
    assert agreed > 15, agreed
    return f"{agreed} cases, shortcut == walk"


def test_the_walk_is_still_taken_on_fractional_criteria():
    """With a ratio the objective is no longer monotone in ``Z``, so the
    shortcut must not fire -- it has to fall back to the walk."""
    model = Model(2).add([1, 0], LE, 4).add([0, 1], LE, 4).add([1, 1], LE, 5)
    problem = MOILFP(model, [FractionalObjective([1, 0], [1, 1], 1, 2),
                             FractionalObjective([0, 1], [1, 0], 0, 3)])
    walked = 0
    for combo in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1)]:
        point = [F(v) for v in combo]
        if not model.is_feasible(point):
            continue
        outcome = test_efficiency(problem, point)
        if outcome.efficient:
            continue
        result = efficient_dominator(problem, outcome)
        assert test_efficiency(problem, result).efficient, result
        walked += 1
    assert walked > 0, "no inefficient point to walk from"
    return f"{walked} fractional repairs, each ending efficient"


def test_a_row_added_to_a_solved_tableau_gives_the_cold_answer():
    """``add_linear_row`` is the general form of the branch & bound's own
    ``add_bound_row``; a box differs from its parent by rows of that shape.

    Checked against the only thing that can arbitrate: rebuilding the model
    with the extra row and solving it from scratch.
    """
    rng = random.Random(4711)
    agreed = empty = 0
    for _ in range(120):
        n = rng.choice([2, 3, 4])
        base = Model(n)
        for j in range(n):
            unit = [0] * n
            unit[j] = 1
            base.add(unit, LE, rng.randint(2, 5))
        for _ in range(rng.randint(1, 2)):
            base.add([rng.randint(1, 4) for _ in range(n)], LE, rng.randint(4, 14))
        objective = [rng.randint(-4, 5) for _ in range(n)]
        root = solve_relaxation(base, objective)
        if root.status != LP_OPTIMAL:
            continue

        coeffs = [F(rng.randint(-3, 4)) for _ in range(n)]
        rhs = F(rng.randint(-4, 10))
        cold = solve_relaxation(base.copy().add(coeffs, LE, rhs), objective)

        tableau = root.tableau.clone()
        pricing = add_linear_row(tableau, root.pricing, coeffs, rhs)
        status = restore_feasibility(tableau, pricing, F(0), F(0))
        if status == STALLED:
            continue
        if status == INFEASIBLE:
            assert cold.status == INFEASIBLE
            empty += 1
            continue
        tableau.run(pricing)
        assert cold.status == LP_OPTIMAL
        assert pricing.value(tableau.solution()) == cold.objective
        agreed += 1
    assert agreed > 50, agreed
    return f"{agreed} warm starts exact, {empty} correctly empty"


def test_a_warm_root_equals_the_cold_root_it_replaces():
    """What the box search relies on: starting from the parent's basis and
    adding the child's rows lands on the same relaxation value as solving the
    child's model from scratch."""
    rng = random.Random(20260922)
    agreed = 0
    for _ in range(10):
        problem, phi, bounds = random_instance(rng)
        parent = Box()
        model = parent.restricted(problem.model)
        root = solve_fractional_milp(model, phi.lift(model.n),
                                     denominator_positive=False)
        if root.root is None:
            continue
        point = [F(rng.randint(0, b)) for b in bounds]
        if not problem.model.is_feasible(point):
            continue
        for k in range(problem.p):
            added = _rows_for(problem, point, k)
            child = Box(parent.rows + added, None, root.root, added)
            child_model = child.restricted(problem.model)
            cold = solve_relaxation(child_model, phi.lift(child_model.n))
            warm = warm_relaxation(child_model, phi.lift(child_model.n),
                                   root.root + (added,))
            assert warm.status == cold.status, (warm.status, cold.status)
            if warm.status == LP_OPTIMAL:
                assert warm.objective == cold.objective
            agreed += 1
    assert agreed > 10, agreed
    return f"{agreed} child roots, warm == cold"


def test_batching_is_refused_on_fractional_criteria():
    """A sum of ratios is not a linear objective; the option says so rather
    than silently doing nothing."""
    model = Model(2).add([1, 0], LE, 4).add([0, 1], LE, 4).add([1, 1], LE, 5)
    problem = MOILFP(model, [FractionalObjective([1, 0], [1, 1], 1, 2),
                             FractionalObjective([0, 1], [1, 0], 0, 3)])
    phi = FractionalObjective([1, 1], [1, 2], 0, 3)
    optimize_over_efficient_set(problem, phi)          # fine without batching
    try:
        optimize_over_efficient_set(problem, phi, batch_cuts_after=1)
    except ValueError as exc:
        assert "linear criteria" in str(exc)
    else:
        raise AssertionError("batching was accepted on fractional criteria")


def test_batching_does_not_change_the_optimum():
    """The extra cuts change how the search gets there, never where it lands.

    Every batch centre is efficient and its slice is banked by Q before the
    cut, so the upper bound and the proof of optimality are untouched.
    """
    rng = random.Random(4242)
    compared = batched = 0
    for _ in range(14):
        problem, phi, bounds = random_instance(rng)
        plain = optimize_over_efficient_set(problem, phi)
        for after in (1, 2):
            quick = optimize_over_efficient_set(problem, phi,
                                                batch_cuts_after=after)
            assert quick.value == plain.value, (after, quick.value, plain.value)
            assert quick.proved_optimal == plain.proved_optimal
            if any(it.batched for it in quick.iterations):
                batched += 1
            compared += 1
    assert batched > 0, "no instance ever reached the batching trigger"
    return f"{compared} runs compared, {batched} of them actually batched"


def test_batching_reaches_the_paper_optimum():
    problem, phi = paper_problem()
    sol = optimize_over_efficient_set(problem, phi, batch_cuts_after=1)
    assert [int(v) for v in sol.x] == [3, 3]
    assert sol.value == Fraction(5, 17)
    assert sol.proved_optimal
    # the batch is recorded in the trace, not applied silently
    assert any(it.batched for it in sol.iterations)


def test_criterion_space_solves_the_paper_example():
    problem, phi = paper_problem()
    sol = optimize_in_criterion_space(problem, phi)
    assert [int(v) for v in sol.x] == [3, 3]
    assert sol.value == Fraction(5, 17)
    assert sol.proved_optimal
    assert sol.upper_bound == sol.value


def test_the_box_split_is_a_disjoint_cover():
    """Definition of the split: removing { Z <= Z(a) } leaves exactly the p
    children, each feasible point of the remainder in exactly one of them."""
    from lfp_efficient.criterion_space import Box, split
    problem, _ = paper_problem()
    enum = enumerate_efficient_set(problem, bounds=[5, 5])
    checked = 0
    for centre in enum.feasible:
        children = split(problem, Box(), centre, None)
        assert len(children) == problem.p
        models = [c.restricted(problem.model) for c in children]
        for x in enum.feasible:
            inside = sum(1 for m in models if m.is_feasible(x))
            removed = all(a <= b for a, b in zip(problem.Z(x), problem.Z(centre)))
            # removed by the cut  <=>  in none of the children
            assert (inside == 0) == removed, (centre, x, inside, removed)
            if not removed:
                assert inside == 1, (centre, x, inside)   # and never in two
            checked += 1
    return f"{checked} (centre, point) pairs, cover exact and disjoint"


def test_criterion_space_agrees_with_the_shipped_method():
    rng = random.Random(99001)
    compared = 0
    for _ in range(18):
        problem, phi, bounds = random_instance(rng)
        shipped = optimize_over_efficient_set(problem, phi)
        boxes = optimize_in_criterion_space(problem, phi)
        assert boxes.value == shipped.value, (boxes.value, shipped.value)
        assert boxes.proved_optimal and shipped.proved_optimal
        # and both against the independent scan
        scan = best_over_efficient_set_by_scan(problem, phi, bounds)[1]
        assert boxes.value == scan, (boxes.value, scan)
        compared += 1
    return f"{compared} instances, criterion space == decision space == scan"


def test_criterion_space_handles_fractional_criteria():
    """A box is written with the integer-valued e_k rows, so ratios need no
    special case -- the numeric bounds on Z_k that a naive box would use are
    rationals and the exact '+1' would be lost."""
    rng = random.Random(31337)
    compared = 0
    for _ in range(10):
        problem, phi, bounds = random_moilfp(rng)
        try:
            shipped = optimize_over_efficient_set(problem, phi)
        except (ZeroDivisionError, ValueError):
            continue          # a vanishing denominator: not this test's subject
        boxes = optimize_in_criterion_space(problem, phi)
        assert boxes.value == shipped.value, (boxes.value, shipped.value)
        compared += 1
    assert compared >= 5, compared
    return f"{compared} fully fractional instances agree"


def test_criterion_space_bound_never_cuts_off_the_optimum():
    """Stopped early, the answer stays certified: the bound the open boxes
    guarantee is never below the true optimum."""
    rng = random.Random(5150)
    checked = 0
    for _ in range(6):
        problem, phi, bounds = random_instance(rng)
        truth = optimize_over_efficient_set(problem, phi).value
        for budget in (0.001, 0.02, 0.2):
            sol = optimize_in_criterion_space(problem, phi, time_budget=budget)
            if sol.value is not None:
                assert sol.value <= truth, (sol.value, truth)
            if sol.upper_bound is not None:
                assert sol.upper_bound >= truth, (sol.upper_bound, truth)
            if sol.proved_optimal:
                assert sol.value == truth
            checked += 1
    return f"{checked} runs across 3 budgets, the bound always holds"


def test_hybrid_solves_the_paper_example():
    problem, phi = paper_problem()
    sol = optimize_hybrid(problem, phi)
    assert [int(v) for v in sol.x] == [3, 3]
    assert sol.value == Fraction(5, 17)
    assert sol.proved_optimal


def test_switch_after_is_a_dial_between_the_two_methods():
    """switch_after = 0 is the pure box search; past the longest run it is the
    pure paper method.  Both ends must give the same answer as the middle."""
    problem, phi = paper_problem()
    ends = [optimize_hybrid(problem, phi, switch_after=k) for k in (0, 1, 2, 50)]
    assert {s.value for s in ends} == {Fraction(5, 17)}
    assert all(s.proved_optimal for s in ends)
    # at switch_after = 0 nothing of the paper's loop ran
    assert ends[0].iterations and ends[0].iterations[0].l == 1


def test_hybrid_agrees_with_both_pure_methods_and_the_scan():
    """The handover must not lose an efficient point.  Checked against the two
    methods it is made of AND against the independent scan, since a handover
    that dropped a box would still agree with itself."""
    rng = random.Random(770077)
    compared = 0
    switched = 0
    for _ in range(14):
        problem, phi, bounds = random_instance(rng)
        paper = optimize_over_efficient_set(problem, phi)
        boxes = optimize_in_criterion_space(problem, phi)
        scan = best_over_efficient_set_by_scan(problem, phi, bounds)[1]
        for k in (1, 2):
            hybrid = optimize_hybrid(problem, phi, switch_after=k)
            assert hybrid.value == paper.value == boxes.value == scan, (
                k, hybrid.value, paper.value, boxes.value, scan)
            assert hybrid.proved_optimal
            if len(paper.iterations) > k:
                switched += 1
            compared += 1
    assert switched > 0, "no instance ever reached the handover"
    return f"{compared} runs, {switched} of them actually switched"


def test_the_handover_replays_the_cuts_rather_than_restarting():
    """A cut deletes { Z <= Z(x~) }; removing that same set from a box list is
    what carries it over.  The two must remove exactly the same points."""
    from lfp_efficient.criterion_space import Box, remove_everywhere
    problem, _ = paper_problem()
    enum = enumerate_efficient_set(problem, bounds=[5, 5])
    centres = [[F(2), F(1)], [F(3), F(3)]]

    region = problem.model.copy()
    boxes = [Box()]
    for centre in centres:
        region = add_dominance_cut(region, problem, centre)
        boxes = remove_everywhere(problem, boxes, centre)

    models = [b.restricted(problem.model) for b in boxes]
    checked = 0
    for x in enum.feasible:
        in_boxes = any(m.is_feasible(x) for m in models)
        # the cut region keeps x iff some criterion strictly improves on every
        # centre -- exactly what the boxes keep
        survives = all(any(a > b for a, b in zip(problem.Z(x), problem.Z(c)))
                       for c in centres)
        assert in_boxes == survives, (x, in_boxes, survives)
        checked += 1
    return f"{checked} points, cut and box list remove the same set"


def test_hybrid_handles_fractional_criteria():
    rng = random.Random(24680)
    compared = 0
    for _ in range(8):
        problem, phi, bounds = random_moilfp(rng)
        try:
            paper = optimize_over_efficient_set(problem, phi)
        except (ZeroDivisionError, ValueError):
            continue
        hybrid = optimize_hybrid(problem, phi, switch_after=1)
        assert hybrid.value == paper.value, (hybrid.value, paper.value)
        compared += 1
    assert compared >= 4, compared
    return f"{compared} fractional instances agree"


def test_hybrid_keeps_the_answer_certified_under_a_budget():
    """Stopped in either phase, the incumbent is real and the bound holds."""
    rng = random.Random(13579)
    checked = 0
    for _ in range(5):
        problem, phi, bounds = random_instance(rng)
        truth = optimize_over_efficient_set(problem, phi).value
        for budget in (0.001, 0.05, 0.5):
            sol = optimize_hybrid(problem, phi, switch_after=1,
                                  time_budget=budget)
            if sol.value is not None:
                assert sol.value <= truth, (budget, sol.value, truth)
            if sol.upper_bound is not None:
                assert sol.upper_bound >= truth, (budget, sol.upper_bound, truth)
            if sol.proved_optimal:
                assert sol.value == truth
            checked += 1
    return f"{checked} runs across 3 budgets, the bound always holds"


def test_the_metaheuristic_never_hands_over_an_inefficient_point():
    """The one property the hybrid's correctness rests on.

    The incumbent is used to prune whole boxes.  If it were the value of a
    DOMINATED point it could exceed the best efficient value and prune away the
    true optimum, so every candidate is verified or repaired before it leaves.
    """
    rng = random.Random(8080)
    checked = 0
    for _ in range(14):
        problem, phi, bounds = random_instance(rng)
        found = metaheuristic_incumbent(problem, phi, seed=checked)
        if found is None:
            continue
        point, value = found
        assert problem.model.is_feasible(point), point
        assert test_efficiency(problem, point).efficient, point
        assert phi(point) == value
        truth = best_over_efficient_set_by_scan(problem, phi, bounds)[1]
        assert value <= truth, (value, truth)     # a lower bound, never above
        checked += 1
    assert checked >= 8, checked
    return f"{checked} incumbents, every one efficient and at most the optimum"


def test_the_archive_holds_only_mutually_non_dominated_points():
    rng = random.Random(1212)
    total = 0
    for _ in range(8):
        problem, phi, bounds = random_instance(rng)
        archive = pareto_local_search(problem, phi, seed=total)
        points = archive.points()
        for a in points:
            for b in points:
                if a is b:
                    continue
                assert not problem.dominates(a, b), (a, b)
        total += len(points)
    assert total > 0
    return f"{total} archive points across 8 instances, none dominating another"


def test_a_maximal_point_cannot_be_increased():
    """What the search starts from: feasible, and on the boundary."""
    import random as _random
    from lfp_efficient.metaheuristic import _IntegerModel
    problem, phi = paper_problem()
    fast = _IntegerModel.build(problem, phi)
    assert fast is not None
    rng = _random.Random(4)
    for _ in range(12):
        x = random_maximal_point(problem.model, rng, fast)
        assert x is not None
        assert problem.model.is_feasible([F(v) for v in x])
        for j in range(problem.n):
            up = list(x)
            up[j] += 1
            assert not problem.model.is_feasible([F(v) for v in up]), (x, j)


def test_the_hybrid_metaheuristic_returns_the_same_proved_optimum():
    rng = random.Random(33445)
    compared = 0
    for _ in range(12):
        problem, phi, bounds = random_instance(rng)
        exact = optimize_in_criterion_space(problem, phi)
        hybrid = optimize_hybrid_metaheuristic(problem, phi, seed=compared)
        scan = best_over_efficient_set_by_scan(problem, phi, bounds)[1]
        assert hybrid.value == exact.value == scan, (hybrid.value, exact.value, scan)
        assert hybrid.proved_optimal
        compared += 1
    return f"{compared} instances, hybrid == exact == scan"


def test_the_neighbourhood_contains_swaps_and_they_are_what_matter():
    """From a maximal point a unit step up is infeasible and a step down lowers
    every criterion with non-negative coefficients, so the archive refuses it;
    the swap is what stays on the boundary and moves along the front.

    Checked structurally on the generator, and then on the effect: with unit
    steps alone the archive collapses to a handful of points.
    """
    from lfp_efficient.metaheuristic import _IntegerModel, _neighbours
    problem, phi = paper_problem()
    base = [3, 3]
    moves = [list(y) for y in _neighbours(base, problem.n)]
    units = [y for y in moves if sum(abs(a - b) for a, b in zip(y, base)) == 1]
    swaps = [y for y in moves if sorted(a - b for a, b in zip(y, base)) == [-1, 1]]
    assert len(units) >= 3 and len(swaps) == 2, (units, swaps)

    # the effect, on an instance big enough to have a front to walk
    rng = random.Random(7)
    problem, phi, _ = random_instance(rng)
    fast = _IntegerModel.build(problem, phi)
    start = random_maximal_point(problem.model, rng, fast)
    unit_only = [y for y in _neighbours(start, problem.n)
                 if sum(abs(a - b) for a, b in zip(y, start)) == 1]
    alive = [y for y in unit_only if fast.feasible(y)
             and not problem.dominates([F(v) for v in start], [F(v) for v in y])]
    assert len(alive) < len(unit_only), (start, alive, unit_only)


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
    return f"{checked} instances vs exhaustive enumeration"


def test_certified_gap_closes_exactly_when_optimality_is_proved():
    """With no budget the run proves optimality, so the gap must be exactly 0."""
    problem, phi = paper_problem()
    solution = optimize_over_efficient_set(problem, phi)
    assert solution.proved_optimal
    assert solution.upper_bound == solution.value == Fraction(5, 17)
    assert solution.gap == 0
    assert solution.gap_closed == 1.0


def test_the_bound_never_cuts_off_the_optimum_at_any_budget(seed=8191):
    """The heart of the anytime claim, checked against the true optimum.

    For every instance and every budget: the reported bound must stay at or
    above the true optimum (or the answer could be wrong without saying so),
    the reported solution must be a genuine efficient point at or below it,
    and spending more time must never make the gap worse.
    """
    rng = random.Random(seed)
    checked = 0
    for _ in range(6):
        problem, phi, bounds = larger_random_instance(rng)
        _, true_opt, _, _ = best_over_efficient_set_by_scan(problem, phi, bounds)
        if true_opt is None:
            continue
        previous = None
        for budget in (0.05, 0.3, 2.0):
            sol = optimize_over_efficient_set(problem, phi, time_budget=budget)
            if sol.upper_bound is not None:
                assert sol.upper_bound >= true_opt, (
                    f"bound {fmt(sol.upper_bound)} cuts off the optimum "
                    f"{fmt(true_opt)}")
            if sol.value is not None:
                assert sol.value <= true_opt
                assert test_efficiency(problem, sol.x).efficient, \
                    "the anytime answer is not an efficient point"
            if sol.proved_optimal:
                assert sol.value == true_opt and sol.gap == 0
            if sol.gap is not None:
                if previous is not None:
                    assert sol.gap <= previous, "more time made the gap worse"
                previous = sol.gap
        checked += 1
    assert checked >= 3
    return f"{checked} instances across 3 budgets"


def test_a_budget_does_not_change_a_run_that_fits_inside_it():
    """A budget large enough to finish must give the same answer as no budget."""
    problem, phi = paper_problem()
    unbounded = optimize_over_efficient_set(problem, phi)
    generous = optimize_over_efficient_set(problem, phi, time_budget=120)
    assert generous.value == unbounded.value
    assert generous.proved_optimal and generous.gap == 0


def test_certificate_proves_the_optimum_and_rejects_impostors():
    """``certify_optimum`` must accept the answer and reject anything else.

    The certificate is the only check that survives past the point where the
    feasible region can be enumerated, so it has to be sharp in both
    directions: it proves the real optimum, and it refuses a point that is
    merely efficient, one that is merely good, and one that is not feasible.
    """
    problem, phi = paper_problem()
    solution = optimize_over_efficient_set(problem, phi)

    proof = certify_optimum(problem, phi, solution.x, solution.value, [5, 5])
    assert proof.valid, proof.reason
    assert proof.challengers >= 1                    # (0,0) beats it on Phi

    # efficient, but not optimal: (2,1) has Phi = 1/5 < 5/17
    impostor = [F(2), F(1)]
    assert test_efficiency(problem, impostor).efficient
    rejected = certify_optimum(problem, phi, impostor, phi(impostor), [5, 5])
    assert not rejected.valid

    # feasible and best on Phi, but dominated -- so not admissible at all
    origin = [F(0), F(0)]
    assert not test_efficiency(problem, origin).efficient
    assert not certify_optimum(problem, phi, origin, phi(origin), [5, 5]).valid

    # not even feasible
    assert not certify_optimum(problem, phi, [F(9), F(9)], F(1), [5, 5]).valid


def test_certificate_agrees_with_the_scan_on_random_instances(trials=10, seed=606):
    """On every instance the scan solves, the certificate must prove that answer."""
    rng = random.Random(seed)
    proved = 0
    for _ in range(trials):
        problem, phi, bounds = larger_random_instance(rng)
        ref_x, ref_value, _, _ = best_over_efficient_set_by_scan(problem, phi, bounds)
        if ref_x is None:
            continue
        proof = certify_optimum(problem, phi, ref_x, ref_value, bounds)
        assert proof.valid, proof.reason
        proved += 1
    assert proved >= 5
    return f"{proved} instances proved"


def test_warm_started_branch_and_bound_matches_a_cold_one():
    """The warm start must not change a single answer, only the time taken.

    Every child node is normally re-optimised from its parent's basis.  Here
    the restoration is forced to report ``STALLED`` on every call, which sends
    the branch & bound down its cold fallback -- rebuilding each node as a
    plain model and solving it from scratch.  The two paths must agree on
    every instance, otherwise the warm start is not a pure optimisation.
    """
    import lfp_efficient.milp as milp
    import lfp_efficient.simplex as simplex

    rng = random.Random(97531)
    warm_results, cold_results = [], []
    instances = [paper_problem()] + [random_instance(rng)[:2] for _ in range(8)]

    for problem, phi in instances:
        warm_results.append(optimize_over_efficient_set(problem, phi).value)

    original = milp.restore_feasibility
    milp.restore_feasibility = lambda *a, **k: simplex.STALLED
    try:
        for problem, phi in instances:
            cold_results.append(optimize_over_efficient_set(problem, phi).value)
    finally:
        milp.restore_feasibility = original

    assert warm_results == cold_results, (
        f"warm {[fmt(v) for v in warm_results]} != "
        f"cold {[fmt(v) for v in cold_results]}")
    return f"{len(instances)} instances, warm == cold"


def test_the_three_reference_methods_agree_on_the_paper_example():
    """Box enumeration, non-dominated enumeration and the Phi-ordered scan."""
    problem, phi = paper_problem()
    _, by_box = enumerate_efficient_set(problem, bounds=[5, 5]).best(phi)
    _, by_nd, enumeration = maximize_by_full_enumeration(problem, phi)
    _, by_scan, _, _ = best_over_efficient_set_by_scan(problem, phi, [5, 5])
    assert by_box == by_nd == by_scan == Fraction(5, 17)
    assert len(enumeration) == 7            # seven non-dominated vectors


def larger_random_instance(rng):
    """n in 4..6, p = 3, x_j in 0..3 -- hundreds of feasible points."""
    n = rng.randint(4, 6)
    ub = 3
    model = Model(n)
    for j in range(n):
        unit = [0] * n
        unit[j] = 1
        model.add(unit, LE, ub)
    for _ in range(3):
        model.add([rng.randint(1, 4) for _ in range(n)], LE, rng.randint(8, 16))
    criteria = [[rng.randint(-3, 5) for _ in range(n)] for _ in range(3)]
    phi = FractionalObjective([rng.randint(-4, 6) for _ in range(n)],
                              [rng.randint(1, 3) for _ in range(n)],
                              rng.randint(-3, 6), rng.randint(2, 6))
    return MOILP(model, criteria), phi, [ub] * n


def test_larger_random_instances_against_the_scan(trials=12, seed=4242):
    """Hundreds of feasible points per instance, checked by the Phi-ordered scan.

    The scan shares no code with the algorithm -- no simplex, no cut, no
    efficiency LP -- so the agreement is an independent check, and it stays
    affordable where enumerating E(P_D) would not.
    """
    rng = random.Random(seed)
    sizes = []
    for _ in range(trials):
        problem, phi, bounds = larger_random_instance(rng)
        ref_x, ref_value, n_feasible, _ = best_over_efficient_set_by_scan(
            problem, phi, bounds)
        if ref_x is None:
            continue
        sol = optimize_over_efficient_set(problem, phi)
        assert sol.value == ref_value, (
            f"mismatch on a {n_feasible}-point instance: "
            f"{fmt(sol.value)} vs {fmt(ref_value)}")
        assert phi(sol.x) == sol.value
        assert problem.model.is_feasible(sol.x)
        sizes.append(n_feasible)
    assert sizes, "no usable instance"
    return (f"{len(sizes)} instances, up to {max(sizes)} feasible points "
            f"({sum(sizes) // len(sizes)} on average)")


# --------------------------------------------------------------------------
# The fractional generalisation: every criterion is a ratio
# --------------------------------------------------------------------------
def random_moilfp(rng, n=3, ub=4):
    """A MOILFP whose criteria are genuine ratios, with positive denominators."""
    model = Model(n)
    for j in range(n):
        unit = [0] * n
        unit[j] = 1
        model.add(unit, LE, ub)
    for _ in range(2):
        model.add([rng.randint(1, 4) for _ in range(n)], LE, rng.randint(5, 4 * ub))
    criteria = [FractionalObjective([rng.randint(-3, 5) for _ in range(n)],
                                    [rng.randint(0, 3) for _ in range(n)],
                                    rng.randint(0, 4), rng.randint(1, 4))
                for _ in range(3)]
    phi = FractionalObjective([rng.randint(-4, 5) for _ in range(n)],
                              [rng.randint(0, 3) for _ in range(n)],
                              rng.randint(-2, 5), rng.randint(1, 5))
    return MOILFP(model, criteria), phi, [ub] * n


def _brute_force(problem, bounds):
    """Feasible points and, by Definition 1 applied to the ratios, efficient ones."""
    from itertools import product
    points = []
    for combo in product(*(range(b + 1) for b in bounds)):
        x = [F(v) for v in combo]
        if problem.model.is_feasible(x):
            points.append(x)
    efficient = [x for x in points
                 if not any(problem.dominates(y, x) for y in points)]
    return points, efficient


def test_linear_criteria_are_the_degenerate_fractional_case():
    """``MOILP`` is ``MOILFP`` with unit denominators, and ``e_k`` reduces."""
    problem, _ = paper_problem()
    for z in problem.criteria:
        assert all(v == 0 for v in z.V) and z.beta == 1      # d_k = 0, b_k = 1
    x_bar, y = [F(3), F(3)], [F(2), F(1)]
    for k in range(problem.p):
        coeffs, const = problem.e_row(k, x_bar)
        e = sum((a * b for a, b in zip(coeffs, y)), F(0)) + const
        assert e == problem.Z(y)[k] - problem.Z(x_bar)[k]    # e_k = C_k y - C_k x_bar


def test_fractional_efficiency_test_matches_the_definition(trials=12, seed=31337):
    """``theta = 0`` iff efficient, checked on *every* feasible point."""
    rng = random.Random(seed)
    tested = 0
    for _ in range(trials):
        problem, _, bounds = random_moilfp(rng)
        points, efficient = _brute_force(problem, bounds)
        if not points:
            continue
        for x in points:
            assert test_efficiency(problem, x).efficient == (x in efficient), x
            tested += 1
    assert tested > 200
    return f"{tested} feasible points, every one classified correctly"


def test_fractional_end_to_end_against_brute_force(trials=15, seed=1009):
    """The whole algorithm on ratios, against the definition-level answer."""
    rng = random.Random(seed)
    checked = 0
    for _ in range(trials):
        problem, phi, bounds = random_moilfp(rng)
        _, efficient = _brute_force(problem, bounds)
        if not efficient:
            continue
        try:
            expected = max(phi(x) for x in efficient)
        except ZeroDivisionError:
            continue
        solution = optimize_over_efficient_set(problem, phi)
        assert solution.value == expected, (
            f"got {fmt(solution.value)} expected {fmt(expected)}")
        assert test_efficiency(problem, solution.x).efficient
        checked += 1
    assert checked >= 8
    return f"{checked} fractional instances"


def test_ecker_kouada_does_not_transfer_to_ratios(seed=2024):
    """Why :func:`repair_to_efficient` exists, stated as a measurement.

    With linear criteria the maximiser of the efficiency test is itself
    efficient, so the dominance chain is always one step. With fractional
    criteria the k-th term of the test's objective carries a factor ``D_k(x)``
    that varies from point to point, so the maximiser only *dominates* -- and
    the chain is measurably longer. Reusing the linear result on ratios would
    therefore be wrong, silently, on a sizeable share of the points.
    """
    from collections import Counter

    def chain_lengths(problem, bounds):
        points, efficient = _brute_force(problem, bounds)
        lengths = []
        for x in points:
            if x in efficient:
                continue
            current, steps = x, 0
            while True:
                outcome = test_efficiency(problem, current)
                if outcome.efficient:
                    break
                current, steps = outcome.witness, steps + 1
            lengths.append(steps)
        return lengths

    rng = random.Random(seed)
    linear = []
    for _ in range(12):
        model = Model(3)
        for j in range(3):
            unit = [0, 0, 0]
            unit[j] = 1
            model.add(unit, LE, 4)
        for _ in range(2):
            model.add([rng.randint(1, 4) for _ in range(3)], LE, rng.randint(5, 16))
        linear += chain_lengths(
            MOILP(model, [[rng.randint(-3, 5) for _ in range(3)] for _ in range(3)]),
            [4, 4, 4])

    rng = random.Random(seed)
    fractional = []
    for _ in range(12):
        problem, _, bounds = random_moilfp(rng)
        fractional += chain_lengths(problem, bounds)

    assert linear and set(linear) == {1}, (
        f"linear chains should all be one step, saw {sorted(Counter(linear))}")
    assert any(length > 1 for length in fractional), (
        "no fractional chain exceeded one step -- the sample is too small to "
        "support the claim that the repair is needed")
    longer = sum(1 for length in fractional if length > 1)
    return (f"linear: {len(linear)} chains, all length 1 | "
            f"fractional: {longer}/{len(fractional)} longer than 1")


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
            suffix = f"  [{extra}]" if isinstance(extra, str) else ""
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
