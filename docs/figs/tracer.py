"""Run the box search on an instance and record every step, with each box
recovered as an axis-aligned region of criterion space.

Nothing here is invented: the boxes, centres, maximisers and values all come
out of a real run of `optimize_in_criterion_space`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from fractions import Fraction
import lfp_efficient.criterion_space as CS
from lfp_efficient import *
from lfp_efficient.rational import F


def criterion_bounds(problem, rows):
    """Recover ``(lo_k, hi_k)`` for each criterion from a box's rows.

    With linear criteria ``e_k(x; a) = C_k x - C_k a``, so every box row is
    either ``Z_k >= Z_k(a) + 1`` or ``Z_k <= Z_k(a)``; matching a row's
    coefficients against ``+-C_k`` says which.
    """
    p = problem.p
    lo = [None] * p
    hi = [None] * p
    for coeffs, rhs in rows:
        for k in range(p):
            Ck = [F(c) for c in problem.criteria[k].U]
            if list(coeffs) == Ck:                       # C_k x <= rhs
                hi[k] = rhs if hi[k] is None else min(hi[k], rhs)
                break
            if list(coeffs) == [-c for c in Ck]:         # -C_k x <= rhs
                bound = -rhs
                lo[k] = bound if lo[k] is None else max(lo[k], bound)
                break
    return lo, hi


def trace(problem, phi):
    steps = []
    seen_boxes = {}
    real_split = CS.split

    def split(prob, box, centre, bound, root=None):
        kids = real_split(prob, box, centre, bound, root)
        steps[-1]["children"] = [criterion_bounds(prob, k.rows) for k in kids]
        return kids
    CS.split = split

    real_pop = CS.heappop

    def pop(h):
        item = real_pop(h)
        steps.append({"box": criterion_bounds(problem, item[3].rows),
                      "inherited": item[3].bound, "open_after": len(h)})
        return item
    CS.heappop = pop

    real_solve = CS.solve_fractional_milp

    def solve(model, f, **k):
        r = real_solve(model, f, **k)
        steps[-1]["status"] = r.status
        steps[-1]["value"] = r.objective
        steps[-1]["x"] = list(r.x[:problem.n]) if r.x else None
        return r
    CS.solve_fractional_milp = solve

    real_test = CS.test_efficiency

    def test(prob, x, **k):
        r = real_test(prob, x, **k)
        steps[-1]["efficient"] = r.efficient
        steps[-1]["psi"] = r.psi
        return r
    CS.test_efficiency = test

    real_dom = CS.efficient_dominator

    def dom(prob, outcome):
        c = real_dom(prob, outcome)
        steps[-1]["centre"] = list(c)
        return c
    CS.efficient_dominator = dom

    try:
        sol = optimize_in_criterion_space(problem, phi)
    finally:
        CS.split, CS.heappop, CS.solve_fractional_milp = real_split, real_pop, real_solve
        CS.test_efficiency, CS.efficient_dominator = real_test, real_dom

    for s, log in zip(steps, sol.iterations):
        if "centre" not in s and log.efficient_point is not None:
            s["centre"] = list(log.efficient_point)
        s["note"] = log.note
        s["incumbent"] = list(log.incumbent) if log.incumbent else None
        s["incumbent_value"] = log.incumbent_value
        if log.best_same_criterion is not None:
            s["q"] = (list(log.best_same_criterion), log.phi_same_criterion)
    return sol, steps


def enumerate_points(problem, bounds):
    from itertools import product
    feasible, eff = [], []
    for combo in product(*(range(b + 1) for b in bounds)):
        x = [F(v) for v in combo]
        if problem.model.is_feasible(x):
            feasible.append(x)
    for x in feasible:
        if not any(problem.dominates(y, x) for y in feasible):
            eff.append(x)
    return feasible, eff
