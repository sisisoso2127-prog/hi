"""Criterion-space enumeration: measured, and rejected before it was built.

THE IDEA, AND WHY IT LOOKED RIGHT

Every earlier attempt failed against the same wall. The disjunctive cuts add
nothing to the LP bound at any valid M, so the tree grows; making the bound bite
(convex hull) costs more in model size than it saves in nodes; removing the
binaries (explicit branching) is 17-64x worse; and raising psi* prunes nothing,
because the non-efficient points that generate the rounds sit a median 3.74x
ABOVE the efficient optimum, not below it. Every one of those is a consequence
of searching decision space and testing efficiency after the fact.

A criterion-space method never touches a non-efficient point. Enumerate the
nondominated set, run Find-Best on each slice, take the best psi. Box
decomposition does it with a model that never grows: for a box with lower bound
l, maximising sum_i c^i x over {x in D, Cx >= l} returns a globally nondominated
z -- positive weights make the optimum efficient, and anything dominating z also
satisfies Cx >= l, so it would have been picked instead. Split into r boxes with
l_i raised to z_i + 1 and recurse. Every subproblem carries r plain rows and no
binaries. That is exactly the property nothing else had.

WHAT KILLED IT

The enumerator below is exact -- it reproduces brute force on tiny instances,
22, 31, 18 and 15 nondominated points, 4/4.

On the instances that matter (n=20, m=10, r=3, rho=0.12, the ones the current
method finishes in 8-120s using 24-60 rounds and 146-338 subproblems), 90
seconds of enumeration found 273 to 479 nondominated points and had NOT
finished, having solved 2866 to 4861 boxes.

So the nondominated set is in the hundreds at least, while the current algorithm
visits about thirty efficient points before its psi-maximiser happens to be
efficient and it stops. Even a perfectly non-redundant decomposition -- the
naive split here solves roughly ten boxes per point found, and Daechert-style
schemes cut that to O(|ND|) -- still needs at least one subproblem per
nondominated point, so at least 273-479 against the current 146-338, on a count
that had not terminated.

The conclusion inverts the premise this direction started from. Walking the
non-efficient band is not the algorithm's weakness. It is how the algorithm
AVOIDS enumerating a Pareto front hundreds of times larger than the part of it
that matters. psi is what lets it stop early, and a method that ignores psi to
stay in criterion space forfeits exactly that.

A psi-pruned box search is the obvious repair and is not promising either: to
discard a box you need an upper bound on psi over it, that bound is the box's
unconstrained psi maximum, and the same 3.74x band makes it loose. That last
step is reasoning, not measurement -- it is the one thing here not settled by
a number.

    python3 criterion_space.py check     # exactness against brute force
    python3 criterion_space.py count     # the measurement that closed this
"""

import argparse
import itertools
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from warmstart_experiment import generate_instance, RNG_SEED


def _require_fixed_warmstart():
    import inspect
    import warmstart_experiment as _w
    if "rho" not in inspect.signature(_w.generate_instance).parameters:
        raise SystemExit(
            "warmstart_experiment.py here is the ORIGINAL version (no rho).\n"
            "  Replace it with the fixed one before running this.")


_require_fixed_warmstart()


def enumerate_nondominated(inst, cap_boxes=200_000, cap_t=90.0):
    """All nondominated criterion vectors, by box decomposition.

    Returns (complete_count_or_None, found_so_far, boxes_solved, seconds).
    A None first element means the cap stopped it: the count is a lower bound.
    """
    C, r, n, ub = inst["C"], inst["r"], inst["n"], inst["ub"]
    A, b = inst["A"], inst["b"]
    env = gp.Env(params={"OutputFlag": 0})
    m = gp.Model(env=env)
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    x = m.addVars(n, vtype=GRB.INTEGER, lb=0, ub=ub, name="x")
    for i in range(A.shape[0]):
        m.addConstr(gp.quicksum(int(A[i, j]) * x[j] for j in range(n)) <= int(b[i]))
    lo = [m.addConstr(gp.quicksum(int(C[i, j]) * x[j] for j in range(n)) >= 0)
          for i in range(r)]
    m.setObjective(gp.quicksum(float(C[:, j].sum()) * x[j] for j in range(n)),
                   GRB.MAXIMIZE)

    l0 = tuple(int(v) for v in (np.minimum(C, 0).sum(axis=1) * ub))
    stack, seen, Z, boxes = [l0], {l0}, set(), 0
    t0 = time.perf_counter()
    try:
        while stack:
            if boxes >= cap_boxes or time.perf_counter() - t0 > cap_t:
                return None, len(Z), boxes, time.perf_counter() - t0
            l = stack.pop()
            boxes += 1
            for i in range(r):
                lo[i].RHS = float(l[i])
            m.reset()
            m.optimize()
            if m.Status != GRB.OPTIMAL:      # box empty
                continue
            xv = np.array([x[j].X for j in range(n)]).round().astype(np.int64)
            z = tuple(int(v) for v in (C @ xv))
            Z.add(z)
            for i in range(r):               # cover {y >= l} minus {y <= z}
                l2 = list(l)
                l2[i] = z[i] + 1
                l2 = tuple(l2)
                if l2 not in seen:
                    seen.add(l2)
                    stack.append(l2)
        return len(Z), len(Z), boxes, time.perf_counter() - t0
    finally:
        m.dispose()
        env.dispose()


def _brute_nondominated(inst):
    n, ub = inst["n"], inst["ub"]
    G = np.array(list(itertools.product(range(ub + 1), repeat=n)), dtype=np.int64)
    X = G[np.all(G @ inst["A"].T <= inst["b"], axis=1)]
    Z = X @ inst["C"].T
    return {tuple(Z[i]) for i in range(len(Z))
            if not np.any(np.all(Z >= Z[i], axis=1) & np.any(Z > Z[i], axis=1))}


def check(reps=6):
    print("CHECK -- box enumeration against exhaustive enumeration (n=6, ub=3)")
    rng = np.random.default_rng(RNG_SEED)
    ok = 0
    for _ in range(reps):
        inst = generate_instance(rng, n=6, m=2, r=3, ub=3)
        truth = _brute_nondominated(inst)
        got, cnt, boxes, _t = enumerate_nondominated(inst)
        good = got == len(truth)
        ok += good
        print(f"   brute force {len(truth):3d}   enumeration {cnt:3d}   "
              f"{'MATCH' if good else 'MISMATCH'}   boxes={boxes}")
    print("CHECK PASSED" if ok == reps else "CHECK FAILED")


def count(reps=5, rho=0.12, cap_t=90.0):
    print(f"COUNT -- nondominated set size, n=20 m=10 r=3 rho={rho}")
    print("  (the current method finishes these in 8-120s, 24-60 rounds, "
          "146-338 subproblems)\n")
    rng = np.random.default_rng(RNG_SEED)
    for rep in range(reps):
        inst = generate_instance(rng, n=20, m=10, r=3, ub=10, rho=rho)
        got, cnt, boxes, t = enumerate_nondominated(inst, cap_t=cap_t)
        tag = f"{cnt}" if got is not None else f">{cnt} (did not finish)"
        print(f"   rep {rep}: nondominated points = {tag:>22}   "
              f"boxes solved = {boxes:6d}   {t:6.1f}s", flush=True)
    print("\nEven a non-redundant decomposition needs one subproblem per point.")
    print("Enumerating the front costs more than the current method spends in")
    print("total, and these counts are lower bounds.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["check", "count"])
    ap.add_argument("--reps", type=int, default=None)
    a = ap.parse_args()
    kw = {"reps": a.reps} if a.reps else {}
    (check if a.what == "check" else count)(**kw)
