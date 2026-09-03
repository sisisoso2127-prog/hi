"""Exact-metaheuristic hybrid: which generator, and how good -- measured.

THE TARGET IS NOT THE EXACT OPTIMUM FASTER

Six directions in this repository failed to accelerate the exact method, each
against a different structural wall (see diagnose_cost.py, hull_experiment.py,
rounds_experiment.py, criterion_space.py). Together they are the case FOR a
hybrid rather than a failure: the exact method cannot be accelerated inside its
own paradigm, so the useful question becomes quality per unit time.

Both hybrids here share the same exact half. A generator proposes efficient
points; find_best() then returns the EXACT best psi on that point's criterion
slice. Both run on models that carry no disjunctive cuts and never grow -- the
property every failed direction lacked. They differ only in the generator:

  WEIGHTED -- maximise sum_i lam_i c^i x for random lam > 0. Cheap and simple.
  EPSILON  -- box search: maximise sum_i c^i x over {Cx >= l}, then split the
              box by raising l_i past the point found.

WHAT THIS FOUND

Both hybrids reach 90-95% of the exact optimum on a 0.5s budget, against 32-55s
for the exact method -- a 64x speedup for roughly a tenth of the objective. On a
0.1s budget they are at 59-77%, so the useful operating point is around half a
second on these instances.

Which generator is better is NOT settled here and this script should not be read
as settling it. Only two of six instances gave an exact reference at all, and on
those two weighted led at 0.5s, 95.1% against 91.1%. Two instances is nothing.

A CORRECTION WORTH KEEPING, because it is the trap this measurement is built to
avoid. An earlier run compared the generators at a fixed NUMBER of draws (2, 8,
32) rather than a fixed time, and weighted appeared to saturate at 43-53% and
stop. That looked exactly like a generator reaching only SUPPORTED efficient
points -- those on the convex hull of the criterion set, which no weighted sum
can see past -- and it is a real phenomenon, so the explanation was plausible.
It was still wrong here. Given equal TIME rather than equal draws, weighted runs
220-298 draws where epsilon manages 36-76, because its subproblem is one MILP
with no added rows while a box step adds rows and a find_best call; and at 298
draws the instance that had been stuck at 53% reaches 90%. Compare generators at
equal time, never at equal iterations, or the cheaper one is silently penalised.

THE CASE THE THESIS ACTUALLY RESTS ON.  Three of six instances here, the exact
method did not finish inside 200s and there is no reference to score against.
That is the practical situation the hybrid exists for: where enumeration of the
efficient set is not available, a hybrid still returns a good solution in half a
second. Quality there can only be bounded, not measured -- which makes a
certified gap the next thing to build, not a nicety.

KNOWN HEADROOM.  The box order in epsilon is arbitrary (depth-first off a stack)
and psi never steers it; weighted draws its weights uniformly and psi never
steers those either. Neither generator uses the objective it is optimising. That
is where a metaheuristic layer belongs, and it is a projection, not a
measurement.

    python3 hybrid_feasibility.py compare      # both generators vs the exact optimum
"""

import argparse
import collections
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from warmstart_experiment import Method, generate_instance, RNG_SEED


def _require_fixed_warmstart():
    import inspect
    import warmstart_experiment as _w
    if "rho" not in inspect.signature(_w.generate_instance).parameters:
        raise SystemExit("warmstart_experiment.py here is the ORIGINAL version "
                         "(no rho). Replace it with the fixed one.")


_require_fixed_warmstart()

CHECKPOINTS = (0.1, 0.5, 2.0, 10.0)


def weighted(inst, budget, seed=0):
    """Random positive weighted sums: efficient, but only supported ones."""
    s = Method(inst, time_limit=30, wall_budget=600)
    rng = np.random.default_rng(seed)
    C = inst["C"]
    curve, best, npts, ci = {}, -np.inf, 0, 0
    t0 = time.perf_counter()
    try:
        mw, xw = s._new_model()
        while True:
            el = time.perf_counter() - t0
            while ci < len(CHECKPOINTS) and el >= CHECKPOINTS[ci]:
                curve[CHECKPOINTS[ci]] = (best, npts)
                ci += 1
            if el > budget:
                break
            lam = rng.random(s.r) + 0.05
            mw.setObjective(gp.quicksum(
                float(sum(lam[i] * C[i, j] for i in range(s.r))) * xw[j]
                for j in range(s.n)), GRB.MAXIMIZE)
            mw.reset()
            xs = s._run(mw)
            if xs is None:
                break
            _xb, ps = s.find_best(xs)
            npts += 1
            best = max(best, ps)
        mw.dispose()
    finally:
        s.close()
    for c in CHECKPOINTS:
        curve.setdefault(c, (best, npts))
    return curve


def epsilon(inst, budget, seed=0):
    """Box search: reaches unsupported efficient points too."""
    s = Method(inst, time_limit=30, wall_budget=600)
    C, r, n, ub = inst["C"], inst["r"], inst["n"], inst["ub"]
    curve, best, npts, ci = {}, -np.inf, 0, 0
    t0 = time.perf_counter()
    try:
        m, x = s._new_model()
        lo = [m.addConstr(gp.quicksum(int(C[i, j]) * x[j] for j in range(n)) >= 0)
              for i in range(r)]
        m.setObjective(gp.quicksum(float(C[:, j].sum()) * x[j] for j in range(n)),
                       GRB.MAXIMIZE)
        l0 = tuple(int(v) for v in (np.minimum(C, 0).sum(axis=1) * ub))
        stack, seen = [l0], {l0}
        while stack:
            el = time.perf_counter() - t0
            while ci < len(CHECKPOINTS) and el >= CHECKPOINTS[ci]:
                curve[CHECKPOINTS[ci]] = (best, npts)
                ci += 1
            if el > budget:
                break
            l = stack.pop()
            for i in range(r):
                lo[i].RHS = float(l[i])
            m.reset()
            m.optimize()
            if m.Status != GRB.OPTIMAL:
                continue
            xv = np.array([x[j].X for j in range(n)]).round().astype(np.int64)
            _xb, ps = s.find_best(xv)          # exact best psi on this slice
            npts += 1
            best = max(best, ps)
            z = C @ xv
            for i in range(r):
                l2 = list(l)
                l2[i] = int(z[i]) + 1
                l2 = tuple(l2)
                if l2 not in seen:
                    seen.add(l2)
                    stack.append(l2)
        m.dispose()
    finally:
        s.close()
    for c in CHECKPOINTS:
        curve.setdefault(c, (best, npts))
    return curve


GENS = (("weighted", weighted), ("epsilon", epsilon))


def compare(reps=6, rho=0.12, budget=10.0, exact_budget=200.0):
    print(f"COMPARE -- n=20 m=10 r=3 rho={rho}, both generators against the "
          f"exact optimum\n")
    rng = np.random.default_rng(RNG_SEED)
    agg = collections.defaultdict(list)
    for rep in range(reps):
        inst = generate_instance(rng, n=20, m=10, r=3, ub=10, rho=rho)
        s = Method(inst, time_limit=60, wall_budget=exact_budget)
        try:
            t0 = time.perf_counter()
            ex = s.solve()
            te = time.perf_counter() - t0
        finally:
            s.close()
        if ex["truncated"] or ex["rounds_exhausted"] or not np.isfinite(ex["psi"]):
            print(f"  rep {rep}: exact did not finish -- no reference, skipped",
                  flush=True)
            continue
        print(f"  rep {rep}: exact psi={ex['psi']:.5f} in {te:6.1f}s")
        for name, fn in GENS:
            cur = fn(inst, budget, seed=1000 + rep)
            row = "      " + f"{name:<9}"
            for c in CHECKPOINTS:
                b, npts = cur[c]
                q = 100 * b / ex["psi"] if np.isfinite(b) else 0.0
                if q > 100.0 + 1e-6:
                    print(f"      !! {name} exceeded the exact optimum -- "
                          f"one of the two is wrong, stop and report this")
                row += f"  {c}s:{q:6.1f}%({npts:3d}pts)"
                agg[(name, c)].append((q, te))
            print(row, flush=True)

    if not agg:
        print("\nno instance where the exact method finished; lower rho")
        return
    print("\n" + "=" * 74)
    print(f"{'generator':<11}{'budget':>9}{'quality':>11}{'exact':>10}{'speedup':>10}")
    for name, _fn in GENS:
        for c in CHECKPOINTS:
            v = agg.get((name, c))
            if not v:
                continue
            qs = [a for a, _ in v]
            es = [b for _, b in v]
            print(f"{name:<11}{c:>8.1f}s{np.median(qs):>10.1f}%"
                  f"{np.median(es):>9.1f}s{np.median(es)/c:>9.0f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["compare"], nargs="?", default="compare")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--rho", type=float, default=0.12)
    ap.add_argument("--budget", type=float, default=10.0)
    a = ap.parse_args()
    compare(a.reps, a.rho, a.budget)
