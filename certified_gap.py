"""Anytime optimisation over the efficient set, with a certified gap.

WHERE THE BOUND COMES FROM -- IT IS ALREADY BEING COMPUTED AND THROWN AWAY

Write E for the efficient set and psi for the fractional objective. After the
exact method has cut on xbar_1..xbar_k, every efficient point is either

  * inside one of the removed boxes {x : Cx <= C xbar_i}, where it is either
    dominated (so not efficient) or lies on the slice {Cx = C xbar_i}, whose
    best psi Find-Best has ALREADY computed exactly and folded into psi*; or
  * inside the residual region R_k, which is what dinkelbach() searches.

So      max_E psi  =  max( psi*,  max{ psi(x) : x in E and x in R_k } )
                   <= max( psi*,  max{ psi(x) : x in R_k } )

and that second term is exactly the psi0 the loop already computes every round
and uses only to decide whether to stop. It is a valid UPPER bound on the
answer, free at every round, and it descends as cuts accumulate. If R_k comes
back infeasible, nothing unexplored beats psi*, the bound equals psi* and
optimality is proved.

That turns Algorithm 1 into an anytime method: stop whenever, report

    LB = psi* (attained at a known efficient point, so a real solution)
    UB = the bound above
    gap = UB - LB, reported ABSOLUTE and as a fraction of the starting gap

which is the exact-exact half of the hybrid. The exact-metaheuristic half feeds
the heuristic's psi in as the starting LB: hybrid_feasibility.py reaches 90-95%
of the optimum in half a second, and that number becomes the primal side of a
certified gap instead of an unquantified guess.

Note, from rounds_experiment.py: a higher starting psi* does NOT accelerate the
descent of UB. The constraint psi >= psi* removes low-psi points, and the bound
is a maximum, so it is untouched. The two halves move independently -- the
heuristic raises LB, the cut loop lowers UB -- and the gap closes from both
sides. Any claim that seeding speeds up the exact side would be wrong.

WHAT THIS MEASURES (n=20, m=10, r=3, rho=0.12, 0.5s of heuristic then the loop)

  gap closed   instances   median time
        25%        6/6         0.68s
        50%        6/6         1.41s
        75%        4/6         4.58s
        90%        4/6        21.10s
       100%        3/6        41.62s     (proved optimal)

Median 97.2% of the initial uncertainty eliminated. The shape is the one that
justifies a matheuristic: half the gap goes in the first second and a half, the
last tenth costs twenty seconds or never arrives. Two instances stall at 70-75%.

VALIDATED: 25/25 instances, the bound never fell below the true optimum at ANY
round, and LB was proved optimal at termination -- checked against exhaustive
enumeration, with and without a heuristic starting LB.

A MEASURE THAT HAD TO BE FIXED. The first version reported (UB-LB)/|LB|. psi is
a ratio that can be negative or near zero, and on an instance with LB=-0.099 a
real absolute gap of 3.56 was printed as "3599%", which reads as a broken method
rather than a broken metric -- that instance had in fact eliminated 74.7% of its
uncertainty. Relative-to-incumbent gaps are standard in MIP because objectives
there are usually bounded away from zero. Do not carry that habit over to a
fractional objective over an efficient set.

THE LIMIT, AND WHERE THE NEXT RESULT IS. UB is the psi maximum over the residual
region, so it counts NON-EFFICIENT points, and diagnose-level measurement put
the unconstrained psi maximum a median 3.74x above the efficient optimum. The
bound must grind that band away one dominated box at a time, which is why the
last tenth is expensive and why two instances stall. The primal side dodges the
band; the dual side has to walk through it. A bound that excludes non-efficient
points without enumerating them is the open question, and it is the one worth a
chapter.

    python3 certified_gap.py validate     # bound never cuts off the optimum
    python3 certified_gap.py trace        # how fast does the gap close
"""

import argparse
import time

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from warmstart_experiment import (Method, generate_instance, brute_force,
                                  RNG_SEED, EPS)


def _require_fixed_warmstart():
    import inspect
    import warmstart_experiment as _w
    if "wall_budget" not in inspect.signature(_w.Method.__init__).parameters:
        raise SystemExit("warmstart_experiment.py here is the ORIGINAL version. "
                         "Replace it with the fixed one.")


_require_fixed_warmstart()


def abs_gap(lb, ub):
    """Absolute gap.  psi is a ratio that may be negative or near zero, so a
    gap relative to |LB| is meaningless here -- an LB of -0.099 turned a real
    absolute gap of 3.56 into '3599%'.  Report the absolute gap, and scale it
    by the gap the method STARTED with (see closed())."""
    if not np.isfinite(lb) or not np.isfinite(ub):
        return np.inf
    return ub - lb


def closed(g0, g):
    """Fraction of the initial uncertainty eliminated: 1 means proved optimal."""
    if not np.isfinite(g0) or g0 <= 1e-12:
        return 1.0
    if not np.isfinite(g):
        return 0.0
    return max(0.0, min(1.0, 1.0 - g / g0))


class AnytimeMethod(Method):
    """Algorithm 1 reporting a certified (LB, UB) at every round."""

    def _set_bound(self, bound, psi_star):
        p, al, q, be = (self.I["p"], self.I["alpha"], self.I["q"], self.I["beta"])
        if bound is not None:
            self.mR.remove(bound)
        b = self.mR.addConstr(
            gp.quicksum((float(p[j]) - psi_star * float(q[j])) * self.xR[j]
                        for j in range(self.n)) >= psi_star * be - al)
        self.mR.update()
        return b

    def solve_anytime(self, max_rounds=200, budget=None, gap_target=0.0,
                      seed_lb=None):
        t0 = time.perf_counter()
        LB = -np.inf if seed_lb is None else float(seed_lb)
        UB = np.inf
        x_star = None
        bound = self._set_bound(None, LB) if np.isfinite(LB) else None
        traj = []
        proved = False

        def record(rounds):
            traj.append(dict(t=time.perf_counter() - t0, rounds=rounds,
                             cuts=self.cuts, milps=self.milps,
                             LB=LB, UB=UB, gap=abs_gap(LB, UB)))

        rounds = 0
        for _ in range(max_rounds):
            if budget is not None and time.perf_counter() - t0 > budget:
                break
            rounds += 1
            x0, psi0 = self.dinkelbach()
            if self.truncated:
                break
            if x0 is None:                 # residual empty: nothing beats LB
                UB = LB if np.isfinite(LB) else UB
                proved = np.isfinite(LB)
                record(rounds)
                break
            UB = min(UB, max(LB, psi0)) if np.isfinite(LB) else min(UB, psi0)
            if abs_gap(LB, UB) <= gap_target:
                record(rounds)
                proved = gap_target <= 0.0
                break
            eff, xp = self.efficiency_test(x0)
            if eff is None:
                break
            if eff:                        # x0 is efficient and tops the region
                if psi0 > LB:
                    LB, x_star = psi0, x0
                UB = LB
                proved = True
                record(rounds)
                break
            xs, ps = self.find_best(xp)
            if self.truncated:
                break
            if ps > LB:
                LB, x_star = ps, xs
                bound = self._set_bound(bound, LB)
            self.add_cut(xp)
            record(rounds)
        else:
            self.rounds_exhausted = True

        if not traj:
            record(rounds)
        return dict(LB=LB, UB=UB, gap=abs_gap(LB, UB), x=x_star, proved=proved,
                    rounds=rounds, cuts=self.cuts, milps=self.milps,
                    t=time.perf_counter() - t0, truncated=self.truncated,
                    rounds_exhausted=self.rounds_exhausted, traj=traj)


def heuristic_lb(inst, budget=0.5, seed=0):
    """Weighted-sum generation + exact per-slice psi; the primal half."""
    s = Method(inst, time_limit=30, wall_budget=600)
    rng = np.random.default_rng(seed)
    C = inst["C"]
    best = -np.inf
    t0 = time.perf_counter()
    try:
        mw, xw = s._new_model()
        while time.perf_counter() - t0 < budget:
            lam = rng.random(s.r) + 0.05
            mw.setObjective(gp.quicksum(
                float(sum(lam[i] * C[i, j] for i in range(s.r))) * xw[j]
                for j in range(s.n)), GRB.MAXIMIZE)
            mw.reset()
            xs = s._run(mw)
            if xs is None:
                break
            _xb, ps = s.find_best(xs)
            best = max(best, ps)
        mw.dispose()
    finally:
        s.close()
    return best, time.perf_counter() - t0


# ------------------------------------------------------------------ validate
def validate(reps=25):
    """The bound must never fall below the true optimum, at ANY round."""
    print("VALIDATE -- UB >= true optimum >= LB at every round (n=6, ub=3)")
    rng = np.random.default_rng(RNG_SEED)
    ok = tot = bad = 0
    for _ in range(reps):
        inst = generate_instance(rng, n=6, m=2, r=3, ub=3)
        truth = brute_force(inst)
        if truth is None:
            continue
        tot += 1
        for seed_lb in (None, "heur"):
            lb0 = None
            if seed_lb == "heur":
                lb0, _t = heuristic_lb(inst, budget=0.05, seed=7)
                if not np.isfinite(lb0):
                    lb0 = None
            s = AnytimeMethod(inst, time_limit=30, wall_budget=120)
            try:
                res = s.solve_anytime(seed_lb=lb0)
            finally:
                s.close()
            if res["truncated"] or res["rounds_exhausted"]:
                print("  TRUNCATED -- proves nothing")
                bad += 1
                break
            for row in res["traj"]:
                if np.isfinite(row["UB"]) and row["UB"] < truth - 1e-6:
                    print(f"  BOUND VIOLATED: UB={row['UB']:.6f} < optimum="
                          f"{truth:.6f} at round {row['rounds']}")
                    bad += 1
                    break
                if np.isfinite(row["LB"]) and row["LB"] > truth + 1e-6:
                    print(f"  LB EXCEEDS OPTIMUM: {row['LB']:.6f} > {truth:.6f}")
                    bad += 1
                    break
            else:
                if not res["proved"] or abs(res["LB"] - truth) > 1e-6:
                    print(f"  NOT PROVED OPTIMAL: LB={res['LB']} truth={truth}")
                    bad += 1
                    break
                continue
            break
        else:
            ok += 1
    print(f"  {ok}/{tot} instances: bound valid at every round, "
          f"LB proved optimal at the end")
    print("VALIDATION PASSED" if ok == tot and tot > 0 and bad == 0
          else "VALIDATION FAILED")


# --------------------------------------------------------------------- trace
# fraction of the initial gap that has been eliminated
CLOSED_AT = (0.25, 0.50, 0.75, 0.90, 0.99, 1.00)


def trace(reps=6, rho=0.12, budget=60.0, heur=0.5):
    print(f"TRACE -- how fast the certified gap closes (n=20 m=10 r=3 rho={rho})\n")
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for rep in range(reps):
        inst = generate_instance(rng, n=20, m=10, r=3, ub=10, rho=rho)
        lb0, th = heuristic_lb(inst, budget=heur, seed=1000 + rep)
        s = AnytimeMethod(inst, time_limit=30, wall_budget=budget)
        try:
            res = s.solve_anytime(budget=budget,
                                  seed_lb=lb0 if np.isfinite(lb0) else None)
        finally:
            s.close()
        g0 = res["traj"][0]["gap"] if res["traj"] else np.inf
        hit = {}
        for row in res["traj"]:
            for f in CLOSED_AT:
                if f not in hit and closed(g0, row["gap"]) >= f - 1e-12:
                    hit[f] = row["t"] + th
        print(f"  rep {rep}: LB={lb0:8.5f} (heur {th:.2f}s)  UB={res['UB']:8.5f}"
              f"  gap {res['gap']:8.4f} of initial {g0:8.4f}"
              f"  -> {100*closed(g0, res['gap']):5.1f}% closed"
              f"  [{res['rounds']:3d} rounds / {res['t']:5.1f}s]"
              + ("  PROVED OPTIMAL" if res["proved"] else ""), flush=True)
        print("      time to close: " + "  ".join(
            f"{int(100*f)}%:" + (f"{hit[f]:.2f}s" if f in hit else "--")
            for f in CLOSED_AT), flush=True)
        rows.append(dict(rep=rep, lb=lb0, t_heur=th, gap0=g0,
                         final_gap=res["gap"], closed=closed(g0, res["gap"]),
                         rounds=res["rounds"], t=res["t"], proved=res["proved"],
                         **{f"t_closed_{int(100*f)}": hit.get(f, np.nan)
                            for f in CLOSED_AT}))

    df = pd.DataFrame(rows)
    df.to_csv("certified_gap.csv", index=False)
    print("\n" + "=" * 74)
    print(f"  proved optimal within budget: {int(df.proved.sum())}/{len(df)}")
    print(f"  median fraction of the initial gap closed: "
          f"{100*df.closed.median():.1f}%")
    print(f"\n{'gap closed':>12}{'instances reaching it':>24}{'median time':>14}")
    for f in CLOSED_AT:
        col = df[f"t_closed_{int(100*f)}"]
        n = int(col.notna().sum())
        med = f"{col.median():.2f}s" if n else "--"
        print(f"{int(100*f):>11}%{n:>16}/{len(df)}{med:>14}")
    print("\nwritten: certified_gap.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["validate", "trace"])
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--rho", type=float, default=0.12)
    ap.add_argument("--budget", type=float, default=60.0)
    a = ap.parse_args()
    if a.what == "validate":
        validate(**({"reps": a.reps} if a.reps else {}))
    else:
        trace(**({"reps": a.reps} if a.reps else {}), rho=a.rho, budget=a.budget)
