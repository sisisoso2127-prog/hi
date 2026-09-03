"""The one reformulation left: Balas' convex hull instead of big-M.

WHY THIS AND NOT ANOTHER TUNING PASS

diagnose_cost.py established that a residual-region solve gets expensive purely
through tree growth, and that the disjunctions contribute NOTHING to the LP
bound -- the root bound sat at 37.0440 whether M was the loose constant or the
tightened per-criterion one, and whether one cut had been added or six. That is
structural: the LP sets one z_i to 1 and the rest to 0, so every unselected
branch reads c^i x >= rhs_i - M_i, and any VALID M_i already makes that hold
everywhere in the region. No constant can move that bound.

Balas' homogenisation is the one encoding that breaks the property. For the
disjunction  OR_i ( c^i x >= rhs_i )  over D = {Ax <= b, 0 <= x <= ub}:

    x = sum_i y^i                       (one copy of x per branch)
    sum_i lam_i = 1
    A y^i <= b lam_i ,  0 <= y^i <= ub lam_i ,  c^i y^i >= rhs_i lam_i

With lam binary this is EXACT, and with the same r binaries per cut that big-M
uses. With lam relaxed it projects to conv(union of the branches) -- the
tightest relaxation the disjunction admits, where big-M's is the weakest.

So the trade is not binaries. It is model size: each cut costs r*n extra
continuous variables and r*(m+n+1)+n+1 extra rows. At n=40, m=25, r=5 that is
205 variables and 371 rows PER CUT, and runs reach 20+ cuts. Whether a bound
that actually bites pays for a model that large is exactly what is unmeasured.

READ THE LICENCE WARNING BELOW BEFORE RUNNING.

    python3 hull_experiment.py validate          # correctness first, ~2 min
    python3 hull_experiment.py bound             # the decisive measurement
    python3 hull_experiment.py endtoend          # whole algorithm, both ways

SIZE-LIMITED LICENCE.  The pip gurobipy that Colab installs is capped at 2000
variables and 2000 rows. The hull model passes 2000 rows at about 5 cuts on the
n=40 grid, so the defaults here use n=20, m=10, r=3, which stays inside the cap
to roughly 15 cuts. That small grid needs its own tightness to be hard at all --
rho=0.12, measured, against 0.02 for the n=40 grid -- and the default follows
the grid, so pass --rho only to sweep it deliberately. Pass --big for the
n=40 m=25 r=5 grid ONLY with a full or academic licence. Every subcommand reports the licence limit cleanly instead of
dying, and says which cut count it reached.

HOW TO READ THE RESULT.  'bound' is the cheap decisive one. Three outcomes:

  hull bound much tighter AND hull MILP faster  -> the lever is real; wire
      HullMethod into warmstart_experiment.py as a third configuration.
  hull bound much tighter but hull MILP slower  -> the bound was never the
      binding constraint; that closes the last reformulation and the answer is
      that this cut family cannot be made cheap, only used less often.
  hull bound NOT tighter                        -> something is wrong with the
      formulation, not with the idea.  Send me the output.

Read the branching count the run prints at the end FIRST. On the small grid many
instances solve at the root, and a comparison with one node cannot separate the
two encodings however the bound moves -- there is no tree to shrink. If that
count is low the run says nothing either way, and the numbers above it are not
evidence.
"""

import argparse
import itertools
import time

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from warmstart_experiment import (Method, generate_instance, brute_force,
                                  RNG_SEED)

SMALL = dict(n=20, m=10, r=3)      # stays inside the size-limited licence
BIG = dict(n=40, m=25, r=5)        # needs a full licence

# Tightness calibrated per grid: the value that reliably drives the algorithm to
# 14 cuts.  rho, not n, is what makes these instances hard (see
# warmstart_experiment.py); the small grid needs a much larger one.
RHO = {False: 0.12, True: 0.02}


class HullMethod(Method):
    """Algorithm 1 with the disjunction in Balas' convex-hull form.

    Same r binaries per cut as the big-M form and the same exact semantics; the
    difference is that relaxing those binaries yields conv(union of branches)
    rather than a constraint set that is vacuous off the selected branch.
    """

    def add_cut(self, xbar):
        C, n, r = self.I["C"], self.n, self.r
        A, b, ub = self.I["A"], self.I["b"], self.ub
        t = self.cuts
        lam = self.mR.addVars(r, vtype=GRB.BINARY, name=f"lam{t}")
        y = self.mR.addVars(r, n, lb=0.0, name=f"y{t}")

        self.mR.addConstr(gp.quicksum(lam[i] for i in range(r)) == 1)
        for j in range(n):                       # x is the sum of its copies
            self.mR.addConstr(gp.quicksum(y[i, j] for i in range(r)) == self.xR[j])
        for i in range(r):
            for qq in range(A.shape[0]):         # A y^i <= b lam_i
                self.mR.addConstr(
                    gp.quicksum(int(A[qq, j]) * y[i, j] for j in range(n))
                    <= int(b[qq]) * lam[i])
            for j in range(n):                   # 0 <= y^i <= ub lam_i
                self.mR.addConstr(y[i, j] <= int(ub) * lam[i])
            self.mR.addConstr(                   # the branch itself
                gp.quicksum(int(C[i, j]) * y[i, j] for j in range(n))
                >= (float(C[i] @ xbar) + 1.0) * lam[i])
        self.mR.update()
        self.cuts += 1


def _sizes(m):
    return m.NumVars, m.NumConstrs


def _collect_bars(inst, stop_after, budget=90.0):
    """Run the real algorithm once (big-M) and keep the sequence of x-bar it cuts on."""
    s = Method(inst, time_limit=30, wall_budget=budget)
    bars = []
    try:
        for _ in range(60):
            x0, _ = s.dinkelbach()
            if x0 is None or s.truncated:
                break
            eff, xp = s.efficiency_test(x0)
            if eff is None or eff:
                break
            s.find_best(xp)
            bars.append(np.array(xp))
            s.add_cut(xp)
            if len(bars) >= stop_after:
                break
    finally:
        s.close()
    return bars


def _build(inst, bars, hull, budget=90.0):
    cls = HullMethod if hull else Method
    s = cls(inst, time_limit=30, wall_budget=budget)
    for xb in bars:
        s.add_cut(xb)
    return s


def _measure(s, p, tl=120.0):
    """Root LP bound, integer optimum, MILP time, nodes, model size."""
    n = s.n
    s.mR.setObjective(gp.quicksum(float(p[j]) * s.xR[j] for j in range(n)),
                      GRB.MAXIMIZE)
    s.mR.update()
    nv, nc = _sizes(s.mR)
    rel = s.mR.relax()
    rel.Params.OutputFlag = 0
    rel.Params.Threads = 1
    rel.optimize()
    root = rel.ObjVal if rel.Status == GRB.OPTIMAL else np.nan
    rel.dispose()
    s.mR.reset()
    s.mR.Params.TimeLimit = tl
    t0 = time.perf_counter()
    s.mR.optimize()
    dt = time.perf_counter() - t0
    ok = s.mR.Status == GRB.OPTIMAL
    return dict(vars=nv, cons=nc, root=root,
                opt=(s.mR.ObjVal if ok else np.nan),
                t=dt, nodes=(s.mR.NodeCount if ok else np.nan), ok=ok)


def _guard(fn, *a, **kw):
    """Run fn, turning the size-limited licence error into a readable stop."""
    try:
        return fn(*a, **kw), None
    except gp.GurobiError as e:
        if "size-limited" in str(e).lower() or "too large" in str(e).lower():
            return None, str(e)
        raise


# ------------------------------------------------------------------ validate
def validate(reps=30):
    """The hull cut must give the same optimum as the big-M cut and as brute force."""
    print("VALIDATE -- hull cut against big-M and against exhaustive enumeration")
    rng = np.random.default_rng(RNG_SEED)
    ok = tot = 0
    for _ in range(reps):
        inst = generate_instance(rng, n=6, m=2, r=3, ub=3)
        truth = brute_force(inst)
        if truth is None:
            continue
        tot += 1
        good = True
        for cls, tag in ((Method, "big-M"), (HullMethod, "hull")):
            s = cls(inst, time_limit=30, wall_budget=60)
            try:
                res = s.solve()
            finally:
                s.close()
            if res["truncated"] or res["rounds_exhausted"]:
                print(f"  TRUNCATED ({tag}) -- proves nothing")
                good = False
                break
            if not np.isfinite(res["psi"]) or abs(res["psi"] - truth) > 1e-6:
                print(f"  MISMATCH ({tag}) got={res['psi']} expected={truth}")
                good = False
                break
        ok += good
    print(f"  {ok}/{tot} instances optimal under both cut forms")
    print("VALIDATION PASSED" if ok == tot and tot > 0 else "VALIDATION FAILED")


# --------------------------------------------------------------------- bound
def bound(big=False, rho=None, reps=8, cuts=(2, 4, 6, 8, 10, 12)):
    """Same cuts, two encodings: does the hull bound bite, and what does it cost?"""
    cfg, rho = (BIG if big else SMALL), (RHO[big] if rho is None else rho)
    print(f"BOUND -- n={cfg['n']} m={cfg['m']} r={cfg['r']} rho={rho}\n")
    rng = np.random.default_rng(RNG_SEED)
    rows, limit_hit = [], None
    for rep in range(reps):
        inst = generate_instance(rng, ub=10, rho=rho, **cfg)
        bars = _collect_bars(inst, max(cuts))
        print(f"  rep {rep}: algorithm produced {len(bars)} cuts", flush=True)
        for k in cuts:
            if k > len(bars):
                break
            row = dict(rep=rep, k=k)
            for hull, tag in ((False, "bigM"), (True, "hull")):
                out, err = _guard(_build, inst, bars[:k], hull)
                if err:
                    limit_hit = limit_hit or (k, err)
                    row = None
                    break
                s = out
                try:
                    mres, err = _guard(_measure, s, inst["p"])
                    if err:
                        limit_hit = limit_hit or (k, err)
                        row = None
                        break
                    for key, v in mres.items():
                        row[f"{tag}_{key}"] = v
                finally:
                    s.close()
            if row is None:
                break
            if not (row["bigM_ok"] and row["hull_ok"]):
                continue
            if abs(row["bigM_opt"] - row["hull_opt"]) > 1e-6:
                print(f"    !! k={k}: encodings disagree, {row['bigM_opt']} vs "
                      f"{row['hull_opt']} -- formulation bug, stop here")
            rows.append(row)
            print(f"    k={k:2d}  bigM {1000*row['bigM_t']:8.1f}ms "
                  f"bound={row['bigM_root']:9.3f} | hull {1000*row['hull_t']:8.1f}ms "
                  f"bound={row['hull_root']:9.3f} | opt={row['bigM_opt']:.3f}",
                  flush=True)

    if limit_hit:
        k, err = limit_hit
        print(f"\n  licence limit reached at k={k}: {err}")
        print("  -> use the default small grid, or a full/academic licence for --big")
    if not rows:
        print("\nno usable comparisons")
        return
    df = pd.DataFrame(rows)
    df.to_csv("hull_bound.csv", index=False)
    df["bigM_gap"] = df.bigM_root - df.bigM_opt
    df["hull_gap"] = df.hull_root - df.hull_opt
    g = df.groupby("k").agg(
        pts=("k", "size"),
        bigM_gap=("bigM_gap", "median"), hull_gap=("hull_gap", "median"),
        bigM_ms=("bigM_t", lambda s: 1000 * s.median()),
        hull_ms=("hull_t", lambda s: 1000 * s.median()),
        bigM_nodes=("bigM_nodes", "median"), hull_nodes=("hull_nodes", "median"),
        hull_vars=("hull_vars", "median"), hull_cons=("hull_cons", "median"))
    g["gap_kept"] = g.hull_gap / g.bigM_gap.replace(0, np.nan)
    g["time_ratio"] = g.hull_ms / g.bigM_ms
    print("\n" + "=" * 78)
    print(g.to_string(float_format=lambda v: f"{v:10.2f}"))
    print("\ngap_kept  < 1 means the hull bound is tighter (0 = bound is exact).")
    print("time_ratio< 1 means the hull MILP is faster.  Both must hold for a win.")

    # A comparison solved at the root has no tree to shrink, so it cannot tell
    # the two encodings apart whatever the bound does.  Say so rather than let
    # a table of ones read as a result.
    br = df[df.bigM_nodes > 1]
    print(f"\ncomparisons where big-M actually branched: {len(br)}/{len(df)}")
    if len(br) < max(3, len(df) // 4):
        print("  !! Too few to conclude anything.  These instances solve at the")
        print("  !! root, where neither encoding can help.  Raise --reps, or run")
        print("  !! --big on a full licence, which is where the tree explosion is.")
    else:
        print(f"  on those: bigM {1000*br.bigM_t.median():.1f}ms / "
              f"{br.bigM_nodes.median():.0f} nodes   "
              f"hull {1000*br.hull_t.median():.1f}ms / "
              f"{br.hull_nodes.median():.0f} nodes   "
              f"ratio {br.hull_t.median()/max(br.bigM_t.median(),1e-9):.2f}x")
        print(f"  median gap  bigM {br.bigM_gap.median():.3f}  "
              f"hull {br.hull_gap.median():.3f}")
    print("written: hull_bound.csv")


# ------------------------------------------------------------------ endtoend
def endtoend(big=False, rho=None, reps=6):
    """Whole algorithm, both cut forms, same instances."""
    cfg, rho = (BIG if big else SMALL), (RHO[big] if rho is None else rho)
    print(f"END TO END -- n={cfg['n']} m={cfg['m']} r={cfg['r']} rho={rho}\n")
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for rep in range(reps):
        inst = generate_instance(rng, ub=10, rho=rho, **cfg)
        rec = dict(rep=rep)
        for cls, tag in ((Method, "bigM"), (HullMethod, "hull")):
            s = cls(inst, time_limit=60, wall_budget=120)
            try:
                res, err = _guard(s.solve)
            finally:
                s.close()
            if err:
                print(f"  rep {rep}: licence limit -- {err}")
                rec = None
                break
            rec[f"{tag}_t"] = res["t_milp"]
            rec[f"{tag}_milps"] = res["milps"]
            rec[f"{tag}_cuts"] = res["cuts"]
            rec[f"{tag}_psi"] = res["psi"]
            rec[f"{tag}_trunc"] = res["truncated"] or res["rounds_exhausted"]
        if rec is None:
            continue
        rec["agree"] = abs(rec["bigM_psi"] - rec["hull_psi"]) < 1e-6
        rows.append(rec)
        print(f"  rep {rep}: bigM {rec['bigM_t']:7.2f}s/{rec['bigM_milps']:3d}milp "
              f"| hull {rec['hull_t']:7.2f}s/{rec['hull_milps']:3d}milp "
              f"| cuts {rec['bigM_cuts']}/{rec['hull_cuts']} "
              f"| agree={rec['agree']}"
              + ("  <-- TRUNCATED" if rec["bigM_trunc"] or rec["hull_trunc"] else ""),
              flush=True)
    if not rows:
        print("\nno usable comparisons")
        return
    df = pd.DataFrame(rows)
    df.to_csv("hull_endtoend.csv", index=False)
    clean = df[~(df.bigM_trunc | df.hull_trunc)]
    print("\n" + "=" * 78)
    print(f"  agree on the optimum: {int(df.agree.sum())}/{len(df)}")
    print(f"  runs where neither truncated: {len(clean)}/{len(df)}")
    if len(clean):
        print(f"  median time  bigM {clean.bigM_t.median():.2f}s   "
              f"hull {clean.hull_t.median():.2f}s   "
              f"ratio {clean.hull_t.median()/max(clean.bigM_t.median(),1e-9):.2f}x")
    print("written: hull_endtoend.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["validate", "bound", "endtoend"])
    ap.add_argument("--big", action="store_true",
                    help="n=40 m=25 r=5; needs a full licence")
    ap.add_argument("--rho", type=float, default=None,
                    help="constraint tightness; default 0.12 small, 0.02 --big")
    ap.add_argument("--reps", type=int, default=None)
    a = ap.parse_args()
    if a.what == "validate":
        validate(**({"reps": a.reps} if a.reps else {}))
    elif a.what == "bound":
        bound(a.big, a.rho, **({"reps": a.reps} if a.reps else {}))
    else:
        endtoend(a.big, a.rho, **({"reps": a.reps} if a.reps else {}))
