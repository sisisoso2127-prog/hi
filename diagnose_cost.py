"""What actually makes a residual-region solve expensive as cuts accumulate.

warmstart_experiment.py establishes that cost per MILP is the factor that grows
(2.7ms -> 101ms per MILP across the swept rho) and that neither warm starts nor
a tightened big-M touches it. This script asks why, and prices the obvious
alternative. Three measurements, each a subcommand:

  profile  -- per-solve cost of the residual model against the number of
              accumulated cuts, split into node count and cost per node, with
              the pure LP relaxation gap alongside.

  bigm     -- the root LP bound under the loose constant and under the tightened
              per-criterion M, on the same models, plus what the disjunction
              binaries do in that relaxation.

  branch   -- the big-M formulation against explicit branching: each node fixes
              one branch of one disjunction as a plain constraint and carries no
              binaries at all, infeasible nodes prune, bound-dominated nodes
              prune. Same answer required, total time compared.

WHAT THESE FOUND (n=40, m=25, r=5, rho in {0.010,0.015,0.020}, one core):

  profile: an mR solve goes 2.1ms at 0 cuts to 706ms at 21+, while node count
  goes 1 to 1476 and cost PER NODE FALLS, 2.09ms to 0.48ms. The growth is tree
  size, not per-node work. The LP gap grows 0.22 to 2.52 over the same range.

  bigm: the root bound is 37.0440 in all six cases -- identical under the loose
  constant and the tightened one, AND unmoved by going from 1 cut to 6. The
  accumulated cuts contribute nothing whatever to the relaxation; the gap grows
  only because the integer optimum falls (33 -> 31) while the bound stands
  still. The reason is visible in the same run: the LP sets one z_i to 1 and the
  rest to 0, so each unselected branch reads c^i x >= rhs_i - M_i, and any VALID
  M_i is by definition at least rhs_i - min_D c^i x, which every point of D
  satisfies. The unselected branches are exactly vacuous for any valid M. So
  tightening M cannot move the LP bound -- the null result is structural, not a
  tuning failure, and no choice of constant will change it.

  branch: explicit branching matched big-M on all 39 comparisons and was 17x to
  64x SLOWER (k=2: 83ms vs 1.37s; k=10: 351ms vs 21.3s). Node counts run 18 to
  226 and each node is a fresh MILP over the base region, which alone costs
  40-50ms. Gurobi's own branching over the disjunction amortises far better
  than any outer decomposition. Removing the binaries is not the lever.

Figures are medians of one run on one core; expect a few percent of drift.

So the cost is tree growth forced by a disjunction that is vacuous in the LP,
and the two reformulations that keep big-M or discard it both fail. What is
left untested is a formulation whose relaxation is NOT vacuous -- a convex-hull
(lift-and-project) disjunction, which costs r copies of x per cut. That is the
next thing to measure, and it is not obviously affordable.

USAGE
    python3 diagnose_cost.py profile
    python3 diagnose_cost.py bigm
    python3 diagnose_cost.py branch
"""

import argparse
import time

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from warmstart_experiment import Method, generate_instance, RNG_SEED

N, M_CON, R, UB = 40, 25, 5, 10
RHOS = (0.015, 0.020)


def _rounds(inst, tight_m=False, stop_at=(), budget=60.0):
    """Drive the real algorithm, yielding (Method, xbar list) at chosen cut counts."""
    solver = Method(inst, tight_m=tight_m, time_limit=30, wall_budget=budget)
    bars = []
    try:
        for _ in range(40):
            x0, _ = solver.dinkelbach()
            if x0 is None or solver.truncated:
                return
            eff, xp = solver.efficiency_test(x0)
            if eff is None or eff:
                return
            solver.find_best(xp)
            bars.append(np.array(xp))
            solver.add_cut(xp)
            if solver.cuts in stop_at:
                yield solver, bars
    finally:
        solver.close()


def _maximise_p(m, x, p, n):
    m.setObjective(gp.quicksum(float(p[j]) * x[j] for j in range(n)), GRB.MAXIMIZE)


# ------------------------------------------------------------------ profile
def profile():
    rows = []
    rng = np.random.default_rng(RNG_SEED)
    for rho in (0.010,) + RHOS:
        for _ in range(6):
            inst = generate_instance(rng, n=N, m=M_CON, r=R, ub=UB, rho=rho)
            solver = Method(inst, time_limit=60, wall_budget=45)
            orig = solver._run

            def logged(m, start=None, _s=solver, _o=orig):
                is_mR = m is _s.mR
                cuts = _s.cuts
                t0 = time.perf_counter()
                out = _o(m, start)
                dt = time.perf_counter() - t0
                if is_mR and m.Status == GRB.OPTIMAL:
                    rel = m.relax()
                    rel.Params.OutputFlag = 0
                    rel.Params.Threads = 1
                    rel.optimize()
                    root = rel.ObjVal if rel.Status == GRB.OPTIMAL else np.nan
                    rel.dispose()
                    rows.append(dict(cuts=cuts, t=dt, nodes=m.NodeCount,
                                     nbin=sum(1 for v in m.getVars()
                                              if v.VType == GRB.BINARY),
                                     gap=(root - m.ObjVal) / max(1.0, abs(m.ObjVal))))
                return out

            solver._run = logged
            try:
                solver.solve()
            finally:
                solver.close()
            print(f"  rho={rho}: {len(rows)} mR solves logged", flush=True)

    df = pd.DataFrame(rows)
    df["bucket"] = pd.cut(df.cuts, [-1, 0, 2, 5, 10, 20, 10**6],
                          labels=["0", "1-2", "3-5", "6-10", "11-20", "21+"])
    out = []
    for name, g in df.groupby("bucket", observed=True):
        out.append(dict(cuts=name, solves=len(g), binaries=g.nbin.median(),
                        ms=1000 * g.t.median(), nodes=g.nodes.median(),
                        ms_per_node=1000 * g.t.median() / max(g.nodes.median(), 1),
                        LP_gap=g.gap.median()))
    print("\ncost of one residual-region solve, by accumulated cuts")
    print(pd.DataFrame(out).to_string(index=False,
                                      float_format=lambda v: f"{v:9.3f}"))
    print("\nIf ms rises while ms_per_node does not, the growth is tree size.")


# --------------------------------------------------------------------- bigm
def bigm():
    print("root LP bound under each constant, and what the binaries do there\n")
    print(f"{'M':<8}{'cuts':>5}{'median z':>10}{'z==1':>6}{'root bound':>13}"
          f"{'integer opt':>13}{'gap':>9}")
    for tight in (False, True):
        inst = generate_instance(np.random.default_rng(RNG_SEED),
                                 n=N, m=M_CON, r=R, ub=UB, rho=0.015)
        p = inst["p"]
        for solver, _bars in _rounds(inst, tight_m=tight, stop_at=(1, 3, 6)):
            _maximise_p(solver.mR, solver.xR, p, solver.n)
            solver.mR.optimize()
            if solver.mR.Status != GRB.OPTIMAL:
                break
            opt = solver.mR.ObjVal
            rel = solver.mR.relax()
            rel.Params.OutputFlag = 0
            rel.optimize()
            zb = [v.VarName for v in solver.mR.getVars() if v.VType == GRB.BINARY]
            zv = np.array([rel.getVarByName(nm).X for nm in zb])
            print(f"{'tight' if tight else 'loose':<8}{solver.cuts:>5}"
                  f"{np.median(zv):>10.4f}{int((zv > 0.99).sum()):>6}"
                  f"{rel.ObjVal:>13.4f}{opt:>13.4f}{rel.ObjVal - opt:>9.4f}")
            rel.dispose()
    print("\nOne z_i per cut goes to 1 and the rest to 0, so every unselected")
    print("branch reads c^i x >= rhs_i - M_i.  A valid M_i already makes that")
    print("hold everywhere in D, so the bound cannot depend on M.")


# ------------------------------------------------------------------- branch
def _dfs(solver, bars, p, cap_nodes=4000, cap_t=60.0):
    C, n, r = solver.I["C"], solver.n, solver.r
    t0 = time.perf_counter()
    best, nodes, leaves, pruned = -np.inf, 0, 0, 0
    stack = [[]]
    while stack:
        if nodes >= cap_nodes or time.perf_counter() - t0 > cap_t:
            return None, nodes, leaves, pruned, time.perf_counter() - t0
        path = stack.pop()
        m, x = solver._new_model()
        for d, i in enumerate(path):
            m.addConstr(gp.quicksum(int(C[i, j]) * x[j] for j in range(n))
                        >= float(C[i] @ bars[d]) + 1.0)
        _maximise_p(m, x, p, n)
        m.Params.TimeLimit = 10
        m.optimize()
        nodes += 1
        status, val = m.Status, (m.ObjVal if m.SolCount else None)
        m.dispose()
        if status == GRB.INFEASIBLE:
            pruned += 1
            continue
        if status != GRB.OPTIMAL:
            return None, nodes, leaves, pruned, time.perf_counter() - t0
        if val <= best + 1e-9:                       # cannot beat the incumbent
            continue
        if len(path) == len(bars):
            leaves += 1
            best = max(best, val)
            continue
        stack.extend(path + [i] for i in range(r))
    return best, nodes, leaves, pruned, time.perf_counter() - t0


def branch():
    rows = []
    rng = np.random.default_rng(RNG_SEED)
    for rho in RHOS:
        for _ in range(5):
            inst = generate_instance(rng, n=N, m=M_CON, r=R, ub=UB, rho=rho)
            p = inst["p"]
            for solver, bars in _rounds(inst, stop_at=(2, 4, 6, 8, 10)):
                _maximise_p(solver.mR, solver.xR, p, solver.n)
                solver.mR.reset()
                solver.mR.Params.TimeLimit = 60
                t0 = time.perf_counter()
                solver.mR.optimize()
                t_big = time.perf_counter() - t0
                if solver.mR.Status != GRB.OPTIMAL:
                    continue
                ref = solver.mR.ObjVal
                best, nodes, leaves, pruned, t_dfs = _dfs(solver, bars, p)
                rows.append(dict(k=solver.cuts, t_big=t_big, t_dfs=t_dfs,
                                 nodes=nodes, leaves=leaves, pruned=pruned,
                                 finished=best is not None,
                                 ok=best is not None and abs(best - ref) < 1e-6))
                print(f"    k={solver.cuts:2d}  bigM={1000*t_big:7.1f}ms"
                      f"  branch={1000*t_dfs:8.1f}ms  nodes={nodes:5d}"
                      f"  match={rows[-1]['ok']}", flush=True)

    df = pd.DataFrame(rows)
    fin = df[df.finished]
    print(f"\n{len(df)} comparisons, {len(fin)} finished within the caps")
    if len(fin):
        print(f"  explicit branching matched big-M in {int(fin.ok.sum())}/{len(fin)}")
        g = fin.groupby("k").agg(pts=("t_big", "size"),
                                 bigM_ms=("t_big", lambda s: 1000 * s.median()),
                                 branch_ms=("t_dfs", lambda s: 1000 * s.median()),
                                 nodes=("nodes", "median"))
        g["branch/bigM"] = g.branch_ms / g.bigM_ms
        print(g.to_string(float_format=lambda v: f"{v:10.2f}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["profile", "bigm", "branch"])
    a = ap.parse_args()
    {"profile": profile, "bigm": bigm, "branch": branch}[a.what]()
