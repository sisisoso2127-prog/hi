"""
Colab experiment -- the ONE lever the measurements point at.

Your decomposition was

    time growth  x8.62  =  (number of MILPs) x1.23   x   (cost per MILP) x7.06

Every heuristic tried so far attacked the first factor, which does not grow.
This attacks the second, which carries ~100% of the growth. Two levers, tested
as a 2x2 factorial on the same instances:

  WARM  -- pass a known feasible point as a MIP start to every subproblem.
           Two of the three subproblem types have a guaranteed feasible start:
           the efficiency test E(x0) is feasible at (x0, w=0), and Find-Best is
           feasible at x'. The Dinkelbach inner loop reuses the previous
           parametric solution, which stays feasible because the region is
           fixed FOR THE DURATION OF THAT LOOP. Across outer rounds there is
           deliberately no start: add_cut() removes exactly the point that
           produced it, so a carried-over incumbent is infeasible by
           construction and only costs Gurobi a sub-MIP to discover that.
           Exactness is untouched: Gurobi proves optimality regardless of where
           it starts.

  TIGHTM -- replace the single loose constant in the disjunctive cut by a
           per-criterion bound M_i = c^i xbar + 1 - min_{x in D} c^i x,
           computed once per instance with r extra MILPs (counted in the cost).
           A reformulation, not a heuristic; it strengthens the LP relaxation
           that the accumulated cuts weaken.

INSTANCE HARDNESS.  The literal RHS of Drici et al. (b_i drawn from [50,100],
independent of n) does not scale: with A_ij in [1,30] it pins sum_j x_j to
about 3.5 at EVERY n in a 25..40 sweep, so ub=10 is never active and cost per
MILP does not grow -- the effect this experiment exists to measure is absent by
construction. The driver of hardness is constraint tightness, not n. So the
RHS is set to b_i = floor(rho * sum_j A_ij * ub) and rho is the swept axis;
rho=0.010 solves in milliseconds and rho=0.020 in seconds, a ~20x spread in
cost per MILP. Pass --drici to reproduce the literal (flat) protocol.

USAGE ON COLAB
    !pip install gurobipy -q
    !python3 warmstart_experiment.py --smoke      # validate first, ~2 min
    !python3 warmstart_experiment.py              # full run, ~18 min (measured, 1 core)
    from google.colab import files; files.download('warmstart_raw.csv')

RUN --smoke FIRST. It checks all four configurations against exhaustive
enumeration on tiny instances, under both RHS protocols. If it does not print
'VALIDATION PASSED', do not trust any timing that follows -- send me the output
instead.

A NOTE ON WHAT THE TIMINGS ARE WORTH. Every subproblem carries a TimeLimit, each
run carries a wall budget (hardness has a heavy tail, so one instance would
otherwise swallow the sweep), and the outer loop carries max_rounds. A run that
hits any of them has NOT proved
optimality, and a configuration that gives up early would otherwise look FAST in
the timing table -- the failure inverts the result it is supposed to inform. So
truncation is tracked per run ('truncated', 'rounds_exhausted') and the summary
refuses to stand behind a table that contains any.
"""

import argparse
import itertools
import time

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    raise SystemExit("gurobipy missing.  Run:  !pip install gurobipy -q")

RNG_SEED = 20260902
LOOSE_M = 10_000
EPS = 1e-9


# --------------------------------------------------------------- instances
def generate_instance(rng, n, m, r, ub, rho=None):
    """Protocol of Drici et al. (2018); rho rescales the RHS with the instance.

    rho=None reproduces the paper's literal b_i ~ U[50,100].  Both paths draw
    from the stream in the same order, so a given seed yields the same A, C, p,
    q, alpha, beta under either protocol -- only b differs.
    """
    A = rng.integers(1, 31, size=(m, n))
    b = rng.integers(50, 101, size=m)
    C = rng.integers(-10, 11, size=(r, n))
    p = rng.integers(-10, 11, size=n)
    alpha = rng.integers(1, 21)
    q = rng.integers(1, 11, size=n)
    beta = rng.integers(1, 21)
    if rho is not None:
        b = np.maximum(1, np.floor(rho * A.sum(axis=1) * ub)).astype(np.int64)
    return dict(A=A, b=b, C=C, p=p, alpha=alpha, q=q, beta=beta,
                n=n, r=r, ub=ub, rho=rho)


# ---------------------------------------------------------------- solver
class Method:
    """Algorithm 1 of the paper, instrumented, with two optional levers."""

    def __init__(self, inst, warm=False, tight_m=False, time_limit=300,
                 wall_budget=None):
        self.I = inst
        self.warm, self.tight_m = warm, tight_m
        self.t_limit = time_limit
        # Hardness has a heavy tail: one instance can otherwise swallow the whole
        # sweep.  The budget bounds a run and marks it truncated -- never silently.
        self.wall_budget = wall_budget
        self.t_deadline = (time.perf_counter() + wall_budget) if wall_budget else None
        self.milps = 0          # subproblems solved
        self.t_milp = 0.0       # seconds inside optimize()
        self.cuts = 0
        self.proven = True      # last solve reached a proved status
        self.truncated = False  # some solve did not -> the run proves nothing
        self.rounds_exhausted = False
        self.loose_bounds = 0   # tightM bounds that fell back to a weaker M
        self.n, self.r, self.ub = inst["n"], inst["r"], inst["ub"]
        self.env = gp.Env(params={"OutputFlag": 0})
        self._check_loose_m()
        self._build()

    def close(self):
        """Gurobi environments are not garbage-collected; a sweep leaks them."""
        for attr in ("mR", "mE", "mS"):
            mod = getattr(self, attr, None)
            if mod is not None:
                mod.dispose()
        self.env.dispose()

    # ---- helpers ---------------------------------------------------
    def _check_loose_m(self):
        """LOOSE_M must dominate  (c^i xbar + 1) - min_D c^i x  or the base
        configuration silently cuts off feasible efficient points.  Bound both
        ends by relaxing Ax<=b away, which costs nothing."""
        C, ub = self.I["C"], self.ub
        hi = np.maximum(C, 0).sum(axis=1) * ub          # >= max_D c^i x
        lo = np.minimum(C, 0).sum(axis=1) * ub          # <= min_D c^i x
        need = float((hi + 1.0 - lo).max())
        if need > LOOSE_M:
            raise SystemExit(f"LOOSE_M={LOOSE_M} is too small for this size "
                             f"(needs {need:.0f}); the base cut would be invalid.")

    def _new_model(self):
        m = gp.Model(env=self.env)
        m.Params.OutputFlag = 0
        m.Params.Threads = 1
        x = m.addVars(self.n, vtype=GRB.INTEGER, lb=0, ub=self.ub, name="x")
        A, b = self.I["A"], self.I["b"]
        for i in range(A.shape[0]):
            m.addConstr(gp.quicksum(int(A[i, j]) * x[j] for j in range(self.n))
                        <= int(b[i]))
        m.update()
        return m, x

    def _budget_left(self):
        """Seconds this solve may use; 0 means the wall budget is spent."""
        if self.t_deadline is None:
            return self.t_limit
        return min(self.t_limit, self.t_deadline - time.perf_counter())

    def _run(self, m, start=None):
        """One counted, timed solve.  Sets self.proven for the caller."""
        if self._budget_left() <= 0.0:
            self.proven = False
            self.truncated = True
            return None
        if self.warm and start is not None:
            for j, v in enumerate(start):
                m.getVarByName(f"x[{j}]").Start = float(v)
        m.Params.TimeLimit = max(1.0, self._budget_left())
        t0 = time.perf_counter()
        m.optimize()
        self.t_milp += time.perf_counter() - t0
        self.milps += 1
        # INFEASIBLE is a proved answer -- the residual region really is empty.
        self.proven = m.Status in (GRB.OPTIMAL, GRB.INFEASIBLE, GRB.INF_OR_UNBD)
        if not self.proven:
            self.truncated = True
        if m.SolCount > 0:      # keep the incumbent instead of discarding it
            return np.array([m.getVarByName(f"x[{j}]").X for j in range(self.n)]
                            ).round().astype(np.int64)
        return None

    def _build(self):
        self.mR, self.xR = self._new_model()          # residual region
        self.mE, self.xE = self._new_model()          # efficiency test
        self.mS, self.xS = self._new_model()          # Find-Best slice
        C, r = self.I["C"], self.r

        # efficiency test:  max sum w  s.t.  c^i x - w_i = c^i x0
        self.wE = self.mE.addVars(r, lb=0.0, name="w")
        self.cE = [self.mE.addConstr(
            gp.quicksum(int(C[i, j]) * self.xE[j] for j in range(self.n))
            - self.wE[i] == 0.0) for i in range(r)]
        self.mE.setObjective(gp.quicksum(self.wE[i] for i in range(r)), GRB.MAXIMIZE)

        # slice:  Cx = Cx'
        self.cS = [self.mS.addConstr(
            gp.quicksum(int(C[i, j]) * self.xS[j] for j in range(self.n)) == 0.0)
            for i in range(r)]

        # per-criterion lower bounds for the tightened M (counted)
        self.Lo = None
        if self.tight_m:
            mb, xb = self._new_model()
            self.Lo = np.empty(r)
            for i in range(r):
                mb.Params.TimeLimit = max(1.0, self._budget_left())  # bypasses _run
                mb.setObjective(gp.quicksum(int(C[i, j]) * xb[j]
                                            for j in range(self.n)), GRB.MINIMIZE)
                mb.reset()
                t0 = time.perf_counter()
                mb.optimize()
                self.t_milp += time.perf_counter() - t0
                self.milps += 1
                # An OVER-estimated Lo makes M_i too small and the cut removes
                # feasible efficient points, so ObjVal is never read unguarded.
                # ObjBound stays a valid lower bound whatever the status.
                if mb.Status == GRB.OPTIMAL:
                    self.Lo[i] = mb.ObjVal
                else:
                    self.loose_bounds += 1
                    try:
                        cand = float(mb.ObjBound)
                    except (AttributeError, gp.GurobiError):
                        cand = -np.inf
                    self.Lo[i] = cand if np.isfinite(cand) else -float(LOOSE_M)
            mb.dispose()

    # ---- subproblems ----------------------------------------------
    def dinkelbach(self, start=None, max_it=40):
        """max psi over the current residual region."""
        p, al, q, be = (self.I["p"], self.I["alpha"], self.I["q"], self.I["beta"])
        lam, best = 0.0, None
        for _ in range(max_it):
            self.mR.setObjective(
                gp.quicksum((float(p[j]) - lam * float(q[j])) * self.xR[j]
                            for j in range(self.n)) + (al - lam * be), GRB.MAXIMIZE)
            self.mR.reset()
            z = self._run(self.mR, start)
            if z is None:
                return None, -np.inf
            val = (p @ z + al) / (q @ z + be)
            start = z
            if abs(val - lam) < 1e-10:
                return z, val
            lam, best = val, z
        self.truncated = True          # the ratio iteration did not converge
        if best is None:
            return None, -np.inf
        return best, (p @ best + al) / (q @ best + be)

    def efficiency_test(self, x0):
        """(True, .) / (False, x') / (None, .) when the test did not resolve."""
        C = self.I["C"]
        for i in range(self.r):
            self.cE[i].RHS = float(C[i] @ x0)
        self.mE.reset()
        xp = self._run(self.mE, x0)                    # (x0, w=0) is feasible
        # An unproved test must NEVER be read as 'x0 is efficient': that ends
        # the run early with a non-efficient point reported as the optimum.
        if not self.proven or xp is None:
            self.truncated = True
            return None, x0
        return abs(self.mE.ObjVal) < 1e-7, xp

    def find_best(self, xp):
        C = self.I["C"]
        for i in range(self.r):
            self.cS[i].RHS = float(C[i] @ xp)
        p, al, q, be = (self.I["p"], self.I["alpha"], self.I["q"], self.I["beta"])
        lam, cur = (p @ xp + al) / (q @ xp + be), xp
        for _ in range(40):
            self.mS.setObjective(
                gp.quicksum((float(p[j]) - lam * float(q[j])) * self.xS[j]
                            for j in range(self.n)) + (al - lam * be), GRB.MAXIMIZE)
            self.mS.reset()
            z = self._run(self.mS, cur)                # xp is feasible
            if z is None:
                return cur, lam
            val = (p @ z + al) / (q @ z + be)
            cur = z
            if abs(val - lam) < 1e-10:
                return z, val
            lam = val
        self.truncated = True
        return cur, lam

    def add_cut(self, xbar):
        """Disjunction removing the box {x : Cx <= C xbar}."""
        C = self.I["C"]
        zb = self.mR.addVars(self.r, vtype=GRB.BINARY)
        for i in range(self.r):
            rhs = float(C[i] @ xbar) + 1.0
            M = (rhs - self.Lo[i]) if self.tight_m else LOOSE_M
            self.mR.addConstr(
                gp.quicksum(int(C[i, j]) * self.xR[j] for j in range(self.n))
                >= rhs - M * (1 - zb[i]))
        self.mR.addConstr(gp.quicksum(zb[i] for i in range(self.r)) >= 1)
        self.mR.update()
        self.cuts += 1

    # ---- main loop -------------------------------------------------
    def solve(self, max_rounds=60):
        p, al, q, be = (self.I["p"], self.I["alpha"], self.I["q"], self.I["beta"])
        psi_star, x_star = -np.inf, None
        bound = None
        t_wall = time.perf_counter()

        for _ in range(max_rounds):
            # No MIP start is carried across rounds: add_cut() removed exactly
            # the x0 of the previous round, so any carried point is infeasible
            # by construction.  The reuse that pays off is inside this call.
            x0, psi0 = self.dinkelbach()
            if self.truncated or x0 is None or psi0 <= psi_star + EPS:
                break
            eff, xp = self.efficiency_test(x0)
            if eff is None:                     # unresolved -- assume nothing
                break
            if eff:
                psi_star, x_star = psi0, x0
                break
            xs, ps = self.find_best(xp)
            if self.truncated:
                break
            if ps > psi_star:
                psi_star, x_star = ps, xs
                if bound is not None:
                    self.mR.remove(bound)
                bound = self.mR.addConstr(                # Remark 3.4
                    gp.quicksum((float(p[j]) - psi_star * float(q[j])) * self.xR[j]
                                for j in range(self.n))
                    >= psi_star * be - al)
            self.add_cut(xp)
        else:
            self.rounds_exhausted = True        # optimality was not proved

        return dict(psi=psi_star, x=x_star, cuts=self.cuts, milps=self.milps,
                    t_milp=self.t_milp, t_wall=time.perf_counter() - t_wall,
                    truncated=self.truncated, rounds_exhausted=self.rounds_exhausted,
                    loose_bounds=self.loose_bounds)


def run_once(inst, warm, tight_m, time_limit=300, wall_budget=None):
    """Solve and release the Gurobi environment."""
    M = Method(inst, warm=warm, tight_m=tight_m, time_limit=time_limit,
               wall_budget=wall_budget)
    try:
        return M.solve()
    finally:
        M.close()


# ---------------------------------------------------------------- brute force
def brute_force(inst):
    n, ub = inst["n"], inst["ub"]
    G = np.array(list(itertools.product(range(ub + 1), repeat=n)), dtype=np.int64)
    X = G[np.all(G @ inst["A"].T <= inst["b"], axis=1)]
    if X.shape[0] == 0:
        return None
    Z = X @ inst["C"].T
    eff = []
    for i in range(X.shape[0]):
        if not np.any(np.all(Z >= Z[i], axis=1) & np.any(Z > Z[i], axis=1)):
            eff.append(i)
    if not eff:
        return None
    psi = (X[eff] @ inst["p"] + inst["alpha"]) / (X[eff] @ inst["q"] + inst["beta"])
    return float(psi.max())


CONFIGS = [("base", False, False), ("warm", True, False),
           ("tightM", False, True), ("both", True, True)]


# --------------------------------------------------------------------- runs
def smoke():
    print("SMOKE TEST -- all four configurations against exhaustive enumeration")
    ok = tot = trunc = 0
    for label, rho in (("drici RHS ", None), ("scaled RHS", 0.20)):
        rng = np.random.default_rng(RNG_SEED)
        sub_ok = sub_tot = 0
        for _ in range(40):
            inst = generate_instance(rng, n=6, m=2, r=3, ub=3, rho=rho)
            truth = brute_force(inst)
            if truth is None:
                continue
            sub_tot += 1
            for name, w, t in CONFIGS:
                res = run_once(inst, w, t)
                if res["truncated"] or res["rounds_exhausted"]:
                    trunc += 1
                    print(f"  TRUNCATED cfg={name} ({label}) -- proves nothing")
                    break
                if not np.isfinite(res["psi"]) or abs(res["psi"] - truth) > 1e-6:
                    print(f"  MISMATCH  cfg={name} ({label})  "
                          f"got={res['psi']}  expected={truth}")
                    break
            else:
                sub_ok += 1
        print(f"  {label}: {sub_ok}/{sub_tot} instances optimal in all four configurations")
        ok += sub_ok
        tot += sub_tot
    print(f"  total {ok}/{tot}, truncated runs {trunc}")
    print("VALIDATION PASSED" if ok == tot and tot > 0 and trunc == 0
          else "VALIDATION FAILED")


# rho is the axis that actually moves cost per MILP; n alone does not.
GRID = [(30, 20, 3, 0.010), (30, 20, 3, 0.015), (30, 20, 3, 0.020),
        (40, 25, 5, 0.010), (40, 25, 5, 0.015), (40, 25, 5, 0.020)]
DRICI_GRID = [(25, 20, 3, None), (30, 20, 3, None), (30, 25, 3, None),
              (35, 20, 3, None), (35, 25, 5, None), (40, 25, 3, None),
              (40, 25, 5, None)]
REPLICATES = 8
WALL_BUDGET = 30.0      # seconds per configuration per instance; see Method


def full(grid=None, reps=REPLICATES, time_limit=300, wall_budget=WALL_BUDGET):
    grid = GRID if grid is None else grid
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for (n, m, r, rho) in grid:
        for k in range(reps):
            inst = generate_instance(rng, n=n, m=m, r=r, ub=10, rho=rho)
            rec = dict(n=n, m=m, r=r, rho=rho, rep=k)
            ref = None
            for name, w, t in CONFIGS:
                res = run_once(inst, w, t, time_limit, wall_budget)
                rec[f"{name}_t"] = res["t_milp"]
                rec[f"{name}_milps"] = res["milps"]
                rec[f"{name}_cuts"] = res["cuts"]
                rec[f"{name}_psi"] = res["psi"]
                rec[f"{name}_trunc"] = bool(res["truncated"] or res["rounds_exhausted"])
                if ref is None:
                    ref = res["psi"]
                rec[f"{name}_agree"] = bool(np.isfinite(res["psi"])
                                            and np.isfinite(ref)
                                            and abs(res["psi"] - ref) < 1e-6)
            rows.append(rec)
            flag = "  <-- TRUNCATED" if any(rec[c + "_trunc"] for c, _, _ in CONFIGS) else ""
            print(f"  n={n} m={m} r={r} rho={rho} rep={k}  "
                  + "  ".join(f"{c}={rec[c+'_t']:.1f}s" for c, _, _ in CONFIGS)
                  + flag, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("warmstart_raw.csv", index=False)

    n_trunc = int(sum(df[c + "_trunc"].sum() for c, _, _ in CONFIGS))
    print("\n" + "=" * 76)
    print("AGREEMENT (every configuration must return base's optimum)")
    for name, _, _ in CONFIGS[1:]:            # base vs base is vacuous
        print(f"  {name:<7} agrees with base in {int(df[name+'_agree'].sum())}/{len(df)}")
    clean = df[~df[[c + "_trunc" for c, _, _ in CONFIGS]].any(axis=1)]
    clean_ok = int(clean[[c + "_agree" for c, _, _ in CONFIGS[1:]]].all(axis=1).sum())
    print(f"  among the {len(clean)} instances where nothing truncated: "
          f"all four agree in {clean_ok}/{len(clean)}")
    print(f"  truncated runs: {n_trunc}/{4*len(df)}")
    if n_trunc:
        print("  !! Truncated runs did NOT prove optimality.  A configuration that")
        print("  !! gives up early looks FAST below.  Raise --time-limit or lower")
        print("  !! rho before reading the timings as evidence.")

    print("\n" + "=" * 76)
    print("THE DECOMPOSITION THAT MATTERS -- medians")
    print(f"{'cfg':<8}{'time (s)':>10}{'#MILP':>9}{'ms/MILP':>10}{'cuts':>7}"
          f"{'vs base':>10}")
    base_t = df["base_t"].median()
    for name, _, _ in CONFIGS:
        t, k = df[name + "_t"], df[name + "_milps"]
        print(f"{name:<8}{t.median():>10.2f}{k.median():>9.0f}"
              f"{(1000*t/k).median():>10.1f}{df[name+'_cuts'].median():>7.0f}"
              f"{100*(t.median()-base_t)/max(base_t,1e-9):>9.1f}%")

    print("\n" + "=" * 76)
    print("COST PER MILP ACROSS THE SWEPT AXIS (base configuration)")
    key = [c for c in ("rho", "n") if df[c].notna().any() and df[c].nunique() > 1]
    if key:
        g = df.groupby(key).apply(
            lambda d: pd.Series({"t": d["base_t"].median(),
                                 "milps": d["base_milps"].median(),
                                 "ms/MILP": (1000 * d["base_t"] / d["base_milps"]).median()}),
            include_groups=False)
        print(g.to_string(float_format=lambda v: f"{v:9.2f}"))

    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    print("\n" + "=" * 76)
    print("PAIRED TESTS on time inside the solver")
    for name, _, _ in CONFIGS[1:]:
        a, b = df["base_t"], df[name + "_t"]
        w = wilcoxon(a, b) if (wilcoxon is not None and (a != b).any()) else None
        print(f"  base vs {name:<7} median change "
              f"{100*(b-a).median()/max(a.median(),1e-9):+7.1f}%  "
              f"p={'n/a' if w is None else f'{w.pvalue:.2g}'}  "
              f"better in {int((b<a).sum())}/{len(df)}")
    print("\nwritten: warmstart_raw.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--drici", action="store_true",
                    help="literal Drici RHS grid (flat hardness; see module docstring)")
    ap.add_argument("--reps", type=int, default=REPLICATES)
    ap.add_argument("--time-limit", type=float, default=300.0,
                    help="cap per subproblem (s)")
    ap.add_argument("--wall-budget", type=float, default=WALL_BUDGET,
                    help="cap per configuration per instance (s); 0 disables")
    a = ap.parse_args()
    if a.smoke:
        smoke()
    else:
        full(DRICI_GRID if a.drici else GRID, a.reps, a.time_limit,
             a.wall_budget or None)
