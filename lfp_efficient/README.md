# Optimizing a linear fractional function over an integer efficient set

Pure-Python, dependency-free, exact-arithmetic implementation of

> Leila Younsi-Abbaci, *Optimizing a linear fractional function over an integer
> efficient set*, **Reliability: Theory & Applications**, No 4 (40), Vol. 11,
> March 2025.

## The problem

Given the multi-objective integer linear program

```
(P_D)   "max"  Z_i = C_i x ,  i = 1..p
        s.t.   x in D = { x in Z^n_+ : A x <~ b }
```

and a decision maker's linear fractional criterion, solve

```
(P_E)   max  Phi(x) = (U'x + alpha) / (V'x + beta)
        s.t. x in E(P_D)                     <- the *efficient* points of (P_D)
```

`E(P_D)` is a union of faces of `D`: non-convex, with no explicit description.
`(P_E)` is therefore a global optimisation problem with, in general, many local
optima. The method reaches the global optimum **without enumerating `E(P_D)`**.

## One iteration

| Step | What it does | Where it lives |
|------|--------------|----------------|
| 1 | `P^l_RF`: maximise `Phi` over the truncated region `D_l` → an **upper bound** on `Phi` over every remaining efficient point | `milp.solve_fractional_milp` |
| 2 | Efficiency test of the optimum `x_l` (Theorem 1, Isermann). `psi* = 0` ⇔ efficient; otherwise the test returns an efficient point `x~_l` dominating it | `efficiency.test_efficiency` |
| 3 | `Q(x~_l)`: best `Phi` among the points sharing the non-dominated vector `C x~_l` — all efficient, and all about to be cut away | `efficiency.best_with_same_criterion` |
| 4 | Walk the edges of `Gamma_l = { j : gamma_j = 0 }` (alternative optima, Definition 2). An efficient point there attains the upper bound ⇒ stop | `edges.explore_edges` |
| 5 | Sylva–Crema cut: delete `{ x : C x <= C x~_l }` and loop | `efficiency.add_sylva_crema_cut` |

Termination (Proposition 3): each iteration removes at least one non-dominated
criterion vector, and that set is finite for a bounded integer program.

## Usage

```python
from lfp_efficient import Model, MOILP, FractionalObjective, LE
from lfp_efficient import optimize_over_efficient_set

D = (Model(2)
     .add([-2, 1], LE, 0)
     .add([6, 1], LE, 21)
     .add([-2, 4], LE, 6))

problem = MOILP(D, criteria=[[1, -2], [-1, 4]])          # max Z1, max Z2
phi = FractionalObjective(U=[1, 1], V=[5, 1], alpha=-1, beta=-1)

solution = optimize_over_efficient_set(problem, phi, verbose=True)
print(solution.x, solution.value)        # [3, 3]  5/17
```

```
python3 examples/paper_example.py        # reproduces section 4 of the paper
python3 examples/large_example.py        # larger instances, up to |D| = 31833
python3 tests/test_lfp_efficient.py      # 15 tests, incl. 72 random instances
```

## Larger instances

The paper's illustration has 11 feasible points, small enough that any method
works. `examples/large_example.py` runs the algorithm on instances up to
`n = 10` variables and `p = 3` criteria, every answer cross-checked against an
independent exact reference:

| instance | \|D\| | `Phi_opt` | iter | generated | algorithm | check |
|---|---:|---:|---:|---:|---:|---:|
| medium `n=4` | 106 | 15/7 | 5 | 5 of 31 efficient (16%) | 0.41 s | 0.01 s |
| medium `n=5` | 250 | 15/7 | 5 | 5 of 54 efficient (9%) | 0.33 s | 0.03 s |
| medium `n=6` | 629 | 2 | 8 | 8 of 71 efficient (11%) | 1.46 s | 0.07 s |
| large `n=10` | 31833 | 43/11 | 1 | 1 | 0.01 s | 6.42 s |
| hard `n=10` | 4994 | 30/11 | 3 | 3 | 0.38 s | 1.28 s |
| hardest `n=10` | 34635 | 13/23 | 11 | 11 | 26.8 s | 7.62 s |

The three `n = 10` rows show the spread in difficulty. In the *large* one the
maximiser of `Phi` over `D` happens to be efficient, so the first efficiency
test settles the problem — 31833 feasible points solved in 0.01 s, six hundred
times faster than merely scanning them. In the *hard* and *hardest* ones the
criteria reward large `x` while `Phi` rewards small `x`, so the maximiser of
`Phi` is dominated and the algorithm has to cut its way through several
non-dominated vectors. That is the regime the method is written for, and the
one where the cost sits: each Sylva–Crema cut adds `p` binaries and `p+1` rows,
so the sub-problems grow with the iteration count.

The middle rows are where the paper's claim is visible: the optimum is reached
after generating 9–16 % of `E(P_D)`.

## Three independent references

Trusting a single implementation to check itself proves nothing, so the package
carries three reference methods that share no code path with the algorithm:

| function | how it works | cost |
|---|---|---|
| `enumerate_efficient_set` | enumerate the box, filter by Definition 1 | `O(\|D\|^2)` pairwise dominance |
| `maximize_by_full_enumeration` | generate *every* non-dominated vector by repeated cuts, maximise `Phi` on each slice | this is the naive method the paper avoids |
| `best_over_efficient_set_by_scan` | sort `D` by decreasing `Phi`, return the first point surviving a dominance test | a handful of tests in practice |

The third one is the practical verifier: the first efficient point in
`Phi`-decreasing order *is* the optimum of `(P_E)`, so the efficient set never
has to be built. On the paper's example it answers after 3 dominance tests; on
the 31833-point instance, after 1.

## Design notes

**Exact rational arithmetic everywhere.** The algorithm rests on exact
predicates — *is `psi*` zero?*, *is `gamma_j` zero?*, *is this component an
integer?* With floats each of those needs a tolerance and silently misclassifies
degenerate vertices. `fractions.Fraction` costs nothing at these sizes.

**A hand-written tableau simplex instead of a solver.** The method is written in
the language of the simplex tableau: it needs the basis `B_k`, the updated
columns `y_{k,j} = B_k^{-1} a_{k,j}` and the reduced gradient `gamma_k`. A
black-box MILP solver returns none of that. Linear and linear fractional
objectives share the pivoting machinery and differ only in their pricing rule
(`LinearPricing` / `FractionalPricing`, the latter being Cambini–Martein's
reduced gradient, i.e. Theorem 3 of the paper).

**Denominators that change sign.** Linear fractional programming assumes
`V'x + beta > 0` on the feasible set — that is what makes `Phi` pseudo-concave
and the branch-and-bound bound valid. *The paper's own example violates it*:
`x = (0,0)` is in `D` and gives `5·0 + 0 − 1 = −1 < 0`; it is exactly where the
paper reads `Phi_sup = Phi(0,0) = 1`. `solve_fractional_milp` therefore splits
the feasible set along the sign of the denominator and solves both branches
(on the negative one, `Phi = (−U'x−alpha)/(−V'x−beta)` is a proper LFP), taking
the better. Points where the denominator vanishes are excluded — `Phi` is
undefined there.

**Deviations from the printed pseudo-code.** "Algorithm 2: part 2" is internally
inconsistent — it stores `X_opt = x_l`, the point just shown *not* to be
efficient, while setting `Phi_opt = Phi(x~_l)`, and re-solves `P_l` inside the
branch that has just solved it. The implementation keeps the mathematics
(bound / test / `Q` / cut / edges) and fixes the bookkeeping: the stored point
is the efficient one that realises `Phi_opt`. One sound early stop is added — if
the upper bound of step 1 does not beat the incumbent, no remaining efficient
point can, so the search ends there. The edge walk only ever *accelerates*: every
candidate is re-validated against `D` and re-tested for efficiency before use,
and the early stop it triggers is conditioned on the value actually matching the
upper bound.

**Performance.** The method is MILP-bound: every iteration maximises `Phi` over
a region carrying `p` extra binaries and `p+1` extra rows per cut already made,
and that one sub-problem is 94 % of the run time. Profiling drove every choice
below; on the `n = 4` instance they compound to **351 s → 0.41 s**.

*Making each linear program cheaper.* Reduced costs are **maintained** through
the pivots rather than recomputed from `c_B' B^-1 A` — `O(n)` per pivot instead
of `O(mn)`, which with exact rationals dominated everything. Pivot updates skip
the zeros of the pivot row. A **crash basis** puts the `≤`-slacks straight into
the initial basis, so phase I is skipped whenever they already cover every row.

*Making the sub-problems smaller.* The Sylva–Crema rows are emitted in `≤` form
so their slacks feed that crash basis, and the `y_i <= 1` rows of equation (5)
are dropped — they are redundant, since `y_i >= 2` only imposes a stronger
requirement and leaves the union over the integer `y` unchanged. `Q(x~)` is
solved over `D` rather than over the truncated region (the criterion slice is
provably untouched by the cuts), which keeps every accumulated binary out of
it. And when `V'x + beta > 0` holds on the whole relaxation, the sign split is
skipped, halving every fractional sub-problem.

*Searching less.* The branch & bound takes an **objective cutoff**: the
algorithm never needs the exact maximum over the truncated region, only whether
it beats the incumbent, so handing the incumbent over prunes most of the tree
in precisely the late iterations where the accumulated binaries would bite.

*Not re-solving what the parent already solved.* This was the big one. A child
node differs from its parent by a single bound row, yet solving it from scratch
paid a full phase I — measured at **20.6 phase-I pivots per node against 4.1
phase-II ones**, five sixths of the work thrown away. Children are now warm
started: the branch row is appended to the parent's optimal tableau and a
dual-simplex restoration repairs the one infeasible row, in **2.8 pivots on
average**. A child whose restoration stalls falls back to a cold solve, so the
warm start can cost time but never correctness — and a test forces every
restoration to stall and checks that warm and cold agree on every instance. On
a hard `n = 10` instance: 2996 children, zero stalls, 21 cold solves in total,
and 657 children proved empty by the dual infeasibility certificate without
solving anything at all.

**Validation.** `tests/` checks the paper's numbers step by step (`M = (−3,−3)`,
`x_1 = (0,0)` with `Phi = 1`, `psi* = 2`, `X_opt = (3,3)`, `Phi_opt = 5/17`,
`|D| = 11`, `|E(P_D)| = 7`), verifies that the three reference methods agree,
and cross-checks 60 random bi-/tri-objective instances against exhaustive
enumeration plus 12 larger ones (up to 404 feasible points) against the
`Phi`-ordered scan. Eight `n = 10` instances with up to 34635 feasible points
are verified against the scan in `examples/large_example.py`.

## Modules

| File | Contents |
|------|----------|
| `rational.py` | exact arithmetic helpers |
| `simplex.py` | tableau simplex, linear and linear-fractional pricing, warm start |
| `milp.py` | warm-started branch & bound; sign-split fractional MILP |
| `model.py` | `Model`, `MOILP`, `FractionalObjective` |
| `efficiency.py` | Theorem 1 test, lower bounds `M_i`, Sylva–Crema cut, `Q(x~)` |
| `edges.py` | reduced gradient, `Gamma_l`, `theta0`, edge walk |
| `algorithm.py` | the main loop, with a full iteration trace |
| `enumeration.py` | the three independent reference methods used by the tests |
