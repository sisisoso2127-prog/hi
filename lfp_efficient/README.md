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
python3 tests/test_lfp_efficient.py      # 12 tests, incl. 60 random instances
```

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

**Validation.** `tests/` checks the paper's numbers step by step (`M = (−3,−3)`,
`x_1 = (0,0)` with `Phi = 1`, `psi* = 2`, `X_opt = (3,3)`, `Phi_opt = 5/17`,
`|D| = 11`, `|E(P_D)| = 7`) and cross-checks 60 random bi-/tri-objective
instances against exhaustive enumeration of the efficient set.

## Modules

| File | Contents |
|------|----------|
| `rational.py` | exact arithmetic helpers |
| `simplex.py` | tableau simplex, linear and linear-fractional pricing |
| `milp.py` | branch & bound; sign-split fractional MILP |
| `model.py` | `Model`, `MOILP`, `FractionalObjective` |
| `efficiency.py` | Theorem 1 test, lower bounds `M_i`, Sylva–Crema cut, `Q(x~)` |
| `edges.py` | reduced gradient, `Gamma_l`, `theta0`, edge walk |
| `algorithm.py` | the main loop, with a full iteration trace |
| `enumeration.py` | brute-force reference used by the tests |
