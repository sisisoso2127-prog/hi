# lfp-efficient

Exact optimization of a **linear fractional function over the integer efficient
set** of a multi-objective program — pure Python, no dependency, rational
arithmetic throughout.

Reference implementation of

> Leila Younsi-Abbaci, *Optimizing a linear fractional function over an integer
> efficient set*, **Reliability: Theory & Applications**, No 4 (40), Vol. 11,
> March 2025,

generalised to criteria that are themselves ratios (MOILFP).

## The problem

```
(P_D)   "max"  Z_k(x) = (c_k'x + a_k) / (d_k'x + b_k) ,  k = 1..p
        s.t.   x in D = { x in Z^n_+ : A x <~ b }

(P_E)   max  Phi(x) = (U'x + alpha) / (V'x + beta)
        s.t. x in E(P_D)                    <- the *efficient* points of (P_D)
```

`E(P_D)` is a union of faces of `D`: non-convex, with no explicit description,
so `(P_E)` is a global optimisation problem with many local optima in general.
The method reaches the global optimum **without enumerating `E(P_D)`** — on the
mid-size instances shipped here it gets there after generating 9–16 % of it.
Linear criteria are the degenerate case `d_k = 0`, `b_k = 1`, so the paper's
program is a special case and `MOILP` is a one-line constructor over `MOILFP`.

## Install

```bash
pip install git+https://github.com/sisisoso2127-prog/hi.git
```

Python 3.10+. Nothing else is required, and that is a design property rather
than an omission: the package carries its own exact simplex and branch & bound
over `fractions.Fraction`, so every test — efficiency, a vanishing reduced
gradient, an integral vertex — is an exact comparison with no tolerance
anywhere. CI imports it on a bare interpreter to keep that true.

## Quick start

```python
from lfp_efficient import Model, MOILP, FractionalObjective, LE
from lfp_efficient import optimize_over_efficient_set

D = Model(2).add([-2, 1], LE, 0).add([6, 1], LE, 21).add([-2, 4], LE, 6)
problem = MOILP(D, criteria=[[1, -2], [-1, 4]])
phi = FractionalObjective(U=[1, 1], V=[5, 1], alpha=-1, beta=-1)

solution = optimize_over_efficient_set(problem, phi)
print(solution.x, solution.value)        # [3, 3]  5/17
print(solution.proved_optimal)           # True
```

The answer is a `Fraction`, exact. `solution` also carries a certified upper
bound and the audit trail of every iteration; with `time_budget=...` the method
becomes **anytime** — stop whenever and get a real solution attained at a known
efficient point, a bound, and the absolute gap between them.

## What is verified, and how

| | |
|---|---|
| `python tests/test_lfp_efficient.py` | 39 tests, no pytest needed (it runs under pytest too) |
| `python examples/paper_example.py` | reproduces §4 of the paper: `X_opt = (3,3)`, `Phi_opt = 5/17` |
| `python examples/large_example.py` | instances up to \|D\| = 34635, each cross-checked against an independent scan |
| `python examples/fractional_example.py` | fully fractional criteria, checked against Definition 1 point by point |
| `python examples/scaling.py` | certified optimal up to `n = 20` (slow; opt-in in CI) |

Correctness rests on **four independent references**, not on the algorithm
agreeing with itself: exhaustive enumeration, a repeated-cut enumeration of the
non-dominated set, a sorted scan with dominance tests, and `certify_optimum`,
which settles 362293 challengers with 3 exact efficiency tests.

## Where the code reads differently from the paper

The long-form notes live in **[`lfp_efficient/README.md`](lfp_efficient/README.md)**:
the full method, the performance work (two orders of magnitude on `n = 4`), the
scaling study, and — stated rather than papered over — every place the
implementation departs from the printed text. The short version:

- **The printed "Algorithm 2: part 2" is internally inconsistent.** It stores
  `X_opt = x_l`, the point just shown *not* to be efficient, while setting
  `Phi_opt = Phi(x~_l)`. The implementation keeps the mathematics and fixes the
  bookkeeping.
- **The paper's own numerical example violates the standing assumption of
  linear fractional programming** that `V'x + beta > 0` on the feasible set:
  `x = (0,0)` is in `D` and gives `-1`. The solver splits the feasible set along
  the sign of the denominator and solves both branches.
- **Ecker & Kouada does not transfer to ratios.** With linear criteria the
  maximiser of the efficiency test is itself efficient; with fractional ones it
  only *dominates*, because the k-th term carries a factor `D_k(x)` that varies
  from point to point. The dominance chain is walked, and a test measures that
  every linear chain is one step while a share of fractional ones are longer.
- **The edge step of Definition 2 is nearly inert**, measured rather than
  assumed: of 174 zero-gradient columns over 80 instances only 8 move `x` at
  all, and the step fires on about one instance in eighty. It is implemented,
  repaired and kept — but the README says plainly that it does not make the
  method faster.

## Two additions the paper does not have

**A criterion-space search** (`optimize_in_criterion_space`). The paper cuts in
decision space, where "delete `{x : Z(x) <= Z(x~)}`" is a disjunction needing
`p` binaries and `p+1` big-M rows per cut — so the region grows every round and
the late sub-problems dominate the cost. Keeping the remainder as a list of
boxes in criterion space instead makes every sub-problem `D` plus a few
ordinary linear rows, and the model never grows:

| | decision space | criterion space | |
|---|---:|---:|---:|
| 35 instances, total | 40.65 s | **8.06 s** | **5.05×** |
| heaviest `n=10` | 23.47 s | **2.61 s** | **9.01×** |

Same optimum everywhere, both proved optimal, faster on 34 of 35 — and the
margin widens with difficulty. It **widens with the number of criteria** too,
which is where the gap becomes a different order of magnitude: at `p = 5` on
six variables the paper's method took 110.95 s against **0.64 s**.

| criteria | decision space | criterion space | |
|---:|---:|---:|---:|
| `p = 2` | 0.61 s | 0.35 s | 1.74× |
| `p = 3` | 1.02 s | 0.34 s | 2.99× |
| `p = 4` | 4.21 s | 0.61 s | 6.92× |
| `p = 5` | 112.77 s | 1.09 s | **103×** |
| `p = 6` | 26.15 s | 1.08 s | 24.3× |

It supports fractional criteria too, and keeps the certified gap and
`time_budget`. Both methods stay: this package is a reference implementation of
the paper, and this search is not in the paper.

**Batch cutting** (`batch_cuts_after`) cuts on several cheaply generated
efficient points at once, taking the heaviest instance from 11 step-1 solves to
8 (27.1 s → 15.7 s). It is **off by default**, because on instances that were
never hard it only grows the model — the notes carry the losses alongside the
win.

## Layout

| Path | Contents |
|---|---|
| `lfp_efficient/` | the package — and its long-form notes in `README.md` |
| `examples/` | the paper's example, larger instances, the scaling study, the fractional case |
| `tests/` | the test suite, runnable with a bare interpreter |
| `studies/` | unrelated earlier experiments kept for the record (these use NumPy) |

## License

MIT — see [`LICENSE`](LICENSE).
