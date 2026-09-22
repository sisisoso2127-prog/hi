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
| `python tests/test_lfp_efficient.py` | 50 tests, no pytest needed (it runs under pytest too) |
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

## Two additions the paper does not have, and one that did not earn its keep

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

**A hybrid** (`optimize_hybrid`) runs the paper's loop and, if it has not
closed after `switch_after` iterations, replays its cuts as a box list and
carries its incumbent over — so nothing proved is proved twice. Measured on 24
instances against the better of the two pure methods on each:

| | total | within 15% of the better |
|---|---:|---:|
| `switch_after = 3` | 13.14 s | 4 of 24 |
| `switch_after = 1` | 8.47 s | 23 of 24 |
| pure box search | **8.11 s** | — |
| the paper's method | 268.61 s | — |

Read that honestly: at `switch_after = 1` the hybrid **ties** the pure box
search (8.47 s against 8.11 s, and 1.47 s against 1.54 s on a family where the
paper's method closes in one iteration — two ~4% differences pointing opposite
ways). It is insurance for the one-iteration case, not a third method that
beats both. `switch_after = 3`, the first default, was measurably wrong.

**An exact–metaheuristic hybrid** (`optimize_hybrid_metaheuristic`) runs a
Pareto local search first and hands the box search a **verified efficient**
incumbent. 9.50 s → 7.60 s over 18 instances (1.25×), reaching 1.56× on the
heaviest and losing only where the exact search already took milliseconds. The
ceiling, measured by handing over the true optimum for free, is 1.42×.

The contrast is the interesting part: the same free optimum saves **zero
iterations** in the paper's method, because what it cuts is decided by the
efficiency test and not by the incumbent. The worth of a heuristic incumbent
depends on which exact method it is hybridised with.

**An early exit on a hopeless best bound.** The box loop pops from a heap
ordered by inherited bound descending, and used to solve every box it popped.
But once the *best* bound still open fails to beat the incumbent, so does every
other, and the whole remaining tail is busywork. Measured before changing
anything: 198 of 733 solves (27 %) had such a bound — and on **every one of the
18 instances** that count equalled the number of boxes still open when the
condition first fired. The waste is exactly a tail, never scattered, which is
what the heap order predicts. Stopping there takes **733 boxes to 549, 25 %
fewer**, same answers, still proved optimal.

It is **not a speed-up**: 7.44 s → 7.27 s at per-instance minima over five
repeats, with per-instance ratios scattered from 0.77× to 1.36×. Since the
change can only remove work, anything under 1.00× is machine noise — and the
noise is bigger than the effect. That is a confirmation rather than a
disappointment: the tail boxes are the *cheapest* ones, each killed by the root
relaxation the moment it sees the cutoff. A quarter of the sub-problems were
genuinely wasted and worth almost nothing.

And the same measurement settles a question worth asking before stacking
ideas: on the metaheuristic-seeded hybrid this exit cuts only **4 %** of boxes
(562 → 539), against 25 % unseeded. The seed and the exit are **substitutes,
not complements** — a good incumbent arrives early enough that the hopeless
tail barely forms.

**Cutting on a point the search already proved efficient.** Splitting a profile
of the work by operation put the cost somewhere unexpected: the efficiency test
is **52 % of all simplex pivots on 146 calls** — 58 pivots each against 11.8
for a box solve. So the question stopped being "make the test cheaper" and
became "pay it less often".

When a box's maximiser `x_b` is inefficient the search runs that test to find an
efficient point dominating it. But it already holds a list of points it has
*proved* efficient, and finding one that dominates `x_b` is arithmetic rather
than an integer program. Measured before building: one does on **21 of 146
tests (14 %), and on 17 % of the inefficient cases**.

| | base | shortcut |
|---|---:|---:|
| simplex pivots | 16169 | **14602** (−9.7 %) |
| boxes solved | 549 | **534** (−15) |
| optima | — | identical, all proved |

The boxes went *down*, which was not the prediction. Among the remembered
points that dominate `x_b` the highest is taken, and `{Z ≤ Z(centre)}` removes
more of the box the higher its centre — so skipping the test and cutting better
compound instead of trading off. One instance gained a box; the rest lost or
held.

Soundness rests on one invariant: **every point the search remembers is
efficient.** `explored` holds the centres it cut on and the `Q` maximisers
beside them, and a `Q` maximiser shares its centre's criterion vector, so it is
efficient whenever the centre is. A single inefficient member would let the
search cut on a dominated point and delete the true optimum, so the invariant
is a test rather than an argument — 0 failures over 34 remembered points, and
the shortcut itself is checked against the independent scan rather than against
a box count, since it deliberately changes which boxes are produced.

**Warm-starting a box from its parent.** A box's region is its parent's plus a
handful of rows, yet every box solved its root relaxation from scratch — and
`solve_standard_form` needs a phase I whenever a row is not covered by a slack,
which every `e_k ≥ 1` row forces. Measured first, before anything was built:
**4992 of 18193 simplex pivots (27 %) were phase I inside cold roots**, and 788
of 824 roots (96 %) paid one. Phase I costs 6.3 pivots per root against 2.3 for
phase II — the same "five sixths thrown away" the package already measured
*inside* a branch-and-bound tree, one level up.

`add_linear_row` is the general form of the branch and bound's own
`add_bound_row`: it appends `coeffs·x ≤ rhs` to a solved tableau and eliminates
every basic variable in it, which one pass does because a basic variable's
column is a unit vector. The parent's basis then stays a basis once the new
slacks join it, only the new rows can be primal infeasible, and the existing
dual restoration fixes that.

| | base | warm |
|---|---:|---:|
| simplex pivots | 18193 | **16169** (−11.1 %) |
| cold root relaxations | 824 | **307** (−63 %) |
| boxes, MILP programs, optima | — | identical |

**The ceiling was never 27 %.** Only a box with a parent can inherit one, and
that is 63 % of roots — the rest are the 146 efficiency tests and 125 `Q`
solves, which extend `D` rather than another box, and `D`'s own basis is free
(all its rows are `≤` with non-negative right-hand sides, so the crash basis
covers them and phase I is skipped outright). So the reachable ceiling was
~17 %, and 11.1 % of it survives after the restoration's own 1583 pivots.

**517 warm starts, 0 stalls** — the cold fallback never fired once. It exists
anyway, because a restoration that gives up must cost time and never
correctness.

The gain grows with `p` (28 % at `p=7`, 1–7 % at `p=3`), which is what more
rows per box predicts, and two tiny instances lose a little where the
restoration costs more than the phase I it replaces.

**Not rediscovering Ecker & Kouada once per box.** When a box's maximiser
fails the efficiency test, the search needs an efficient point to cut on
instead, and called `repair_to_efficient` on the test's witness. That walk
*begins* with a full efficiency test on the witness — and with linear criteria
the answer is known in advance. The test maximises `Σₖ(Zₖ(y) − Zₖ(x*))`, a
strictly positive weighted sum of `Z` up to a constant, over
`{y ∈ D : Z(y) ≥ Z(x*)}`. If some `z` dominated a maximiser `y*`, then
`Z(z) ≥ Z(y*) ≥ Z(x*)`, so `z` lies in that same region and scores strictly
higher — contradiction. No maximiser is dominated, ties included.

Checked before being relied on: **349 witnesses of failed tests across 60
linear-criteria instances, none of them dominated.** `efficient_dominator`
takes the shortcut when the criteria are linear and falls back to the walk when
they are not — with a ratio the `Dₖ(y)` factor breaks monotonicity in `Z` and a
dominating point can score lower.

| | base | shortcut |
|---|---:|---:|
| integer programs | 1466 | **1341** (−8.5 %) |
| branch-and-bound nodes | 8105 | **7618** (−6.0 %) |
| boxes solved, optima | — | identical |

**A note on how that was measured, which matters more than the 6 %.** The
wall-clock A/B for this change reads 0.96× — *slower* — and is faster on 9 of
18 instances. That cannot be causal: removing 125 integer programs cannot cost
time. This machine's noise floor is roughly ±30 % per instance even at
per-instance minima over five repeats, so it simply cannot resolve a 6 %
effect. The table above therefore counts **deterministic work** — programs and
nodes — which is the same number on every run and every machine.

The prediction was wrong too, and in an instructive way. The profile that
motivated the change showed the efficiency test at 18.4 ms/call, the dearest
operation in the search; but what the shortcut removes is the *repair's* test
at 7.68 ms — the cheap kind, because proving `θ = 0` on a point already known
efficient is easier than searching for a dominator. 8.5 % of the programs are
only 6.0 % of the nodes. Reasoning from the mean cost of a call would have
overstated this by half.

The shortcut is used at all four sites that pass a failed test's witness, so
the paper's own method gets the same saving and the comparison between the two
stays fair.

**The augmented weighted Tchebychev program**
(`augmented_tchebychev_efficient`) is the scalarisation the surrounding
literature reaches for — Chaabane, Brahmi and Ramdani (2012) optimise over an
integer efficient set with it, Younsi-Abbaci and Moulai (2021) use it over the
Pareto front. It is implemented here as a **generator of certified efficient
points**, and measured against the two generators already in the package.

It is strictly stronger than the weighted sum, which is the reason the
literature prefers it. A weighted sum can only maximise at a *supported*
efficient point; varying the Tchebychev weights reaches unsupported ones too.
Over 20 random instances, from the same grid of 27 weight vectors:

| | efficient points reached | of which unsupported |
|---|---:|---:|
| weighted sum | 66 / 175 | **0 / 54** |
| augmented Tchebychev | 112 / 175 | 29 / 54 |

and every one of the 540 optima it returned was efficient, none merely weakly
so — which is what the augmentation `rho` buys. Those are the numbers
`examples/tchebychev_study.py` prints; a wider sweep over 40 instances and 1080
programs gives the same picture (216/369 against 135/369, 56 unsupported
against 0, and again no inefficient optimum).

**As a seed for the box search it loses, and the table says why.** Against the
same 18 instances used for the metaheuristic above:

| seed | total | seed cost | exactly optimal |
|---|---:|---:|---:|
| none (pure box search) | 9.64 s | — | — |
| Pareto local search | **7.97 s** (1.21×) | 0.57 s | 12 of 18 |
| augmented Tchebychev | 13.50 s (0.71×) | 4.25 s | 5 of 18 |

Both halves of the loss are structural, not an artefact of this
implementation. The cost is `p + 1` integer programs, each carrying an extra
continuous variable and `p` extra rows; the ideal point is not the culprit
(0.01–0.13 s of the seed, against 0.13–0.85 s for the scalarisations
themselves). And the quality gap is the point: the Tchebychev program steers by
criterion-space geometry relative to `z*` — **it never looks at `Phi`**. The
local search is guided by `Phi` at every move. A generator of a good spread of
efficient points is not the same thing as a generator of good `Phi`.

That ordering is stable but the size of the win is not, and the two should not
be confused. On the looser family in `examples/tchebychev_study.py` the Pareto
seed is *exactly optimal on 15 of 24 instances* and still buys only 1.05×,
because that search is not bound-limited — seed **quality** is not seed
**value**, which is the same lesson as the zero-iteration result above.
Tchebychev loses on every family tried (0.73×, 0.91×); the Pareto seed ties or
wins on every one.

The same split shows up when the two are compared as front generators rather
than as seeds. On 20 instances at `n = 4`, Tchebychev reached 64% of `E(P_D)`
in 2.1 s and the Pareto archive 99% in 0.04 s. But the archive filters by
dominance among the points it has *seen*, which is not the exact test: over a
wider sweep one member in roughly 2000 turned out not to be efficient, while
every Tchebychev optimum is efficient by construction. Rare is not never, and
that is precisely why `metaheuristic_incumbent` puts the archive's best through
`test_efficiency` before any bound is allowed to cross into the exact search.

A note on how this was measured: an earlier version of the harness scored a
seed by the ratio `Phi(seed) / Phi*`, which **inverts when `Phi*` is negative**,
as it is on some of these instances. The tables above use the non-negative gap
`(Phi* - Phi(seed)) / |Phi*|`, zero exactly when the seed is optimal.

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
| `studies/` | earlier metaheuristic experiments kept for the record (these use NumPy) |
| `docs/` | the write-up: every method, every measurement, and every negative result |

## License

MIT — see [`LICENSE`](LICENSE).
