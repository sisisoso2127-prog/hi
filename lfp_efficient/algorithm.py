"""The algorithm of L. Younsi-Abbaci, *Optimizing a linear fractional function
over an integer efficient set* (RT&A, No 4 (40), Vol. 11, March 2025).

Problem solved
--------------
::

    (P_E)   max  Phi(x) = (U'x + alpha) / (V'x + beta)
            s.t. x in E(P_D)

where ``E(P_D)`` is the set of efficient (Pareto optimal) points of the
multi-objective integer linear program

    (P_D)   "max" { C x : x in D },   D = { x in Z^n_+ : A x <~ b } .

``E(P_D)`` is a union of faces of ``D``: it is non-convex and has no explicit
description, which is what makes ``(P_E)`` a global optimisation problem.  The
whole point of the method is to reach the global optimum *without enumerating*
``E(P_D)``.

Skeleton of one iteration
-------------------------
1. ``P^l_RF``: maximise ``Phi`` over the current truncated region ``D_l``.
   Its value is an **upper bound** on ``Phi`` over every efficient point still
   available, since ``D_l`` contains them all.
2. Test the efficiency of the optimum ``x_l`` (Theorem 1).
   *Efficient* -> the upper bound is attained by an efficient point: stop.
   *Not efficient* -> the test hands over an efficient point ``x~_l``
   dominating it (Ecker & Kouada).
3. ``Q(x~_l)``: best value of ``Phi`` among the points sharing the
   non-dominated vector ``C x~_l`` -- they are all efficient, and the cut of
   step 5 is about to erase them, so their contribution is banked now.
4. Edge exploration: if an edge of ``Gamma_l`` (zero reduced gradient, i.e.
   alternative optima of step 1) carries an efficient integer point, that point
   attains the upper bound -> stop.
5. Sylva-Crema cut: delete ``{ x : C x <= C x~_l }`` from the region and loop.

Termination (Proposition 3): each iteration strictly shrinks the region by at
least one non-dominated criterion vector, and the non-dominated set of a
bounded integer program is finite.

Deviations from the printed pseudo-code
---------------------------------------
The pseudo-code of "Algorithm 2: part 2" is internally inconsistent -- it
stores ``X_opt = x_l`` (the point that was just found *not* to be efficient)
while setting ``Phi_opt = Phi(x~_l)``, and it re-solves ``P_l`` inside the
branch it has just solved.  The implementation below keeps the mathematics of
the paper (upper bound / efficiency test / ``Q`` / cut / edges) but stores the
*efficient* point that realises ``Phi_opt``, and adds one sound early stop:
if the upper bound of step 1 does not beat the incumbent, no remaining
efficient point can either, so the search is over.
"""

from dataclasses import dataclass, field
from time import monotonic
from fractions import Fraction
from typing import Callable, List, Optional, Sequence

from .edges import EdgeCandidate, clean_tableau_at, explore_edges
from .efficiency import (add_dominance_cut, best_with_same_criterion,
                         has_linear_criteria, lower_bounds, repair_to_efficient,
                         spread_weights, test_efficiency, weighted_sum_efficient)
from .milp import (CUTOFF, INTERRUPTED, denominator_stays_positive,
                   solve_fractional_milp)
from .model import FractionalObjective, MOILFP, Model
from .rational import F, fmt
from .simplex import OPTIMAL


@dataclass
class IterationLog:
    """Everything that happened during one pass of the main loop."""

    l: int
    relaxed_point: Optional[List[Fraction]] = None
    upper_bound: Optional[Fraction] = None
    psi: Optional[Fraction] = None
    efficient_point: Optional[List[Fraction]] = None      # x~_l
    criterion_vector: Optional[List[Fraction]] = None     # C x~_l
    best_same_criterion: Optional[List[Fraction]] = None  # argmax of Q(x~_l)
    phi_same_criterion: Optional[Fraction] = None
    edge_candidate: Optional[EdgeCandidate] = None
    #: efficient points cut in one batch before this iteration's step 1
    batched: List[List[Fraction]] = field(default_factory=list)
    incumbent: Optional[List[Fraction]] = None
    incumbent_value: Optional[Fraction] = None
    note: str = ""

    def __str__(self) -> str:
        def pt(v):
            return "(" + ", ".join(fmt(c) for c in v) + ")" if v else "-"
        lines = [f"--- iteration {self.l} " + "-" * 40]
        if self.relaxed_point is not None:
            lines.append(f"  P^{self.l}_RF : x_{self.l} = {pt(self.relaxed_point)}"
                         f"   Phi = {fmt(self.upper_bound)}   (upper bound)")
        if self.psi is not None:
            verdict = "efficient" if self.psi == 0 else f"NOT efficient (psi* = {fmt(self.psi)})"
            lines.append(f"  efficiency test : {verdict}")
        if self.efficient_point is not None:
            lines.append(f"  efficient point x~_{self.l} = {pt(self.efficient_point)}"
                         f"   C x~ = {pt(self.criterion_vector)}")
        if self.best_same_criterion is not None:
            lines.append(f"  Q(x~_{self.l}) : {pt(self.best_same_criterion)}"
                         f"   Phi = {fmt(self.phi_same_criterion)}")
        if self.batched:
            lines.append(f"  batch : cut on {len(self.batched)} extra efficient "
                         f"point(s) from the weighted-sum scalarisation")
        if self.edge_candidate is not None:
            c = self.edge_candidate
            lines.append(f"  edge exploration : efficient point {pt(c.x)} on the edge "
                         f"of column {c.j} at theta = {c.theta}, Phi = {fmt(c.phi)}")
        if self.incumbent is not None:
            lines.append(f"  incumbent : X_opt = {pt(self.incumbent)}"
                         f"   Phi_opt = {fmt(self.incumbent_value)}")
        if self.note:
            lines.append(f"  >> {self.note}")
        return "\n".join(lines)


@dataclass
class Solution:
    """Answer to ``(P_E)``, with a certified gap and the audit trail of the run.

    ``value`` is a **lower** bound attained at a known efficient point -- a real
    solution, not an estimate -- and ``upper_bound`` bounds the optimum from
    above.  When ``proved_optimal`` they coincide; when the run stopped on its
    time budget they do not, and ``gap`` says by how much the answer could
    still be wrong.
    """

    status: str
    x: Optional[List[Fraction]] = None
    value: Optional[Fraction] = None
    #: the efficient points actually generated -- a strict subset of E(P_D)
    explored: List[List[Fraction]] = field(default_factory=list)
    iterations: List[IterationLog] = field(default_factory=list)
    lower_bounds: List[Fraction] = field(default_factory=list)
    #: certified upper bound on the optimum of (P_E)
    upper_bound: Optional[Fraction] = None
    #: the gap at the first round, used to scale the progress made
    initial_gap: Optional[Fraction] = None
    proved_optimal: bool = False

    @property
    def gap(self) -> Optional[Fraction]:
        """``upper_bound - value``: the **absolute** remaining uncertainty.

        Deliberately not ``(UB - value) / |value|``.  A relative-to-incumbent
        gap is a habit from mixed integer programming, where objectives are
        usually bounded away from zero; ``Phi`` is a ratio that can be negative
        or near zero, and the relative form then reports a tiny real gap as a
        four-digit percentage and reads as a broken method.  Scale by
        :attr:`initial_gap` instead -- see :attr:`gap_closed`.
        """
        if self.upper_bound is None or self.value is None:
            return None
        return self.upper_bound - self.value

    @property
    def gap_closed(self) -> Optional[float]:
        """Fraction of the starting uncertainty eliminated, in ``[0, 1]``."""
        gap = self.gap
        if gap is None or self.initial_gap is None:
            return None
        if self.initial_gap == 0:
            return 1.0
        return float(1 - gap / self.initial_gap)

    def report(self, include_iterations: bool = True) -> str:
        out = [str(it) for it in self.iterations] if include_iterations else []
        out.append("=" * 54)
        if self.x is None:
            out.append(f"no optimal solution ({self.status})")
        else:
            pt = "(" + ", ".join(fmt(v) for v in self.x) + ")"
            out.append(f"X_opt = {pt}    Phi_opt = {fmt(self.value)}")
            if self.proved_optimal:
                out.append("proved optimal (the bound meets the solution)")
            else:
                closed = self.gap_closed
                out.append(f"NOT proved optimal: Phi_opt <= {fmt(self.upper_bound)}"
                           f"   gap = {fmt(self.gap)}"
                           + (f"   ({100 * closed:.1f}% of the initial gap closed)"
                              if closed is not None else ""))
            out.append("efficient points generated: " + ", ".join(
                "(" + ", ".join(fmt(v) for v in p) + ")" for p in self.explored))
        return "\n".join(out)


def optimize_over_efficient_set(problem: MOILFP, phi: FractionalObjective,
                                use_edge_exploration: bool = True,
                                batch_cuts_after: Optional[int] = None,
                                max_iterations: int = 1000,
                                time_budget: Optional[float] = None,
                                verbose: bool = False) -> Solution:
    """Solve ``max { Phi(x) : x efficient for (P_D) }`` exactly.

    Parameters
    ----------
    problem
        The multi-objective integer program ``(P_D)``; ``D`` must be non-empty
        and bounded.
    phi
        The decision maker's linear fractional criterion.
    use_edge_exploration
        Walk the zero-reduced-gradient edges (Definition 2) before cutting.
        Switching it off changes nothing to the returned optimum, only to the
        number of iterations.
    batch_cuts_after
        Off by default, and deliberately so -- see *What batching is and is
        not worth* below.  When set to ``k`` and the loop is still open after
        ``k`` iterations, generate ``p + 1`` efficient points in one go by
        weighted-sum scalarisation and cut on all of them at once, then carry
        on.  Requires linear criteria.
    time_budget
        Seconds after which to stop and return the best answer found **with a
        certified gap** instead of running to optimality.  Without it the
        method runs to proven optimality as before.
    verbose
        Print the trace of each iteration as it is produced.

    The certified gap, and where it comes from
    ------------------------------------------
    Step 1 already computes the maximum of ``Phi`` over the truncated region,
    and the loop uses it only to decide whether to stop -- so a valid upper
    bound on the answer is produced every round and thrown away.  It is valid
    because after cutting on ``x^1..x^l`` every efficient point either

    * lies in a removed set ``{x : C x <= C x^s}``, where it is dominated, or
      sits on the slice ``{C x = C x^s}`` whose best ``Phi`` sub-problem ``Q``
      has already folded into ``Phi_opt``; or
    * lies in the region that step 1 searches,

    so ``max_E Phi <= max(Phi_opt, max{Phi(x) : x in D_l})``, the second term
    being exactly what step 1 returns.  It descends as cuts accumulate, and
    when the region empties it equals ``Phi_opt`` and optimality is proved.

    That makes the method **anytime**: stop whenever and report a real solution
    (attained at a known efficient point), a bound, and the distance between
    them.  With a *time_budget* the branch & bound of step 1 is stopped too,
    and its own best open bound is used -- so the answer stays certified even
    when no sub-problem ran to completion.

    Interrupting inside step 1 is the only place the budget can be honoured
    mid-iteration; the efficiency test and ``Q`` are left to finish, since a
    truncated efficiency test yields no usable efficient point.

    What batching is and is not worth
    ---------------------------------
    The run time is (number of step 1 solves) x (size of the region), and the
    region grows by ``p`` binaries and ``p + 1`` rows per cut.  Handing the
    method a good incumbent was measured **not to touch the first factor at
    all** -- given the optimum for free at iteration 1, all three reference
    instances still took exactly as many iterations, because what gets cut is
    decided by the efficiency test on ``x_l``, not by ``Phi_opt``.

    Batching does touch it.  On the heaviest instance shipped here it takes 11
    step 1 solves down to 8 and the run from 27.1 s to 15.7 s.  But the extra
    cuts are a bet that those points are ones the loop would have had to cut
    anyway, and when the bet loses the model has grown for nothing: on
    ``medium n=6`` the solve count does not move and the run goes from 1.45 s
    to 2.20 s.  Over 21 instances the total falls 34.3 s -> 23.6 s, and
    essentially all of that is the one heavy instance.

    Filtering out generated centres that are already inside an existing cut
    was tried and changes nothing -- they are genuinely new non-dominated
    vectors, just not ones on the path.  Raising the trigger to 5 removes the
    tax on the cheap instances and gives most of the win back.

    So it is insurance for the heavy tail, not a general speed-up, and that is
    why it is off unless asked for.
    """
    if batch_cuts_after is not None and not has_linear_criteria(problem):
        raise ValueError(
            "batch_cuts_after needs linear criteria: it generates its extra "
            "cut centres by weighted-sum scalarisation, and a sum of ratios "
            "is not a linear objective")
    deadline = None if time_budget is None else monotonic() + time_budget
    M = lower_bounds(problem)
    # Established once on D: every truncated region is a subset of it, so the
    # verdict carries over to all the sub-problems of the run.
    positive_denominator = denominator_stays_positive(problem.model, phi)
    region = problem.model.copy()
    phi_opt: Optional[Fraction] = None
    x_opt: Optional[List[Fraction]] = None
    explored: List[List[Fraction]] = []
    logs: List[IterationLog] = []

    # cache: efficiency is a property of D, so it is tested once per point
    cache = {}

    def is_efficient(x: Sequence[Fraction]) -> bool:
        key = tuple(x)
        if key not in cache:
            cache[key] = test_efficiency(problem, list(x)).efficient
        return cache[key]

    upper_bound: Optional[Fraction] = None
    initial_gap: Optional[Fraction] = None

    def keep_best_bound(current: Optional[Fraction],
                        candidate: Fraction) -> Fraction:
        """Keep the tightest upper bound seen so far.

        Every bound recorded here is valid on its own, so the smallest of them
        is valid too -- and it is the one worth reporting.  Taking the running
        minimum rather than the latest value matters because the two sources
        are not equally sharp: a completed step 1 returns the exact maximum
        over the region, while an interrupted one returns the largest bound
        still open in its search tree, which is a relaxation value and can sit
        *above* the exact maximum of a larger, earlier region.  Overwriting
        would then let a longer run report a worse bound than a shorter one.
        """
        return candidate if current is None else min(current, candidate)

    def finish(status: str, proved: bool) -> Solution:
        # the iteration logs have already been streamed when verbose, so the
        # closing report only carries the summary
        ub = phi_opt if proved else upper_bound
        sol = Solution(status, x_opt, phi_opt, explored, logs, M,
                       upper_bound=ub, initial_gap=initial_gap,
                       proved_optimal=proved)
        if verbose:
            print(sol.report(include_iterations=False))
        return sol

    batched = False
    cut_vectors: List[List[Fraction]] = []

    for l in range(1, max_iterations + 1):
        log = IterationLog(l)
        logs.append(log)

        # ---- step 0 (optional): one batch of cheap efficient points --------
        # Each centre is efficient by the weighted-sum theorem, so it owes the
        # same Q(x~) as any other efficient centre before its cut: the slice
        # {Z = Z(x~)} it destroys is entirely efficient and has to be banked.
        if batch_cuts_after is not None and not batched and l > batch_cuts_after:
            batched = True
            for w in spread_weights(problem.p):
                point = weighted_sum_efficient(problem, w)
                if point is None:
                    continue
                z = problem.Z(point)
                if z in cut_vectors:
                    continue
                cut_vectors.append(z)
                cache[tuple(point)] = True
                if point not in explored:
                    explored.append(point)
                q = best_with_same_criterion(
                    region, problem, point, phi, cutoff=phi_opt,
                    denominator_positive=positive_denominator)
                if q.feasible and (phi_opt is None or q.objective > phi_opt):
                    x_opt, phi_opt = q.x, q.objective
                region = add_dominance_cut(region, problem, point)
                log.batched.append(point)

        # ---- step 1: upper bound on the truncated region ------------------
        # the incumbent is handed over as a cutoff: the only thing that matters
        # is whether the region still holds something better than Phi_opt
        if deadline is not None and monotonic() > deadline:
            log.note = ("time budget spent: returning the incumbent with its "
                        "certified gap.")
            if verbose:
                print(log)
            return finish(OPTIMAL if x_opt is not None else "infeasible", False)

        relaxed = solve_fractional_milp(region, phi, cutoff=phi_opt,
                                        denominator_positive=positive_denominator,
                                        deadline=deadline)
        if relaxed.status == INTERRUPTED:
            # step 1 did not finish, but its best open bound still bounds the
            # answer from above -- so the result stays certified
            if relaxed.bound is not None:
                candidate = (relaxed.bound if phi_opt is None
                             else max(phi_opt, relaxed.bound))
                upper_bound = keep_best_bound(upper_bound, candidate)
            log.upper_bound = upper_bound
            log.note = ("time budget spent inside P^l_RF: stopping with the "
                        "bound its unexplored nodes still guarantee.")
            if verbose:
                print(log)
            return finish(OPTIMAL if x_opt is not None else "infeasible", False)
        if relaxed.status == CUTOFF:
            log.note = (f"the maximum of Phi over the truncated region is "
                        f"<= Phi_opt = {fmt(phi_opt)}: no remaining efficient "
                        "point can improve the incumbent. Stop.")
            if verbose:
                print(log)
            return finish(OPTIMAL, True)
        if not relaxed.feasible:
            log.note = ("P^l_RF is infeasible: the region is exhausted, every "
                        "non-dominated vector has been generated. Stop.")
            if verbose:
                print(log)
            return finish(OPTIMAL if x_opt is not None else "infeasible",
                          x_opt is not None)

        x_l = relaxed.x[:problem.n]
        log.relaxed_point, log.upper_bound = x_l, relaxed.objective
        upper_bound = keep_best_bound(upper_bound, relaxed.objective)

        if phi_opt is not None and relaxed.objective <= phi_opt:
            log.note = (f"upper bound {fmt(relaxed.objective)} <= Phi_opt "
                        f"{fmt(phi_opt)}: no remaining efficient point can "
                        "improve the incumbent. Stop.")
            if verbose:
                print(log)
            return finish(OPTIMAL, True)

        # ---- step 2: efficiency test (Theorem 1) --------------------------
        test = test_efficiency(problem, x_l)
        log.psi = test.psi
        if test.efficient:
            cache[tuple(x_l)] = True
            if phi_opt is None or relaxed.objective > phi_opt:
                x_opt, phi_opt = x_l, relaxed.objective
            explored.append(x_l)
            log.efficient_point = x_l
            log.criterion_vector = problem.C(x_l)
            log.incumbent, log.incumbent_value = x_opt, phi_opt
            log.note = ("the maximiser of the upper bound is itself efficient: "
                        "it is the global optimum of (P_E). Stop.")
            if verbose:
                print(log)
            return finish(OPTIMAL, True)

        # The maximiser of the test dominates x_l but, with fractional
        # criteria, need not be efficient itself: the k-th term of the
        # objective carries a factor D_k(x) that varies from point to point.
        # Walking the dominance chain settles it, and costs one confirming
        # test when the chain is already at its end -- which is always the
        # case for linear criteria.
        x_tilde = repair_to_efficient(problem, test.witness)
        cut_vectors.append(problem.Z(x_tilde))
        cache[tuple(x_tilde)] = True
        explored.append(x_tilde)
        log.efficient_point = x_tilde
        log.criterion_vector = problem.C(x_tilde)

        # ---- step 3: Q(x~) -- best Phi on the slice about to be cut -------
        q = best_with_same_criterion(region, problem, x_tilde, phi, cutoff=phi_opt,
                                     denominator_positive=positive_denominator)
        if q.feasible:
            log.best_same_criterion, log.phi_same_criterion = q.x, q.objective
            # C q.x = C x~ is non-dominated, so q.x is efficient as well
            cache[tuple(q.x)] = True
            if q.x not in explored:
                explored.append(q.x)
            if phi_opt is None or q.objective > phi_opt:
                x_opt, phi_opt = q.x, q.objective
        if initial_gap is None and phi_opt is not None and upper_bound is not None:
            initial_gap = upper_bound - phi_opt
        log.incumbent, log.incumbent_value = x_opt, phi_opt

        # ---- step 4: edges of Gamma_l (alternative optima) ----------------
        # Read at a basis of the truncated region itself, not at the branch &
        # bound node that produced x_l: that node's bound rows pin variables and
        # leave the edges no room to step.  ``gamma_j = 0`` keeps Phi constant
        # along the whole edge whether or not x_l maximises the *relaxation*, so
        # an integer point found there scores exactly the upper bound.
        tableau = (clean_tableau_at(region, relaxed.x)
                   if use_edge_exploration else None)
        if tableau is not None:
            candidate = explore_edges(tableau, problem, phi, is_efficient,
                                      n_model=problem.n)
            if candidate is not None:
                log.edge_candidate = candidate
                if candidate.x not in explored:
                    explored.append(candidate.x)
                if phi_opt is None or candidate.phi > phi_opt:
                    x_opt, phi_opt = candidate.x, candidate.phi
                log.incumbent, log.incumbent_value = x_opt, phi_opt
                # Phi is constant along an edge of Gamma_l, so the candidate is
                # expected to reach the upper bound exactly.  The stop is
                # nevertheless conditioned on that equality being *verified*:
                # should the walk ever return a lesser point, it is kept as a
                # plain incumbent and the normal cutting loop carries on.
                if candidate.phi == relaxed.objective:
                    log.note = ("an efficient point attains the upper bound along "
                                "an edge of Gamma_l: it is the global optimum. Stop.")
                    if verbose:
                        print(log)
                    return finish(OPTIMAL, True)

        # ---- step 5: Sylva-Crema truncation -------------------------------
        region = add_dominance_cut(region, problem, x_tilde)
        log.note = (f"cut off every x with C x <= C x~_{l}; "
                    f"go to iteration {l + 1}")
        if verbose:
            print(log)

    raise RuntimeError("iteration limit reached")
