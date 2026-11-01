"""Exactly how far the slice machinery reaches, and what sits just outside it.

:mod:`lfp_efficient.complete` turns the front into ``E(P_D)`` by enumerating,
for each non-dominated ``v``, the set ``{x in D : Z(x) = v}``.  That is cheap
because for a linear fractional criterion the condition ``Z_k(x) = v_k`` is a
single linear equation.  This module answers the question that invites: **what
is the largest class of criteria for which it still works?**

The answer has two halves, and the first is a limitation rather than a reach.

The slice theorem itself needs nothing
--------------------------------------
``v`` non-dominated implies every ``x in D`` with ``Z(x) = v`` is efficient.
The proof is a dominance argument and uses no property of ``Z`` whatsoever --
not linearity, not continuity, not a ratio form.  So the *characterisation* of
``E(P_D)`` as a union of slices holds for arbitrary criteria.  What needs
structure is only the *enumeration* of each slice.

Half one: linear fractional is exactly the one-hyperplane case
--------------------------------------------------------------
Suppose the level set of ``Z`` is a single hyperplane whose data depends
affinely on the level --- in full generality, not in a chosen parametrisation,

    ``{x : Z(x) = v} = {x : alpha(v)'x = beta(v)}`` ,
    ``alpha(v) = a0 + v a1`` ,   ``beta(v) = b0 + v b1`` .

Then for any ``x`` with ``b1 - a1'x != 0``, writing ``v = Z(x)`` and
substituting,

    ``(a0 + v a1)'x = b0 + v b1``
    ``a0'x - b0 = v (b1 - a1'x)``
    ``v = (a0'x - b0) / (b1 - a1'x)`` ,

so ``Z`` is linear fractional with ``c = a0``, ``a = -b0``, ``d = -a1`` and
``b = b1``.  The converse is the computation already in
:func:`~lfp_efficient.complete.slice_equations`.  Hence:

    **On the set where its denominator does not vanish, a criterion has
    one-hyperplane level sets with data affine in the level if and only if it
    is linear fractional.**

Two points of care.  The derivation needs ``b1 - a1'x != 0``, which is exactly
the positive-denominator hypothesis of the model restated, so the
equivalence holds on ``D`` and is stated with that qualifier rather than
without it.  And it holds for *any* affine ``alpha, beta`` and not only for the
parametrisation ``alpha = c - v d``, ``beta = v b - a`` that an earlier draft
named --- checked on 400 random affine parametrisations with no exception, so
the theorem is the general one.

That is a statement about where the method stops, and it is worth more than a
reach theorem: within this framework the fractional case is not a convenient
choice, it is the maximal one.  Nothing is gained by looking for a larger class
*of that shape*, because there is none.

Half two: but the machinery never needed one hyperplane
-------------------------------------------------------
:func:`~lfp_efficient.complete.points_satisfying` takes a conjunction of linear
equations and enumerates ``D`` subject to it.  Running it once per element of a
*finite union* costs a finite multiple and loses nothing.  So the real
requirement is weaker than half one assumed:

    **It is enough that, for each level ``v``, the set ``{x : Z(x) = v}`` is a
    finite union of affine subspaces, computable from ``v``.**

That class is strictly larger than linear fractional.  ``Z(x) =
|c'x + a| / (d'x + b)`` is in it and is not linear fractional: its level set is
the pair of hyperplanes ``c'x + a = +-v (d'x + b)``.  So are integer powers of
a ratio, and sign-split forms generally.

Half three: and not even affine pieces
--------------------------------------
The same argument runs once more.  ``points_satisfying`` takes a conjunction of
equations *and* ``a'x <= b`` inequalities, and an inequality costs it nothing
extra -- it goes through the same substitution as the model's own rows.  So the
pieces of the union need only be **polyhedra**:

    **It is enough that, for each level ``v``, the set ``{x : Z(x) = v}`` is a
    finite union of polyhedra, computable from ``v``.**

That reaches a criterion type the affine version does not.  For a maximum of
ratios, ``max_i Z_i(x) = v`` holds exactly when some ``Z_j(x) = v`` and every
other ``Z_i(x) <= v``, so each branch is one equation together with ``m - 1``
inequalities -- a polyhedron, not an affine subspace.  Minimax criteria are the
same construction with the inequalities reversed, and
:func:`max_of_ratios_branches` builds either.

The three widenings are the same observation applied three times: what the
slice enumeration needs is not a shape but *finitely many linear systems*.
What it cannot absorb is curvature, which is what
Half four below is about.

Half four: what the characterisation predicts about a published class
---------------------------------------------------------------------
The limit above is not only about this package.  It is sharp enough to say
which *published* problem classes the slice machinery cannot reach, and one of
them sits directly next door.

Cherfaoui and Moula\"i (Annals of Operations Research **296** (2021), 735-753)
generate the efficient set of multiobjective integer **linear plus linear
fractional** programs, whose criteria are

    ``Z(x) = c'x + (p'x + alpha) / (q'x + beta)`` ,

a *sum* rather than a ratio.  Clearing the denominator at level ``v`` gives

    ``(c'x)(q'x + beta) + p'x + alpha - v (q'x + beta) = 0`` ,

whose quadratic part is ``(c'x)(q'x)``, with matrix ``(c q^T + q c^T)/2``.
That matrix is non-zero whenever ``c`` and ``q`` both are, so **the level set
is a conic and not a hyperplane** -- checked here rather than asserted, both by
forming the matrix and by exhibiting three non-collinear rational points on one
level of a concrete instance.

A conic is not a finite union of affine subspaces, so the class falls outside
the condition above.  Two consequences, and the second is the useful one:

1. The method of this package **does not reach** linear-plus-linear-fractional
   criteria, and the characterisation says so in advance rather than leaving it
   to be discovered.
2. That paper therefore **cannot** be generating its efficient set by the slice
   argument used here, since its criteria have no linear slices to enumerate.
   The two works address disjoint criterion classes.

The premise is the problem class as that title states it; the inference from
there is the algebra above.  Confirming it still needs the paper itself, which
could not be reached from the environment this was written in.

The other half of the method, and why its reach is smaller
----------------------------------------------------------
Everything above is about *completing* a front into ``E(P_D)``.  The front
enumeration of :mod:`lfp_efficient.front` is a separate question with a
separate answer, and the two do not coincide.

Its box search splits at a centre ``a`` into ``Z_k(x) > Z_k(a)`` and
``Z_k(x) <= Z_k(a)``.  For each child to be one ordinary linear program, both
sides must be half-spaces.  So:

    **The front enumeration requires that the sublevel set
    ``{x : Z(x) <= t}`` be a half-space at every *attained* level
    ``t`` in ``Z(D)``**, and for linear fractional criteria the integer-valued
    ``e_k`` supplies them, so there the condition is met.

Two qualifications belong to the claim rather than to its footnotes.

The *levels*: the split only ever produces levels of the form ``Z(a)`` for a
centre ``a`` in ``D``, so attained levels are all the argument delivers.  A
statement about every real ``t`` would be stronger than its proof, and on a
finite ``D`` the attained levels are all the enumeration ever sees.

The *direction*: the first half is a *necessary* condition on the split
mechanism and is what rules criteria out; the second is the sufficiency, and it
is established only for the fractional class, by construction.  An earlier
draft wrote "reaches exactly", which asserts an equivalence --- a criterion
could satisfy the condition and still not be reachable, because the method must
also *build* those half-spaces from the data, and nothing above says a general
such criterion admits that.  The claim is therefore stated as the necessary
condition it is.

That condition is strictly tighter than the slice one, and the gap is not a
technicality --- it is the reason this package is a *MOILFP* method:

* the class sits inside the polyhedral slice class.  The tempting one-line
  argument --- a half-space sublevel set has a hyperplane boundary, which is
  the level set --- is *false*: a topological boundary need not be a level set,
  and on a finite ``D`` it has no content at all.  What carries it is the
  difference of consecutive sublevel sets: with ``t_1 < ... < t_N`` the
  attained levels and ``H_i = {x : Z(x) <= t_i}`` a half-space by hypothesis,
  ``{x : Z(x) = t_i} == H_i \ H_{i-1}`` (with ``H_0`` empty), a half-space
  intersected with the complement of a half-space, hence a polyhedron;
* ``|c'x + a| / (d'x + b)`` fails it: its sublevel set is a *slab*, the
  intersection of two half-spaces, whose complement is not convex;
* ``max`` of ratios fails it too: the sublevel set is an intersection of ``m``
  half-spaces, a polyhedron, and the complement is a *union* --- which is
  exactly what a single split cannot express.

Checked on a 49-point grid rather than argued: a ratio and a linear form have
half-space sublevel sets at every level, ``|ratio|`` and ``max``-of-ratios do
not.

So the asymmetry is now characterised on both sides.  **The slice reaches
finite unions of polyhedra; the front *requires* half-space sublevel sets at
the attained levels, which is essentially the fractional case.**  For a criterion between the two --- and
``max`` of ratios is one --- the front must be obtained some other way, and the
completion is then exact.  Claiming the front generalises too would be the kind
of unchecked statement the audit of this package exists to prevent.
"""

from fractions import Fraction
from itertools import product
from typing import Callable, List, Sequence

from .complete import points_satisfying, variable_bounds
from .model import MOILFP
from .rational import F

__all__ = ["level_set_branches_fractional", "generalized_slice_points",
           "abs_ratio_branches", "max_of_ratios_branches"]


def level_set_branches_fractional(problem: MOILFP, k: int, v: Fraction):
    """The single branch of a linear fractional criterion: one equation."""
    z = problem.criteria[k]
    coeffs = [F(z.U[j]) - v * F(z.V[j]) for j in range(problem.n)]
    return [(coeffs, v * F(z.beta) - F(z.alpha))]


def abs_ratio_branches(c: Sequence, a, d: Sequence, b, n: int):
    """``|c'x + a| / (d'x + b)`` -- two branches, and not linear fractional.

    Returns a callable of the level, so it plugs into
    :func:`generalized_slice_points` like any other criterion.  The level set
    is ``c'x + a = +-v (d'x + b)``: two hyperplanes, verified over every level
    of a 36-point grid before this was written rather than after.
    """
    def branches(v: Fraction):
        out = []
        for sign in (1, -1):
            coeffs = [F(c[j]) - sign * v * F(d[j]) for j in range(n)]
            out.append((coeffs, sign * v * F(b) - F(a)))
        return out
    return branches


def _as_branch(piece):
    """Accept a branch as ``(coeffs, rhs)`` or as ``(equations, inequalities)``."""
    if (len(piece) == 2 and piece[0] and isinstance(piece[0][0], (list, tuple))
            and not isinstance(piece[0], tuple)):
        return list(piece[0]), list(piece[1])
    eqs, ines = piece
    if isinstance(eqs, (list, tuple)) and eqs and isinstance(eqs[0], (list, tuple)):
        return list(eqs), list(ines)
    return [(eqs, ines)], []


def generalized_slice_points(problem: MOILFP, bounds: Sequence[int],
                             v: Sequence[Fraction],
                             branch_fns: Sequence[Callable]) -> List[tuple]:
    """``{x in D : Z(x) = v}`` where each criterion may contribute a union.

    ``branch_fns[k](v[k])`` returns the affine pieces of criterion ``k``'s
    level set, each as ``(coeffs, rhs)``.  The slice is the union over the
    Cartesian product of those pieces, so a criterion with one branch -- every
    linear fractional one -- costs exactly what it costs today, and the product
    only grows for criteria that genuinely need it.

    Points are deduplicated: two branches may intersect, and a point on the
    intersection belongs to the slice once.
    """
    per_criterion = [[_as_branch(b) for b in branch_fns[k](v[k])]
                     for k in range(len(branch_fns))]
    seen, out = set(), []
    for combo in product(*per_criterion):
        eqs, ines = [], []
        for e, i in combo:
            eqs.extend(e)
            ines.extend(i)
        for pt in points_satisfying(problem, bounds, eqs, ines):
            if pt not in seen:
                seen.add(pt)
                out.append(pt)
    return out


def max_of_ratios_branches(ratios, n: int):
    """``max_i (c_i'x + a_i)/(d_i'x + b_i)`` -- a union of **polyhedra**.

    ``max_i Z_i(x) = v`` holds exactly when some ``Z_j(x) = v`` and every other
    ``Z_i(x) <= v``.  Each branch is therefore one equation together with
    ``len(ratios) - 1`` inequalities, which is a polyhedron rather than an
    affine subspace --- the second widening of the class, past the
    union-of-hyperplanes case that ``|c'x+a|/(d'x+b)`` needed.

    *ratios* is a sequence of ``(c, a, d, b)`` with ``d'x + b > 0`` on ``D``,
    so clearing a denominator preserves the direction of ``<=``.  A minimax
    criterion is the same construction with the inequalities reversed.
    """
    def branches(v: Fraction):
        out = []
        for j, (cj, aj, dj, bj) in enumerate(ratios):
            eq = ([F(cj[t]) - v * F(dj[t]) for t in range(n)],
                  v * F(bj) - F(aj))
            ines = []
            for i, (ci, ai, di, bi) in enumerate(ratios):
                if i == j:
                    continue
                # (c_i'x + a_i)/(d_i'x + b_i) <= v   <=>
                # (c_i - v d_i)'x <= v b_i - a_i     (denominator > 0)
                ines.append(([F(ci[t]) - v * F(di[t]) for t in range(n)],
                             v * F(bi) - F(ai)))
            out.append(([eq], ines))
        return out
    return branches
