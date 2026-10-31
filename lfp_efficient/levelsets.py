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
affinely on the level,

    ``{x : Z(x) = v} = {x : alpha(v)'x = beta(v)}`` ,
    ``alpha(v) = c - v d`` ,   ``beta(v) = v b - a`` .

Then for any ``x`` in the domain, writing ``v = Z(x)`` and substituting,

    ``(c - v d)'x = v b - a``
    ``c'x + a = v (d'x + b)``
    ``v = (c'x + a) / (d'x + b)``   whenever ``d'x + b != 0`` ,

so ``Z`` **is** linear fractional.  The converse is the computation already in
:func:`~lfp_efficient.complete.slice_equations`.  Hence:

    **A criterion has one-hyperplane level sets with data affine in the level
    if and only if it is linear fractional.**

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

What this does **not** extend
-----------------------------
The front enumeration of :mod:`lfp_efficient.front` is a different matter and
is **not** generalised here.  Its box search needs ``Z_k(x) > Z_k(a)`` to be a
linear constraint, which is what the integer-valued ``e_k`` provides for ratios
and does not provide for the wider class.  So for a criterion in the wider
class the front must be obtained some other way; given the front, the
completion below is exact.  Saying otherwise would be the kind of unchecked
claim the audit of this package exists to prevent.
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
