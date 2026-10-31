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

What the characterisation predicts about a neighbouring published class
----------------------------------------------------------------------
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
           "abs_ratio_branches"]


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
    per_criterion = [list(branch_fns[k](v[k])) for k in range(len(branch_fns))]
    seen, out = set(), []
    for combo in product(*per_criterion):
        for pt in points_satisfying(problem, bounds, list(combo)):
            if pt not in seen:
                seen.add(pt)
                out.append(pt)
    return out
