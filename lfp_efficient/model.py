"""Problem data structures: general-form models, MOILP, fractional objective."""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Sequence

from .rational import F, ZERO, Number, dot, fmt, mat, vec

LE, GE, EQ = "<=", ">=", "="


@dataclass
class Constraint:
    """A single linear constraint ``coeffs . x  (sense)  rhs``."""

    coeffs: List[Fraction]
    sense: str
    rhs: Fraction

    def holds(self, x: Sequence[Fraction]) -> bool:
        lhs = dot(self.coeffs, x)
        if self.sense == LE:
            return lhs <= self.rhs
        if self.sense == GE:
            return lhs >= self.rhs
        return lhs == self.rhs

    def __str__(self) -> str:
        terms = " + ".join(f"{fmt(c)}*x{j + 1}" for j, c in enumerate(self.coeffs) if c != 0)
        return f"{terms or '0'} {self.sense} {fmt(self.rhs)}"


@dataclass
class Model:
    """``{ x >= 0 : constraints hold, x_j integer for j in integrality }``.

    Variables are always non-negative.  ``integrality[j] = True`` requests an
    integer variable; continuous variables are used for the slack ``Psi`` of the
    efficiency test, and binary variables (integer + an explicit ``<= 1`` row)
    for the Sylva-Crema cuts.
    """

    n: int
    constraints: List[Constraint] = field(default_factory=list)
    integrality: List[bool] = field(default_factory=list)

    def __post_init__(self):
        if not self.integrality:
            self.integrality = [True] * self.n

    # ---- construction helpers -------------------------------------------
    def copy(self) -> "Model":
        return Model(self.n, [Constraint(c.coeffs[:], c.sense, c.rhs) for c in self.constraints],
                     self.integrality[:])

    def add(self, coeffs: Sequence[Number], sense: str, rhs: Number) -> "Model":
        row = vec(coeffs)
        row += [ZERO] * (self.n - len(row))
        self.constraints.append(Constraint(row, sense, F(rhs)))
        return self

    def add_variables(self, count: int, integer: bool = True) -> List[int]:
        """Append ``count`` fresh variables, return their indices."""
        start = self.n
        self.n += count
        self.integrality += [integer] * count
        for c in self.constraints:
            c.coeffs += [ZERO] * count
        return list(range(start, self.n))

    def is_feasible(self, x: Sequence[Fraction]) -> bool:
        if any(v < 0 for v in x):
            return False
        if any(self.integrality[j] and x[j].denominator != 1 for j in range(self.n)):
            return False
        return all(c.holds(x) for c in self.constraints)

    # ---- conversion to the simplex standard form ------------------------
    def to_standard_form(self):
        """Return ``(A, b, n_structural)`` with slack/surplus columns appended.

        The first ``n`` columns of ``A`` are the model variables, so a standard
        form solution can be projected back with ``x[:n]``.
        """
        rows, rhs = [], []
        extra = sum(1 for c in self.constraints if c.sense in (LE, GE))
        k = 0
        for c in self.constraints:
            row = c.coeffs[:] + [ZERO] * extra
            if c.sense == LE:
                row[self.n + k] = F(1)
                k += 1
            elif c.sense == GE:
                row[self.n + k] = F(-1)
                k += 1
            rows.append(row)
            rhs.append(c.rhs)
        return rows, rhs, self.n


@dataclass
class FractionalObjective:
    """``Phi(x) = (U'x + alpha) / (V'x + beta)`` -- the paper's main criterion."""

    U: List[Fraction]
    V: List[Fraction]
    alpha: Fraction = ZERO
    beta: Fraction = ZERO

    def __init__(self, U: Sequence[Number], V: Sequence[Number],
                 alpha: Number = 0, beta: Number = 0):
        self.U, self.V = vec(U), vec(V)
        self.alpha, self.beta = F(alpha), F(beta)

    def __call__(self, x: Sequence[Fraction]) -> Fraction:
        den = dot(self.V, x[:len(self.V)]) + self.beta
        if den == 0:
            raise ZeroDivisionError(f"V'x + beta = 0 at x = {[fmt(v) for v in x]}")
        return (dot(self.U, x[:len(self.U)]) + self.alpha) / den

    def lift(self, n: int) -> "FractionalObjective":
        """Same objective seen in a larger space (extra variables get coefficient 0)."""
        pad = [ZERO] * (n - len(self.U))
        return FractionalObjective(self.U + pad, self.V + pad, self.alpha, self.beta)

    def __str__(self) -> str:
        def side(coeffs, const):
            terms = [f"{fmt(c)}*x{j + 1}" for j, c in enumerate(coeffs) if c != 0]
            if const != 0:
                terms.append(fmt(const))
            return " + ".join(terms) if terms else "0"
        return f"({side(self.U, self.alpha)}) / ({side(self.V, self.beta)})"


@dataclass
class MOILFP:
    """Multi-objective integer **linear fractional** program::

        "max"  Z_k(x) = (c_k'x + a_k) / (d_k'x + b_k) ,  k = 1..p
        s.t.   x in D = { x in Z^n_+ : A x <~ b }

    Each criterion is a ratio.  A *linear* criterion ``C_k x`` is the
    degenerate case ``d_k = 0``, ``b_k = 1`` -- which is what :class:`MOILP`
    builds -- so the linear program of the paper is a special case of this one
    and every formula below reduces to the linear one on it.

    All criteria are maximised; a criterion to be minimised is passed with the
    sign of its numerator flipped.

    Assumption, checked on construction of the algorithm's sub-problems: each
    ``D_k(x) = d_k'x + b_k`` stays **strictly positive** on ``D``.  Without it
    the ratio is not even continuous on the feasible set and the sign argument
    that the whole method rests on fails.

    The data of every criterion is scaled to integers at construction (a
    criterion's numerator and denominator are multiplied by the same factor, so
    the ratio is untouched).  That is what makes ``e_k`` below integer-valued,
    and with it the exact ``>= 1`` threshold that replaces an estimated minimal
    step.
    """

    model: Model
    criteria: List[FractionalObjective]

    def __init__(self, model: Model, criteria: Sequence[FractionalObjective]):
        self.model = model
        self.criteria = [_scaled_to_integers(z.lift(model.n)) for z in criteria]

    @property
    def p(self) -> int:
        return len(self.criteria)

    @property
    def n(self) -> int:
        return self.model.n

    def Z(self, x: Sequence[Fraction]) -> List[Fraction]:
        """The criterion vector ``(Z_1(x), ..., Z_p(x))``."""
        return [z(x) for z in self.criteria]

    #: kept so that code written for the linear case keeps reading naturally
    C = Z

    def numerator(self, k: int, x: Sequence[Fraction]) -> Fraction:
        z = self.criteria[k]
        return dot(z.U, x[:len(z.U)]) + z.alpha

    def denominator(self, k: int, x: Sequence[Fraction]) -> Fraction:
        z = self.criteria[k]
        return dot(z.V, x[:len(z.V)]) + z.beta

    def e_row(self, k: int, x_bar: Sequence[Fraction]):
        """The linear form ``e_k( . ; x_bar)`` of Theorem 4, as ``(coeffs, const)``.

            e_k(x) = (D_k(x_bar) c_k - N_k(x_bar) d_k)'x
                     + (D_k(x_bar) a_k - N_k(x_bar) b_k)
                   = D_k(x_bar) * D_k(x) * ( Z_k(x) - Z_k(x_bar) )

        Both denominators being positive, ``e_k`` carries the **sign** of
        ``Z_k(x) - Z_k(x_bar)`` while being *linear in x* -- that is the whole
        trick, and it is what lets an ordinary integer linear solver answer
        questions about ratios.  With integer data ``e_k`` is integer-valued on
        integer points, so

            Z_k(x) >  Z_k(x_bar)  <=>  e_k(x) >= 1
            Z_k(x) == Z_k(x_bar)  <=>  e_k(x) == 0

        exactly, with no minimal step to estimate.  On a linear criterion
        (``d_k = 0``, ``b_k = 1``) it collapses to ``C_k x - C_k x_bar``.
        """
        z = self.criteria[k]
        n_bar = self.numerator(k, x_bar)
        d_bar = self.denominator(k, x_bar)
        coeffs = [d_bar * c - n_bar * d for c, d in zip(z.U, z.V)]
        const = d_bar * z.alpha - n_bar * z.beta
        return coeffs, const

    def dominates(self, x: Sequence[Fraction], y: Sequence[Fraction]) -> bool:
        """``Z(x) >= Z(y)`` componentwise with at least one strict inequality."""
        zx, zy = self.Z(x), self.Z(y)
        return all(a >= b for a, b in zip(zx, zy)) and any(a > b for a, b in zip(zx, zy))


def _scaled_to_integers(z: FractionalObjective) -> FractionalObjective:
    """Multiply a criterion's numerator *and* denominator by one integer factor.

    The ratio is unchanged; the point is to make ``e_k`` integer-valued.
    """
    values = list(z.U) + list(z.V) + [z.alpha, z.beta]
    scale = 1
    for v in values:
        d = v.denominator
        g = _gcd(scale, d)
        scale = scale * d // g
    if scale == 1:
        return z
    s = F(scale)
    return FractionalObjective([s * u for u in z.U], [s * v for v in z.V],
                               s * z.alpha, s * z.beta)


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def MOILP(model: Model, criteria: Sequence[Sequence[Number]]) -> MOILFP:
    """The paper's program: ``p`` **linear** criteria ``Z_i = C_i x``.

    A thin constructor over :class:`MOILFP`: each row ``C_i`` becomes the ratio
    ``(C_i x + 0) / (0'x + 1)``.  Every fractional formula then reduces to the
    linear one, so the linear case is not a separate code path -- it is the
    same code with unit denominators.
    """
    n = model.n
    return MOILFP(model, [FractionalObjective(list(row), [0] * n, 0, 1)
                          for row in criteria])
