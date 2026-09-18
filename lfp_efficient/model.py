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
class MOILP:
    """The multi-objective integer linear program ``(P_D)`` of the paper::

        "max"  Z_i = C_i x ,  i = 1..p
        s.t.   x in D = { x in Z^n_+ : A x <~ b }

    ``criteria`` is the matrix ``C`` (p rows).  All criteria are maximised; a
    criterion to be minimised is simply passed with its sign flipped.
    """

    model: Model
    criteria: List[List[Fraction]]

    def __init__(self, model: Model, criteria: Sequence[Sequence[Number]]):
        self.model = model
        self.criteria = mat(criteria)

    @property
    def p(self) -> int:
        return len(self.criteria)

    @property
    def n(self) -> int:
        return self.model.n

    def C(self, x: Sequence[Fraction]) -> List[Fraction]:
        """The criterion vector ``Cx`` of a point."""
        return [dot(row, x[:len(row)]) for row in self.criteria]

    def dominates(self, x: Sequence[Fraction], y: Sequence[Fraction]) -> bool:
        """``Cx >= Cy`` componentwise with at least one strict inequality."""
        cx, cy = self.C(x), self.C(y)
        return all(a >= b for a, b in zip(cx, cy)) and any(a > b for a, b in zip(cx, cy))
