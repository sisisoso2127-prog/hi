"""Exact rational arithmetic helpers.

The whole package works with :class:`fractions.Fraction` instead of floats.
Reason: the algorithm of Younsi-Abbaci relies on *exact* tests

  * "is the optimal value of the efficiency test equal to 0 ?"           (Theorem 1)
  * "is the reduced gradient component gamma_j equal to 0 ?"             (Theorem 3)
  * "is this component of the simplex solution an integer ?"             (branching)

A floating point implementation needs tolerances for each of those tests and
silently produces wrong answers on degenerate instances.  Since the problems
solved here are small (academic MOILP instances), exact arithmetic is cheap.
"""

from fractions import Fraction
from typing import Iterable, List, Sequence, Union

Number = Union[int, float, Fraction, str]

ZERO = Fraction(0)
ONE = Fraction(1)


def F(value: Number) -> Fraction:
    """Convert *value* to an exact :class:`Fraction`.

    ``int``/``Fraction``/``str`` convert exactly.  A ``float`` is first routed
    through ``str`` so that ``0.1`` becomes ``1/10`` and not the binary
    approximation ``3602879701896397/36028797018963968``.
    """
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(str(value))
    return Fraction(value)


def vec(values: Iterable[Number]) -> List[Fraction]:
    """Convert an iterable of numbers to a list of fractions."""
    return [F(v) for v in values]


def mat(rows: Iterable[Iterable[Number]]) -> List[List[Fraction]]:
    """Convert a 2-D iterable of numbers to a matrix of fractions."""
    return [vec(row) for row in rows]


def dot(u: Sequence[Fraction], v: Sequence[Fraction]) -> Fraction:
    """Inner product of two equally sized rational vectors."""
    return sum((a * b for a, b in zip(u, v)), ZERO)


def is_integral(value: Fraction) -> bool:
    """True when *value* is an integer (exactly, no tolerance involved)."""
    return value.denominator == 1


def fmt(value: Fraction) -> str:
    """Compact human readable form: ``3`` for 3, ``5/17`` for 5/17."""
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"
