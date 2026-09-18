"""Exhaustive enumeration -- the reference implementation used for testing.

Nothing here is part of the algorithm: this module simply enumerates every
integer point of a box, keeps the feasible ones, filters the efficient ones by
the definition (Definition 1 of the paper) and maximises ``Phi`` over them.
It is exponential and only usable on toy instances, which is exactly the point:
it gives an independent ground truth for the algorithm.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from typing import List, Optional, Sequence, Tuple

from .model import MOILP
from .rational import F


@dataclass
class Enumeration:
    feasible: List[List[Fraction]] = field(default_factory=list)
    efficient: List[List[Fraction]] = field(default_factory=list)

    def best(self, phi) -> Tuple[Optional[List[Fraction]], Optional[Fraction]]:
        """``argmax`` and ``max`` of ``Phi`` over the efficient set."""
        best_x, best_v = None, None
        for x in self.efficient:
            try:
                v = phi(x)
            except ZeroDivisionError:      # Phi undefined there, skip the point
                continue
            if best_v is None or v > best_v:
                best_x, best_v = x, v
        return best_x, best_v


def enumerate_efficient_set(problem: MOILP, bounds: Sequence[int]) -> Enumeration:
    """Enumerate ``D`` inside ``0 <= x_j <= bounds[j]`` and extract ``E(P_D)``.

    A point ``x0`` is efficient when no feasible ``x1`` satisfies
    ``C x1 >= C x0`` with at least one strict inequality (Definition 1).
    """
    result = Enumeration()
    for combo in product(*(range(b + 1) for b in bounds)):
        x = [F(v) for v in combo]
        if problem.model.is_feasible(x):
            result.feasible.append(x)

    for x0 in result.feasible:
        if not any(problem.dominates(x1, x0) for x1 in result.feasible):
            result.efficient.append(x0)
    return result
