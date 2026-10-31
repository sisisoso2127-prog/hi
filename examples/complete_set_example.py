#!/usr/bin/env python3
"""The complete efficient set, on an instance where the front is not enough.

The criteria ignore ``x3``, so two non-dominated vectors carry six efficient
points between them.  ``enumerate_front`` proves the *vector* list complete and
returns two points; ``complete_efficient_set`` returns all six, and proves it.

Run from the repository root::

    python examples/complete_set_example.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lfp_efficient import (FractionalObjective, LE, MOILP, Model,
                           complete_efficient_set, enumerate_efficient_set,
                           enumerate_front)


def main() -> int:
    model = (Model(3).add([1, 0, 0], LE, 2).add([0, 1, 0], LE, 2)
             .add([0, 0, 1], LE, 2).add([1, 1, 0], LE, 3))
    problem = MOILP(model, [[1, 0, 0], [0, 1, 0]])       # x3 absent from Z
    phi = FractionalObjective([1, 1, 3], [1, 1, 1], 0, 1)   # but present in Phi

    front = enumerate_front(problem, phi)
    print("the front:")
    print(front.report())

    result = complete_efficient_set(problem, phi)
    print("\nthe complete efficient set:")
    print(result.report())

    print("\nevery efficient point, by criterion vector:")
    for v, members in zip(result.vectors, result.slices):
        listed = "  ".join("(" + ", ".join(str(c) for c in x) + ")"
                           for x in members)
        print(f"  Z = ({', '.join(str(c) for c in v)}):  {listed}")

    truth = enumerate_efficient_set(problem, [2, 2, 2]).efficient
    assert {tuple(x) for x in result.points} == {tuple(x) for x in truth}
    print(f"\nchecked against exhaustive enumeration: {len(truth)} points, "
          "identical")
    print(f"the front alone returned {len(front.points)} of them, and still "
          f"found Phi* = {result.best_value} -- because Q keeps the best point "
          "of each slice")
    return 0


if __name__ == "__main__":
    sys.exit(main())
