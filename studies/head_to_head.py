"""The comparison this project was missing: two routes to the *same* set.

Every other measurement here compares criterion space against decision space on
the **optimisation** task, against the Sylva--Crema cut.  None of them compared
the complete-set route of :mod:`lfp_efficient.complete` against a decision-space
method that also returns all of ``E(P_D)`` -- and such a method predates this
work, so "we produce the whole efficient set" was never the claim to defend.

This script runs both routes on the same instances and reports:

* whether they return the **same set** (they must; a difference is a bug, not a
  finding, and the run says so loudly);
* the **efficiency tests** each pays.  This is the mechanism difference and it
  is deterministic: the slice route pays none, by the slice theorem, while the
  cut route tests every integer point it meets;
* seconds, with the usual warning that this machine's noise floor is about
  +/-30% per instance, so the counts are the finding and the time is context.

Censoring is reported, never hidden: a side stopped by its budget makes the
whole instance censored, and a censored instance is excluded from the medians
rather than counted at its budget.

Running it::

    python studies/head_to_head.py              # the default sweep
    python studies/head_to_head.py --budget 30  # tighter budget per side
    python studies/head_to_head.py --quick      # a pilot
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lfp_efficient import FractionalObjective, LE, MOILFP, Model      # noqa: E402
from lfp_efficient.complete import complete_efficient_set             # noqa: E402
from lfp_efficient.decision_space import enumerate_by_cuts            # noqa: E402

RESULTS = Path(__file__).with_name("head-to-head.json")


def say(*a):
    print(*a, flush=True)


def instance(seed: int, n: int, p: int, m: int | None = None) -> MOILFP:
    """One instance of the same family the rest of the campaign uses."""
    rnd = random.Random(seed)
    m = m or max(2, n // 2)
    model = Model(n)
    for _ in range(m):
        model.add([rnd.randint(1, 9) for _ in range(n)], LE,
                  rnd.randint(4 * n, 8 * n))
    criteria = []
    for _ in range(p):
        U = [rnd.randint(-5, 9) for _ in range(n)]
        V = [rnd.randint(0, 4) for _ in range(n)]
        criteria.append(FractionalObjective(U, V, rnd.randint(0, 5),
                                            rnd.randint(1, 6)))
    return MOILFP(model, criteria)


def median(values):
    values = sorted(values)
    return values[len(values) // 2] if values else None


def run(configs, seeds, budget):
    say(f"two routes to all of E(P_D); budget {budget}s per side\n")
    say(f"{'n':>3} {'p':>2} {'seed':>4} | {'|E|':>4} | "
        f"{'slice s':>8} {'boxes':>6} {'tests':>5} | "
        f"{'cut s':>8} {'nodes':>6} {'tests':>5} | {'ratio':>6} agree")
    say("-" * 92)

    records = []
    for n, p in configs:
        for seed in seeds:
            problem = instance(seed, n, p)

            started = perf_counter()
            slices = complete_efficient_set(problem, time_budget=budget)
            slice_seconds = perf_counter() - started

            cuts = enumerate_by_cuts(problem, time_budget=budget)

            censored = (not slices.complete) or (not cuts.complete)
            left = sorted(tuple(x) for x in slices.points)
            right = sorted(cuts.efficient)
            if censored:
                verdict = "censored"
            elif left == right:
                verdict = "same"
            else:
                verdict = "*** DIFFER ***"

            ratio = cuts.seconds / slice_seconds if slice_seconds else float("inf")
            say(f"{n:>3} {p:>2} {seed:>4} | {len(left):>4} | "
                f"{slice_seconds:>8.2f} {slices.boxes:>6} {0:>5} | "
                f"{cuts.seconds:>8.2f} {cuts.nodes:>6} {cuts.tests:>5} | "
                f"{ratio:>5.2f}x {verdict}")

            records.append({
                "n": n, "p": p, "seed": seed, "size": len(left),
                "slice_seconds": slice_seconds, "slice_boxes": slices.boxes,
                "slice_tests": 0,
                "cut_seconds": cuts.seconds, "cut_nodes": cuts.nodes,
                "cut_tests": cuts.tests, "cut_cuts": cuts.cuts,
                "censored": censored, "same": left == right,
            })

            if verdict.startswith("***"):
                say(f"    !! left={left}\n    !! right={right}")

    usable = [r for r in records if r["same"] and not r["censored"]]
    say("")
    say(f"instances run      : {len(records)}")
    say(f"censored           : {sum(1 for r in records if r['censored'])}")
    say(f"disagreements      : {sum(1 for r in records if not r['same'])}"
        "   (any at all is a bug)")
    if usable:
        say(f"usable             : {len(usable)}")
        say(f"median time ratio  : "
            f"{median([r['cut_seconds'] / r['slice_seconds'] for r in usable]):.2f}x"
            "   (cut route / slice route)")
        say(f"median |E|         : {median([r['size'] for r in usable])}")
        say(f"median efficiency tests  cut route: "
            f"{median([r['cut_tests'] for r in usable])}   slice route: 0")
        say(f"total efficiency tests   cut route: "
            f"{sum(r['cut_tests'] for r in usable)}   slice route: 0")
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=float, default=60.0,
                    help="seconds allowed to each side, per instance")
    ap.add_argument("--quick", action="store_true", help="a small pilot")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    if args.quick:
        configs, seeds = [(5, 2), (6, 3)], range(2)
    else:
        configs = [(5, 2), (5, 3), (6, 2), (6, 3), (7, 2), (7, 3),
                   (8, 2), (8, 3), (9, 2), (10, 2)]
        seeds = range(args.seeds)

    records = run(configs, seeds, args.budget)
    RESULTS.write_text(json.dumps(
        {"meta": {"budget": args.budget, "quick": args.quick},
         "records": records}, indent=1))
    say(f"\nraw records in {RESULTS}")


if __name__ == "__main__":
    main()
