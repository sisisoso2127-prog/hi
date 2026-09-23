#!/usr/bin/env python3
"""A measured campaign: every headline claim, on a sample big enough to quote.

Why this exists
---------------
Most figures in the write-up were first taken on four to eight instances.  The
*direction* of each was argued from a mechanism, which is what made them worth
reporting at all -- but a total over four runs is not an estimate, and one
instance supplying $110.95$ of $112.77$ seconds is not a distribution.  This
script re-measures the same claims over thirty instances per configuration and
reports **medians with an interquartile range**, not totals.

What it measures, and in what unit
----------------------------------
The machine's noise floor is roughly +/-30% per instance, so wall-clock cannot
resolve a small effect and no number of repeats fixes that.  Every experiment
here therefore reports a **deterministic work count** -- sub-problems solved
and branch & bound nodes, identical on every run and every machine -- next to
the time.  Where the two disagree, the count is the finding and the seconds are
context.

Censoring is reported, never hidden.  A run stopped by its time budget is
counted as censored and the median is quoted only while fewer than half the
runs were; past that the row says so instead of quoting a number.

Running it
----------
    python studies/campaign.py --quick        # a pilot: 5 instances, small p
    python studies/campaign.py                # the full campaign
    python studies/campaign.py --only margin

Nothing else should run on the machine at the same time; a contaminated
measurement is discarded rather than quoted.  Raw per-instance records are
written to ``studies/campaign-results.json`` so the tables can be rebuilt, and
re-checked, without paying for the run again.
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lfp_efficient.complete as complete_mod
import lfp_efficient.criterion_space as cs_mod
import lfp_efficient.algorithm as alg_mod
import lfp_efficient.efficiency as eff_mod
import lfp_efficient.front as front_mod
import lfp_efficient.tchebychev as tch_mod
from lfp_efficient import (FractionalObjective, LE, MOILP, Model,
                           enumerate_nondominated,
                           metaheuristic_incumbent,
                           optimize_in_criterion_space,
                           optimize_over_efficient_set,
                           tchebychev_incumbent)
from lfp_efficient.complete import complete_efficient_set, variable_bounds
from lfp_efficient.front import enumerate_front

OUT = Path(__file__).resolve().parent / "campaign-results.json"

# --------------------------------------------------------------------------
# the instance family
# --------------------------------------------------------------------------
def build(seed, n, p, ub=3, tightness=0.30, rows=4):
    """One reproducible instance of the family used throughout the campaign.

    ``tightness`` is the share of maximal resource consumption each constraint
    allows, so it sets ``|D|`` directly.  The criteria reward large ``x`` while
    ``Phi`` rewards small ``x``, so the maximiser of ``Phi`` over ``D`` is
    dominated and the method has to work its way to the answer -- an instance
    where it were not would measure nothing.
    """
    rng = random.Random(seed * 7919 + n * 101 + p)
    model = Model(n)
    for j in range(n):
        unit = [0] * n
        unit[j] = 1
        model.add(unit, LE, ub)
    for _ in range(rows):
        row = [rng.randint(1, 4) for _ in range(n)]
        model.add(row, LE, max(1, int(sum(row) * ub * tightness)))
    criteria = [[rng.randint(1, 5) for _ in range(n)] for _ in range(p)]
    phi = FractionalObjective([-rng.randint(1, 5) for _ in range(n)],
                              [rng.randint(1, 3) for _ in range(n)],
                              rng.randint(20, 45), rng.randint(4, 10))
    return MOILP(model, criteria), phi, [ub] * n


# --------------------------------------------------------------------------
# instrumentation: count the deterministic work, not only the seconds
# --------------------------------------------------------------------------
_MODULES = (alg_mod, cs_mod, eff_mod, front_mod, tch_mod, complete_mod)
_NAMES = ("solve_milp", "solve_linear_milp", "solve_fractional_milp")


class Work:
    """Counts sub-programs solved and branch & bound nodes explored.

    The library modules import the solvers by name, so each module's own
    attribute is patched; patching :mod:`lfp_efficient.milp` alone would count
    nothing.
    """

    def __init__(self):
        self.programs = 0
        self.nodes = 0
        self._saved = []

    def __enter__(self):
        for module in _MODULES:
            for name in _NAMES:
                original = getattr(module, name, None)
                if original is None:
                    continue
                self._saved.append((module, name, original))
                setattr(module, name, self._wrap(original))
        return self

    def _wrap(self, original):
        def counted(*args, **kwargs):
            result = original(*args, **kwargs)
            self.programs += 1
            self.nodes += getattr(result, "nodes", 0) or 0
            return result
        return counted

    def __exit__(self, *exc):
        for module, name, original in self._saved:
            setattr(module, name, original)
        self._saved.clear()
        return False


def timed(fn, *args, **kwargs):
    """``(value, seconds, programs, nodes)`` for one measured call."""
    with Work() as work:
        start = time.perf_counter()
        value = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
    return value, elapsed, work.programs, work.nodes


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def summary(values):
    """Median and interquartile range; ``None`` when there is nothing to say."""
    kept = [v for v in values if v is not None]
    if not kept:
        return None
    kept.sort()
    median = statistics.median(kept)
    if len(kept) >= 4:
        half = len(kept) // 2
        low = statistics.median(kept[:half])
        high = statistics.median(kept[-half:])
    else:
        low, high = kept[0], kept[-1]
    return {"n": len(kept), "median": median, "q1": low, "q3": high,
            "min": kept[0], "max": kept[-1]}


def line(label, stats, censored, total, unit="s", width=22):
    if stats is None or stats["n"] * 2 <= total:
        return (f"{label:<{width}} | "
                f"more than half censored ({censored}/{total}) -- not quoted")
    def f(v):
        return f"{v:.3f}" if unit == "s" else f"{v:,.0f}"
    note = f"  [{censored}/{total} censored]" if censored else ""
    return (f"{label:<{width}} | median {f(stats['median']):>10}{unit} "
            f"[{f(stats['q1'])}, {f(stats['q3'])}]  "
            f"min {f(stats['min'])}  max {f(stats['max'])}{note}")


def ratios(pairs):
    """Median of the *per-instance* ratios -- not the ratio of the medians.

    A ratio of totals is dominated by the single heaviest instance, which is
    exactly the flaw the campaign exists to remove.
    """
    each = [a / b for a, b in pairs if b]
    return summary(each)


# --------------------------------------------------------------------------
# experiment 1: the margin, decision space against criterion space
# --------------------------------------------------------------------------
def experiment_margin(instances, budget, ps, n=6):
    """The headline claim, per ``p``, as a distribution of per-instance ratios.

    Both methods are given the same time budget.  A run that spends it is
    censored: its time is a lower bound, so it is excluded from the medians and
    counted in the open.  Censoring is one-sided -- it only ever makes the
    decision-space method look *better* -- so a margin quoted despite it is a
    conservative one.
    """
    rows = []
    for p in ps:
        records = []
        censored = 0
        for seed in range(instances):
            problem, phi, _ = build(seed, n, p)
            dec, t_dec, prog_dec, nodes_dec = timed(
                optimize_over_efficient_set, problem, phi, time_budget=budget)
            crit, t_crit, prog_crit, nodes_crit = timed(
                optimize_in_criterion_space, problem, phi, time_budget=budget)
            done = dec.proved_optimal and crit.proved_optimal
            if not done:
                censored += 1
            if dec.proved_optimal and crit.proved_optimal:
                assert dec.value == crit.value, (
                    f"p={p} seed={seed}: {dec.value} != {crit.value}")
            records.append({
                "seed": seed, "p": p, "n": n, "proved": done,
                "t_decision": t_dec, "t_criterion": t_crit,
                "programs_decision": prog_dec, "programs_criterion": prog_crit,
                "nodes_decision": nodes_dec, "nodes_criterion": nodes_crit,
            })
            print(f"    p={p} seed={seed:2d}: "
                  f"{t_dec:7.2f}s / {t_crit:6.2f}s"
                  f"{'' if done else '   (censored)'}", flush=True)
        good = [r for r in records if r["proved"]]
        rows.append({
            "p": p, "n": n, "instances": instances, "censored": censored,
            "records": records,
            "time_ratio": ratios([(r["t_decision"], r["t_criterion"])
                                  for r in good]),
            "program_ratio": ratios([(r["programs_decision"],
                                      r["programs_criterion"]) for r in good]),
            "node_ratio": ratios([(r["nodes_decision"], r["nodes_criterion"])
                                  for r in good]),
            "t_decision": summary([r["t_decision"] for r in good]),
            "t_criterion": summary([r["t_criterion"] for r in good]),
        })
    return rows


# --------------------------------------------------------------------------
# experiment 2: what a seed is worth
# --------------------------------------------------------------------------
def experiment_seeds(instances, budget, n=6, p=4):
    """No seed against a Pareto seed against a Tchebychev seed.

    The three runs answer the same question on the same instance, so the
    comparison is paired: the per-instance ratio is the statistic, and its
    median is quoted rather than a ratio of totals.
    """
    records = []
    for seed in range(instances):
        problem, phi, _ = build(seed, n, p)
        plain, t_plain, prog_plain, _ = timed(
            optimize_in_criterion_space, problem, phi, time_budget=budget)
        if not plain.proved_optimal:
            print(f"    seed={seed:2d}: censored", flush=True)
            continue

        pareto = metaheuristic_incumbent(problem, phi)
        tcheby = tchebychev_incumbent(problem, phi)
        row = {"seed": seed, "n": n, "p": p, "t_plain": t_plain,
               "programs_plain": prog_plain, "optimum": str(plain.value)}
        for name, start in (("pareto", pareto), ("tchebychev", tcheby)):
            if start is None or start[0] is None:
                row[f"t_{name}"] = None
                continue
            point, value = start[0], start[1]
            seeded, elapsed, programs, _ = timed(
                optimize_in_criterion_space, problem, phi,
                time_budget=budget, incumbent=point, incumbent_value=value)
            if not seeded.proved_optimal:
                row[f"t_{name}"] = None
                continue
            assert seeded.value == plain.value, (
                f"seed={seed}: the {name} seed changed the answer")
            row[f"t_{name}"] = elapsed
            row[f"programs_{name}"] = programs
            # the non-negative quality gap, well defined whatever the sign
            row[f"gap_{name}"] = float(
                (plain.value - value) / abs(plain.value)) if plain.value else None
        records.append(row)
        print(f"    seed={seed:2d}: plain {t_plain:6.2f}s  "
              f"pareto {row.get('t_pareto') or float('nan'):6.2f}s  "
              f"tcheby {row.get('t_tchebychev') or float('nan'):6.2f}s",
              flush=True)

    out = {"instances": instances, "n": n, "p": p, "records": records}
    for name in ("pareto", "tchebychev"):
        paired = [(r["t_plain"], r[f"t_{name}"]) for r in records
                  if r.get(f"t_{name}")]
        out[f"speedup_{name}"] = ratios(paired)
        out[f"gap_{name}"] = summary([r.get(f"gap_{name}") for r in records])
        out[f"exact_{name}"] = sum(1 for r in records
                                   if r.get(f"gap_{name}") == 0.0)
        out[f"paired_{name}"] = len(paired)
    return out


# --------------------------------------------------------------------------
# experiment 3: enumerating the front, criterion space against decision space
# --------------------------------------------------------------------------
def _one_front(problem, budget):
    """Both enumerations of one instance's front, with their work counts."""
    crit, t_crit, prog_crit, nodes_crit = timed(
        enumerate_front, problem, None, time_budget=budget)
    start = time.perf_counter()
    with Work() as work:
        try:
            dec = enumerate_nondominated(problem)
            t_dec, ok = time.perf_counter() - start, True
        except Exception:                                  # noqa: BLE001
            dec, t_dec, ok = None, time.perf_counter() - start, False
    if not (crit.complete and ok):
        return None
    assert len(dec.vectors) == len(crit.vectors), (
        f"{len(dec.vectors)} vectors in decision space, "
        f"{len(crit.vectors)} in criterion space")
    return {"front": len(crit.vectors),
            "t_criterion": t_crit, "t_decision": t_dec,
            "programs_criterion": prog_crit, "programs_decision": work.programs,
            "nodes_criterion": nodes_crit, "nodes_decision": work.nodes}


def experiment_front(instances, budget, n=5, p=3,
                     tightnesses=(0.25, 0.35, 0.45)):
    """Both enumerations of the whole front, over a sweep that moves ``|F|``.

    The claim under test is not only that criterion space wins but that *the
    margin tracks the size of the front* -- every vector is a cut, and in
    decision space a cut costs ``p`` binaries and ``p+1`` rows.  One tightness
    cannot show that, so the sweep loosens the constraints to grow ``|F|`` and
    the relationship is estimated rather than asserted.
    """
    records, censored = [], 0
    for tightness in tightnesses:
        for seed in range(instances):
            problem, _, _ = build(seed, n, p, tightness=tightness)
            row = _one_front(problem, budget)
            if row is None:
                censored += 1
                print(f"    t={tightness} seed={seed:2d}: censored", flush=True)
                continue
            row.update(seed=seed, n=n, p=p, tightness=tightness)
            records.append(row)
            print(f"    t={tightness} seed={seed:2d}: "
                  f"|F|={row['front']:3d}  {row['t_criterion']:6.3f}s / "
                  f"{row['t_decision']:8.2f}s", flush=True)

    by_size = {}
    for r in records:
        bucket = ("1-4" if r["front"] <= 4
                  else "5-12" if r["front"] <= 12 else "13+")
        by_size.setdefault(bucket, []).append(r)
    return {"instances": instances, "n": n, "p": p, "censored": censored,
            "tightnesses": list(tightnesses), "records": records,
            "time_ratio": ratios([(r["t_decision"], r["t_criterion"])
                                  for r in records]),
            "node_ratio": ratios([(r["nodes_decision"], r["nodes_criterion"])
                                  for r in records]),
            "front": summary([r["front"] for r in records]),
            "by_front_size": {
                bucket: {"n": len(rs),
                         "front": summary([r["front"] for r in rs]),
                         "time_ratio": ratios([(r["t_decision"],
                                                r["t_criterion"])
                                               for r in rs])}
                for bucket, rs in sorted(by_size.items())}}


# --------------------------------------------------------------------------
# experiment 4: completing the front to the whole efficient set
# --------------------------------------------------------------------------
def experiment_complete(instances, budget, n=5, p=3):
    records = []
    for seed in range(instances):
        problem, phi, _ = build(seed, n, p)
        front, t_front, _, _ = timed(enumerate_front, problem, None,
                                     time_budget=budget)
        if not front.complete:
            continue
        bounds, t_bounds, _, _ = timed(variable_bounds, problem)
        whole, t_slices, _, _ = timed(complete_efficient_set, problem, phi,
                                      bounds=bounds, front=front)
        assert whole.complete
        records.append({"seed": seed, "n": n, "p": p,
                        "front": len(front.vectors),
                        "points": len(whole.points),
                        "t_front": t_front, "t_bounds": t_bounds,
                        "t_slices": t_slices,
                        "overhead": (t_bounds + t_slices) / t_front})
        print(f"    seed={seed:2d}: |F|={len(front.vectors):3d} "
              f"|E|={len(whole.points):4d}  front {t_front:6.3f}s  "
              f"+{(t_bounds + t_slices):.3f}s", flush=True)
    return {"instances": instances, "n": n, "p": p, "records": records,
            "overhead": summary([r["overhead"] for r in records]),
            "extra_points": summary([r["points"] - r["front"]
                                     for r in records]),
            "ties": sum(1 for r in records if r["points"] > r["front"])}


# --------------------------------------------------------------------------
def report(results):
    out = []
    if "margin" in results:
        out.append("\n=== 1. decision space against criterion space "
                   f"(n={results['margin'][0]['n']}) ===")
        out.append(f"{'p':>3} {'inst':>5} {'cens':>5} | "
                   f"{'time ratio (median, IQR)':<34} | "
                   f"{'sub-programs ratio':<24}")
        out.append("-" * 76)
        for row in results["margin"]:
            tr, pr = row["time_ratio"], row["program_ratio"]
            if tr is None or tr["n"] * 2 <= row["instances"]:
                out.append(f"{row['p']:>3} {row['instances']:>5} "
                           f"{row['censored']:>5} | more than half censored")
                continue
            out.append(
                f"{row['p']:>3} {row['instances']:>5} {row['censored']:>5} | "
                f"{tr['median']:>7.2f}x  [{tr['q1']:.2f}, {tr['q3']:.2f}]"
                f"   max {tr['max']:>7.1f}x | "
                f"{pr['median']:>6.2f}x  [{pr['q1']:.2f}, {pr['q3']:.2f}]")

    if "seeds" in results:
        s = results["seeds"]
        out.append(f"\n=== 2. what a seed is worth (n={s['n']}, p={s['p']}, "
                   f"{len(s['records'])} paired instances) ===")
        for name in ("pareto", "tchebychev"):
            sp, gap = s[f"speedup_{name}"], s[f"gap_{name}"]
            if sp is None:
                out.append(f"{name:<12} | no paired run completed")
                continue
            out.append(
                f"{name:<12} | speed-up median {sp['median']:.2f}x "
                f"[{sp['q1']:.2f}, {sp['q3']:.2f}]  "
                f"| seed gap median {gap['median']:.3f} "
                f"| exactly optimal {s[f'exact_{name}']}/{len(s['records'])}")

    if "front" in results:
        f = results["front"]
        out.append(f"\n=== 3. enumerating the front (n={f['n']}, p={f['p']}, "
                   f"{len(f['records'])} instances, {f['censored']} censored) ===")
        tr = f["time_ratio"]
        if tr:
            out.append(f"time ratio    | median {tr['median']:.1f}x "
                       f"[{tr['q1']:.1f}, {tr['q3']:.1f}]  "
                       f"min {tr['min']:.1f}x  max {tr['max']:.1f}x")
            nr = f["node_ratio"]
            out.append(f"b&b nodes     | median {nr['median']:.1f}x "
                       f"[{nr['q1']:.1f}, {nr['q3']:.1f}]")
            out.append(f"|front|       | median {f['front']['median']:.0f} "
                       f"[{f['front']['q1']:.0f}, {f['front']['q3']:.0f}]")
            out.append("  the margin against the size of the front:")
            for bucket, cell in f["by_front_size"].items():
                cr = cell["time_ratio"]
                if cr is None:
                    continue
                out.append(f"    |F| in {bucket:<5} ({cell['n']:>3} inst) | "
                           f"median {cr['median']:>6.1f}x "
                           f"[{cr['q1']:.1f}, {cr['q3']:.1f}]  "
                           f"max {cr['max']:.1f}x")

    if "complete" in results:
        c = results["complete"]
        out.append(f"\n=== 4. completing the front (n={c['n']}, p={c['p']}, "
                   f"{len(c['records'])} instances) ===")
        ov = c["overhead"]
        if ov:
            out.append(f"overhead      | median {ov['median']:.1%} "
                       f"[{ov['q1']:.1%}, {ov['q3']:.1%}]  "
                       f"max {ov['max']:.1%}")
            out.append(f"points gained | {c['ties']} of {len(c['records'])} "
                       "instances had a non-singleton slice")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=30)
    parser.add_argument("--budget", type=float, default=60.0,
                        help="seconds per run before a result is censored")
    parser.add_argument("--quick", action="store_true",
                        help="a pilot run: few instances, small p")
    parser.add_argument("--only", action="append", default=None,
                        choices=["margin", "seeds", "front", "complete"])
    args = parser.parse_args()

    instances = 5 if args.quick else args.instances
    budget = 10.0 if args.quick else args.budget
    ps = [2, 3, 4] if args.quick else [2, 3, 4, 5, 6]
    wanted = args.only or ["margin", "seeds", "front", "complete"]

    print(f"campaign: {instances} instances per configuration, "
          f"{budget:.0f}s budget per run")
    print("nothing else should be running on this machine\n")

    results = {"meta": {"instances": instances, "budget": budget,
                        "ps": ps, "quick": args.quick,
                        "started": time.strftime("%Y-%m-%d %H:%M:%S")}}
    started = time.perf_counter()

    if "margin" in wanted:
        print("  [1/4] the margin, per p")
        results["margin"] = experiment_margin(instances, budget, ps)
    if "seeds" in wanted:
        print("  [2/4] what a seed is worth")
        results["seeds"] = experiment_seeds(instances, budget)
    if "front" in wanted:
        print("  [3/4] enumerating the front")
        results["front"] = experiment_front(max(8, instances // 2), budget)
    if "complete" in wanted:
        print("  [4/4] completing the front")
        results["complete"] = experiment_complete(instances, budget)

    results["meta"]["seconds"] = time.perf_counter() - started
    OUT.write_text(json.dumps(results, indent=1, default=str))
    print(report(results))
    print(f"\ntotal {results['meta']['seconds']:.0f}s; "
          f"raw records in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
