"""Emit LaTeX table bodies from ``campaign-results.json``.

Every other table in ``docs/`` was transcribed by hand from the campaign's
printed output.  That is a quiet error source in a project that has already
had to withdraw a figure and correct a chart caption, so the break-even table
is generated instead: run this, paste the block between the rules, and the
document cannot drift from the measurement.

    python3 studies/tables.py breakeven

Prints nothing but the rows, so the output is safe to paste directly.
"""

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "campaign-results.json"


def _f(x, places=1):
    return f"{x:.{places}f}"


def breakeven(data):
    rows = data.get("breakeven")
    if not rows:
        raise SystemExit("no breakeven experiment in campaign-results.json")
    out = []
    allk = []
    for b in rows:
        kp, size = b["k_programs"], b["size"]
        allk += [r["k_programs"] for r in b["records"]
                 if r["k_programs"] is not None]
        cfg = f"$n={b['n']}$, $p={b['p']}$"
        kept = len(b["records"])
        if kp is None or size is None:
            out.append(f"{cfg} & {kept} & {b['censored']} & --- & --- \\\\")
            continue
        out.append(
            f"{cfg} & {kept} & {b['censored']} & "
            f"${size['median']:.0f}$ & "
            f"$\\mathbf{{{_f(kp['median'])}}}$ "
            f"$[{_f(kp['q1'])},\\ {_f(kp['q3'])}]$ \\\\")
    if allk:
        allk.sort()
        import statistics
        half = len(allk) // 2
        q1 = statistics.median(allk[:half]) if len(allk) >= 4 else allk[0]
        q3 = statistics.median(allk[-half:]) if len(allk) >= 4 else allk[-1]
        out.append("\\midrule")
        out.append(
            f"\\textbf{{all}} & {len(allk)} & & & "
            f"$\\mathbf{{{_f(statistics.median(allk))}}}$ "
            f"$[{_f(q1)},\\ {_f(q3)}]$ \\\\")
        out.append(f"% min {_f(min(allk))}  max {_f(max(allk))}")
    return "\n".join(out)


def filter_(data):
    f = data.get("filter")
    if not f:
        raise SystemExit("no filter experiment in campaign-results.json")
    ms = f["ms"]
    out = []
    for r in f["rows"]:
        cells = []
        for m in ms:
            st = r["shares"].get(str(m))
            cells.append(f"${st['median']*100:.0f}\\%$" if st else "---")
        sat = r["saturates_at"]
        thin = "$^{\\dagger}$" if r["instances"] < 15 else ""
        out.append(f"${r['size']['median']:.0f}${thin} & {r['instances']} & "
                   + " & ".join(cells)
                   + f" & $\\mathbf{{m={sat}}}$ \\\\"
                   if sat else " & --- \\\\")
    out.append(f"% columns: m = {', '.join(str(m) for m in ms)}")
    out.append("% dagger marks a row resting on fewer than 15 instances")
    return "\n".join(out)


def order(data):
    o = data.get("order")
    if not o:
        raise SystemExit("no order experiment in campaign-results.json")
    kept = len(o["records"])
    out = []
    for name, label in (("archive", "archive order"),
                        ("probe", "\\textbf{probe order}"),
                        ("reversed", "reversed (control)")):
        st = o[f"{name}_probes"]
        if st is None:
            continue
        med = (f"$\\mathbf{{{st['median']:.2f}}}\\times$" if name == "probe"
               else f"${st['median']:.2f}\\times$")
        raised = o[f"{name}_raised"]
        rz = (f"\\textbf{{none}}" if raised == 0 else f"{raised} of {kept}")
        out.append(f"{label} & {med} & $[{st['q1']:.2f},\\ {st['q3']:.2f}]$ "
                   f"& {rz} \\\\")
    out.append(f"% {kept} instances, {o['censored']} censored")
    return "\n".join(out)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "breakeven"
    data = json.loads(RESULTS.read_text())
    print({"breakeven": breakeven, "filter": filter_,
       "order": order}[which](data))


if __name__ == "__main__":
    main()
