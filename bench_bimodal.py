#!/usr/bin/env python3
"""
bench_bimodal.py
================
LA BIMODALITE COMME QUESTION, ET NON PLUS COMME EXCUSE.

Ce memoire se sert de la bimodalite des ecarts garantis -- des lignes a
0 %, des lignes au-dessus de 80 %, presque rien entre les deux -- pour
expliquer pourquoi une mediane est un mauvais juge. C'est correct, et
c'est un usage DEFENSIF : la forme de la distribution y sert a ecarter une
lecture, jamais a produire une connaissance. Nous ne lui avons jamais pose
la question evidente : QU'EST-CE QUI SEPARE LES DEUX MODES ?

Il se trouve que la methode enregistre deja de quoi repondre. La borne du
theoreme 5' s'ecrit

        q_ub  =  q_ref  +  U / (Q * D)

et `certify` en garde la trace par tour : q_ref, Q, U et D separement --
precisement pour qu'une borne qui recule soit diagnosticable. L'ecart
garanti n'est donc pas une grandeur opaque : c'est U / (Q * D) rapporte a
q_ub, et il a TROIS causes possibles, qu'on peut departager au lieu de les
deviner.

  U        ce que le relache laisse encore sur la table. Grand U = coupes
           insuffisantes, ou region encore trop lache.
  D        le denominateur garanti (D+ quand la seconde route le repare).
           Petit D = la division explose, et la borne avec elle.
  Q        le facteur d'echelle du substitut.

L'hypothese que nous testons est celle que la theorie designe : les lignes
qui ne ferment pas seraient celles ou D est PETIT, pas celles ou U est
grand -- autrement dit un defaut de la BORNE, pas de la recherche. Si
c'est U qui separe les modes, la conclusion est inverse et vise les
coupes. Les deux sont publiables ; ce qui ne l'est pas, c'est de ne pas
avoir regarde.

CE QUE CE BANC NE FAIT PAS. Il ne prouve aucune causalite : il decompose
une identite. Que D soit petit sur les lignes ouvertes ne dit pas que
reparer D les fermerait -- U pourrait grandir d'autant. Il dit ou regarder
ensuite, et c'est tout ce qu'une decomposition peut dire.

Usage :  python bench_bimodal.py [graines] [plafond]
"""

from __future__ import annotations

import statistics
import sys
from typing import Dict, List, Optional

from molfp_core import reset_oracle_counter
from molfp_instance import generate
from molfp_hybride import matheuristic_P
from bench_sensibilite import PROD

# Un lot plus large que d'habitude : departager deux modes demande des
# effectifs dans CHACUN, et non une mediane globale.
LOT = [dict(n=n, m=max(3, n // 2 + 1), p=p, seed=s, rhs_scale=1.0, corr=c)
       for n in (20, 30) for p in (3, 4) for c in (0.00, 0.50)
       for s in (1, 2)]

FERME = 1.0        # « ferme » : ecart garanti sous 1 %
OUVERT = 50.0      # « ouvert » : ecart garanti au-dessus de 50 %


def une(inst, cap: int, graine: int) -> Dict:
    reset_oracle_counter()
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6,
                       seed=graine, ilp_budget=cap, **PROD)
    trace = (r.cert or {}).get("trace") or []
    # le dernier tour est celui qui a produit la borne publiee
    d = trace[-1] if trace else {}
    return dict(ecart=(r.gap * 100) if r.gap is not None else float("nan"),
                prouve=bool(r.proved_optimal),
                q_ref=d.get("q_ref"), Q=d.get("Q"), U=d.get("U"),
                denom=d.get("denom"), q_ub=r.q_ub, cuts=d.get("cuts"))


def _med(v: List[float]) -> Optional[float]:
    v = [x for x in v if x is not None and x == x]
    return statistics.median(v) if v else None


def _aff(nom: str, v: List[Optional[float]]) -> str:
    m = _med(v)
    return f"{nom} {m:>12.4g}" if m is not None else f"{nom} {'--':>12}"


def main() -> int:
    graines = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    print("=" * 100)
    print("DECOMPOSITION DE L'ECART GARANTI :  q_ub = q_ref + U / (Q * D)")
    print(f"  {len(LOT)} instances x {len(graines)} graines, plafond {cap}")
    print("=" * 100)
    print(f"\n{'n':>3}{'p':>3}{'corr':>6}{'gr':>4}{'ecart %':>10}"
          f"{'q_ref':>12}{'U':>12}{'Q':>8}{'D':>12}{'U/(QD)':>12}{'pr':>4}")
    print("-" * 100)

    lect: List[Dict] = []
    for spec in LOT:
        inst = generate(**spec)
        for g in graines:
            d = une(inst, cap, g)
            d.update(spec=spec, graine=g)
            lect.append(d)
            terme = (d["U"] / (d["Q"] * d["denom"])
                     if None not in (d["U"], d["Q"], d["denom"])
                     and d["Q"] and d["denom"] else None)
            d["terme"] = terme
            print(f"{spec['n']:>3}{spec['p']:>3}{spec['corr']:>6.2f}{g:>4}"
                  f"{d['ecart']:>10.2f}"
                  f"{_x(d['q_ref']):>12}{_x(d['U']):>12}{_x(d['Q']):>8}"
                  f"{_x(d['denom']):>12}{_x(terme):>12}"
                  f"{'o' if d['prouve'] else '.':>4}", flush=True)

    fermes = [d for d in lect if d["ecart"] == d["ecart"] and d["ecart"] <= FERME]
    ouverts = [d for d in lect if d["ecart"] == d["ecart"] and d["ecart"] >= OUVERT]
    milieu = [d for d in lect if d["ecart"] == d["ecart"]
              and FERME < d["ecart"] < OUVERT]

    print("\n" + "=" * 100)
    print(f"DEUX MODES  --  fermes (<= {FERME:g} %) : {len(fermes)}    "
          f"ouverts (>= {OUVERT:g} %) : {len(ouverts)}    "
          f"entre les deux : {len(milieu)}")
    print("=" * 100)
    if len(milieu) > max(len(fermes), len(ouverts)):
        print("\n!! Le « milieu » est le mode le plus peuple : sur ce lot la")
        print("   distribution n'est PAS bimodale, et la suite ne veut rien")
        print("   dire. C'est un resultat, pas un echec du banc.")

    for nom, grp in (("fermes ", fermes), ("ouverts", ouverts)):
        if not grp:
            continue
        print(f"\n  {nom} (medianes) : "
              + "   ".join([_aff("q_ref", [d["q_ref"] for d in grp]),
                            _aff("U", [d["U"] for d in grp]),
                            _aff("Q", [d["Q"] for d in grp]),
                            _aff("D", [d["denom"] for d in grp]),
                            _aff("U/(QD)", [d["terme"] for d in grp])]))

    if fermes and ouverts:
        print("\n  RAPPORT ouverts / fermes, terme par terme :")
        for cle, lib in (("U", "U      "), ("Q", "Q      "),
                         ("denom", "D      "), ("terme", "U/(QD) ")):
            a, b = _med([d[cle] for d in ouverts]), _med([d[cle] for d in fermes])
            if a is None or b is None or b == 0:
                print(f"    {lib} --")
            else:
                print(f"    {lib} {a / b:>10.3f}")
        print("\n  Lecture. Un rapport proche de 1 sur un terme dit que ce")
        print("  terme ne separe pas les modes. Le terme dont le rapport")
        print("  s'ecarte le plus est celui qui porte la bimodalite -- et")
        print("  c'est lui, non la mediane globale, qu'il faudra attaquer.")
    return 0


def _x(v) -> str:
    return "--" if v is None else f"{v:.4g}"


if __name__ == "__main__":
    raise SystemExit(main())
