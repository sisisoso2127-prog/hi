#!/usr/bin/env python3
"""
bench_bimodal.py
================
LA BIMODALITE COMME QUESTION -- SECONDE VERSION, LA PREMIERE MESURAIT LA
MAUVAISE BORNE.

Ce memoire se sert de la bimodalite des ecarts garantis -- des lignes a
0 %, des lignes au-dessus de 80 %, presque rien entre -- pour expliquer
qu'une mediane est un mauvais juge. C'est un usage DEFENSIF : la forme de
la distribution y sert a ecarter une lecture, jamais a produire une
connaissance. La question evidente n'a jamais ete posee : qu'est-ce qui
SEPARE les deux modes ?

CE QUE LA PREMIERE VERSION FAISAIT DE FAUX. Elle decomposait la borne du
theoreme 5' -- q_ref + U/(Q D) -- lue dans la trace de `certify`. Or cette
borne n'est presque jamais celle qui est publiee : sur 29 executions ou une
borne doit etre produite, le seuil a l'incumbent l'egale ZERO fois, contre
27 pour le seuil libre (remarque « route porteuse »). Le banc decomposait
donc une quantite sans rapport avec le resultat : il affichait un terme de
7,8 puis 24,2 sur des lignes dont l'ecart valait 0,00. Il a ete arrete.

CE QUE CELLE-CI MESURE. La borne publiee vient de la seconde route, et
cette route porte en elle sa propre reponse. A la convergence -- statut
`optimal`, c'est-a-dire max_R G_t <= 0 -- la borne vaut EXACTEMENT max_R f.
L'ecart qui reste ne vient alors plus de la borne du tout : il vient de ce
que le relache R contient encore autre chose que E. C'est un probleme de
COUPES. Quand la route n'a pas converge -- statut `limit` -- la borne est
lache pour son propre compte, et c'est un probleme de BUDGET.

La question « qu'est-ce qui separe les deux modes ? » devient donc
verifiable sans instrument nouveau :

    le mode ouvert est-il peuple de lignes CONVERGEES, auquel cas il faut
    couper davantage -- ou de lignes NON CONVERGEES, auquel cas il faut
    payer davantage ?

Les deux reponses envoient vers des travaux opposes, et c'est ce qui rend
la mesure utile.

CE QU'IL NE FAIT PAS. Prouver une causalite. Il classe des lignes selon un
statut que la methode enregistre deja. Que le mode ouvert soit convergé ne
dit pas que couper davantage le fermerait ; cela dit ou porter l'effort.

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

LOT = [dict(n=n, m=max(3, n // 2 + 1), p=p, seed=s, rhs_scale=1.0, corr=c)
       for n in (20, 30) for p in (3, 4) for c in (0.00, 0.50)
       for s in (1, 2)]

FERME = 1.0        # « ferme » : ecart garanti sous 1 %
OUVERT = 50.0      # « ouvert » : ecart garanti au-dessus de 50 %

# `optimal` : max_R G_t <= 0, la borne vaut max_R f exactement.
# `empty`   : R est vide, meme conclusion en plus fort.
CONVERGE = ("optimal", "empty")


def une(inst, cap: int, graine: int) -> Dict:
    reset_oracle_counter()
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6,
                       seed=graine, ilp_budget=cap, **PROD)
    info = r.cert or {}
    return dict(ecart=(r.gap * 100) if r.gap is not None else float("nan"),
                prouve=bool(r.proved_optimal),
                statut=info.get("geom"), geom=info.get("geom_ub"),
                tours=info.get("geom_tours"), ilp=info.get("geom_ilp"),
                q_lb=float(r.q_lb) if r.q_lb is not None else None,
                q_ub=r.q_ub)


def _m(v) -> Optional[float]:
    v = [x for x in v if x is not None and x == x]
    return statistics.median(v) if v else None


def main() -> int:
    graines = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    print("=" * 98)
    print("CE QUI SEPARE LES DEUX MODES DE L'ECART GARANTI")
    print(f"  {len(LOT)} instances x {len(graines)} graines, plafond {cap}")
    print("=" * 98)
    print(f"\n{'n':>3}{'p':>3}{'corr':>6}{'gr':>4}{'ecart %':>10}"
          f"{'statut':>10}{'tours':>7}{'ilp':>6}"
          f"{'q_lb':>12}{'q_ub':>12}{'pr':>4}")
    print("-" * 98)

    lect: List[Dict] = []
    for spec in LOT:
        inst = generate(**spec)
        for g in graines:
            d = une(inst, cap, g)
            lect.append(d)
            print(f"{spec['n']:>3}{spec['p']:>3}{spec['corr']:>6.2f}{g:>4}"
                  f"{d['ecart']:>10.2f}{str(d['statut']):>10}"
                  f"{_x(d['tours']):>7}{_x(d['ilp']):>6}"
                  f"{_x(d['q_lb']):>12}{_x(d['q_ub']):>12}"
                  f"{'o' if d['prouve'] else '.':>4}", flush=True)

    fin = [d for d in lect if d["ecart"] == d["ecart"]]
    fermes = [d for d in fin if d["ecart"] <= FERME]
    ouverts = [d for d in fin if d["ecart"] >= OUVERT]
    milieu = [d for d in fin if FERME < d["ecart"] < OUVERT]

    print("\n" + "=" * 98)
    print(f"DEUX MODES  --  fermes (<= {FERME:g} %) : {len(fermes)}    "
          f"ouverts (>= {OUVERT:g} %) : {len(ouverts)}    "
          f"entre les deux : {len(milieu)}")
    print("=" * 98)
    if len(milieu) > max(len(fermes), len(ouverts)):
        print("\n!! Le « milieu » est le mode le plus peuple : sur ce lot la")
        print("   distribution n'est PAS bimodale, et la suite ne veut rien")
        print("   dire. C'est un resultat, pas un echec du banc.")

    print(f"\n{'groupe':<12}{'lignes':>8}{'convergees':>13}"
          f"{'non conv.':>12}{'sans statut':>13}{'tours med':>11}"
          f"{'ilp med':>10}")
    for nom, grp in (("fermes", fermes), ("milieu", milieu),
                     ("ouverts", ouverts)):
        if not grp:
            continue
        cv = sum(1 for d in grp if d["statut"] in CONVERGE)
        nc = sum(1 for d in grp if d["statut"] is not None
                 and d["statut"] not in CONVERGE)
        ss = sum(1 for d in grp if d["statut"] is None)
        print(f"{nom:<12}{len(grp):>8}{cv:>13}{nc:>12}{ss:>13}"
              f"{_f(_m([d['tours'] for d in grp])):>11}"
              f"{_f(_m([d['ilp'] for d in grp])):>10}")

    if fermes and ouverts:
        co = sum(1 for d in ouverts if d["statut"] in CONVERGE)
        print(f"\nVERDICT. Sur les {len(ouverts)} lignes du mode OUVERT, "
              f"{co} ont une borne CONVERGEE.")
        if co > len(ouverts) / 2:
            print("La borne y vaut donc max_R f EXACTEMENT : elle n'est pas")
            print("perfectible, et l'ecart restant mesure ce que le relache")
            print("contient encore en plus de E. C'est un probleme de")
            print("COUPES, pas de budget -- payer plus ne fermerait rien.")
        elif co < len(ouverts) / 2:
            print("La borne n'y a donc PAS converge : elle est lache pour")
            print("son propre compte, et c'est un probleme de BUDGET --")
            print("couper davantage ne servirait a rien tant que la route")
            print("n'a pas le temps d'aller au bout.")
        else:
            print("Le mode ouvert est coupe en deux parts egales : le statut")
            print("de convergence ne le separe pas, et il faut chercher")
            print("ailleurs. C'est un resultat negatif, et il compte.")
    return 0


def _x(v) -> str:
    return "--" if v is None else (f"{v:.6g}" if isinstance(v, float)
                                   else str(v))


def _f(v) -> str:
    return "--" if v is None else f"{v:.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
