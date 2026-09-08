#!/usr/bin/env python3
"""
verify_geom.py
==============
VALIDITE de la seconde route (`borne_geometrique`).

Une borne superieure fausse est la seule faute que cette methode ne peut pas
se permettre : elle transformerait un « ecart garanti » en affirmation vide,
et pire, un « optimalite prouvee » en mensonge. La propriete a verifier est
donc unique et non negociable :

        q_ub  >=  q*        sur CHAQUE execution.

On la verifie par ENUMERATION EXHAUSTIVE, en arithmetique rationnelle
exacte, sur des instances assez petites pour que q* soit connu sans le
moindre doute. Trois controles :

  C1  la borne du bras « avec » majore q* ;
  C2  la borne du bras « sans » la majore aussi (temoin : si C2 tombe, le
      probleme n'est pas dans la seconde route) ;
  C3  quand la seconde route annonce le statut 'optimal', elle pretend avoir
      atteint max_R f. On verifie alors qu'elle majore bien q*, et qu'elle
      ne descend PAS sous lui -- c'est la ou une erreur de raisonnement se
      manifesterait en premier.

On balaie aussi corr, qui gouverne l'epaisseur de E, parce que la coupe
d'efficacite -- la seule qui retire des points EFFICACES, et donc la seule
qui pourrait faire descendre la borne trop bas -- ne se declenche pas de la
meme facon selon ce regime.

UN PLAFOND SERRE EST NECESSAIRE. Sous un plafond confortable, ces
instances se prouvent optimales avant que la seconde route ne s'execute :
le statut vaut 'None' partout et C3 ne teste rien. On rejoue donc le lot
sous un plafond DELIBEREMENT INSUFFISANT, ou la certification n'aboutit
pas et ou la seconde route est justement sollicitee. C'est la seule facon
de tester ce qu'on pretend tester.

Usage :  python verify_geom.py [nb_graines] [plafond_appels]
"""

import sys

import numpy as np

from molfp_core import reset_oracle_counter
from molfp_enum import ground_truth
from molfp_instance import generate
from molfp_matheuristic import matheuristic_P

TAILLES = [(5, 1.8), (6, 1.5), (7, 1.2), (8, 1.0)]
CORRS = [0.00, 0.50, 0.90]
PS = [2, 3, 4]


def main() -> int:
    n_gr = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    print("=" * 96)
    print(f"VALIDITE DE LA SECONDE ROUTE - verite terrain par enumeration "
          f"exacte, {n_gr} graines, plafond {cap}")
    print("=" * 96)
    print(f"{'instance':<26}{'q*':>10}{'ub sans':>11}{'ub avec':>11}"
          f"{'statut':>10}{'C1':>5}{'C2':>5}{'C3':>5}")
    print("-" * 96)

    ko1 = ko2 = ko3 = 0
    n_tot = n_opt = 0
    serres = []
    for n, rhs in TAILLES:
        m = max(3, n // 2 + 1)
        for p in PS:
            for corr in CORRS:
                for g in range(1, n_gr + 1):
                    inst = generate(n=n, m=m, p=p, seed=g, rhs_scale=rhs,
                                    corr=corr)
                    gt = ground_truth(inst, limit=300_000)
                    qs = float(gt.q_star)

                    res = {}
                    for nom, geom in (("sans", False), ("avec", True)):
                        reset_oracle_counter()
                        r = matheuristic_P(inst, time_budget=1e6,
                                           bound_budget=1e6, seed=g,
                                           archive_cuts=True, ilp_budget=cap,
                                           cut_diversify=False,
                                           geom_bound=geom,
                                           # validite de la ROUTE, pas de son
                                           # declencheur : on la force.
                                           geom_gate=False)
                        res[nom] = r

                    ua = res["avec"].q_ub
                    us = res["sans"].q_ub
                    st = res["avec"].cert.get("geom")

                    c1 = ua is None or ua >= qs - 1e-7
                    c2 = us is None or us >= qs - 1e-7
                    c3 = (st != "optimal") or (ua is not None
                                               and ua >= qs - 1e-7)
                    ko1 += (not c1)
                    ko2 += (not c2)
                    ko3 += (not c3)
                    n_tot += 1
                    n_opt += (st == "optimal")
                    if ua is not None and us is not None and us > 1e-12:
                        serres.append((us - ua) / abs(us) * 100)

                    nom = f"n{n} p{p} c{corr:.2f} g{g}"
                    flag = "" if (c1 and c2 and c3) else "   <-- ECHEC"
                    print(f"{nom:<26}{qs:>10.4f}"
                          f"{(us if us is not None else float('nan')):>11.4f}"
                          f"{(ua if ua is not None else float('nan')):>11.4f}"
                          f"{str(st):>10}"
                          f"{('ok' if c1 else 'KO'):>5}"
                          f"{('ok' if c2 else 'KO'):>5}"
                          f"{('ok' if c3 else 'KO'):>5}{flag}", flush=True)

    print("-" * 96)
    print(f"  C1  borne « avec » >= q*   : {n_tot - ko1}/{n_tot}")
    print(f"  C2  borne « sans » >= q*   : {n_tot - ko2}/{n_tot}")
    print(f"  C3  statut 'optimal' sain  : {n_tot - ko3}/{n_tot}")
    print(f"  seconde route convergee    : {n_opt}/{n_tot}")
    if serres:
        s = np.asarray(serres)
        print(f"  resserrement de la borne   : median {np.median(s):+.2f} %  "
              f"moyen {s.mean():+.2f} %  max {s.max():+.2f} %")
    ok = (ko1 == 0 and ko2 == 0 and ko3 == 0)
    print(f"  VERDICT : {'TOUT VALIDE' if ok else 'ECHEC DE VALIDITE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
