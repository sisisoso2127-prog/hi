#!/usr/bin/env python3
"""
bench_geom.py
=============
LA SECONDE ROUTE VERS LA BORNE DEPLACE-T-ELLE LES 92-98 % ?

D'ou vient la question. Le banc de couverture (bench_couverture.py) mesure,
sur les instances ou la verite terrain est calculable, trois choses :

  * les coupes centrees sur E TOUT ENTIER retirent 100 % des points fautifs
    (26 lignes sur 26) : l'outil de coupe est SUFFISANT en principe ;
  * l'archive, qui ne retient que 20 a 35 % des vecteurs criteres de E,
    en domine deja pres de 100 % : la couverture N'EST PAS le verrou ;
  * et pourtant, sur les lignes difficiles, la borne annoncee reste tres
    au-dessus du maximum de f sur la region REELLEMENT survivante --
    32,3 contre 8,5 a n = 12 ; 38,8 contre 9,3 a n = 20.

Le verrou n'etait donc ni le plafond de coupes (deja refute par CGLP), ni
l'ordre des coupes, ni la couverture de l'archive : il est dans le SEUIL
auquel le Th. 5' est evalue. La certification pose toujours t = q, alors
que la borne vaut pour TOUT seuil, et que chaque point du relache en
fournit un meilleur. Faire avancer ce seuil est un Dinkelbach ordinaire,
mene sur le relache au lieu de S (cf. `borne_geometrique`).

PROTOCOLE. Identique a bench_cglp.py, pour que les deux resultats se lisent
dans la meme unite : budget DETERMINISTE en appels au solveur entier,
identique dans les deux bras, comparaison APPARIEE instance par instance,
chaque configuration jouee DEUX FOIS pour verifier la reproductibilite au
lieu de la supposer, et coherence des encadrements verifiee a chaque ligne.

CE QUI EST EN JEU POUR LE BRAS « avec ». La seconde route se paie sur le
MEME plafond d'appels entiers : une part lui est reservee en amont, donc la
boucle de coupes en pose moins. Si elle ne rapporte pas plus qu'elle ne
coute, l'ecart doit EMPIRER -- le banc peut donc conclure contre elle.

Usage :  python bench_geom.py [plafond_appels] [nb_graines] [declencheur]
         declencheur : 1 (defaut) pour n'engager la seconde route que la ou
         la premiere ne ferme pas, 0 pour l'engager systematiquement.
"""

import sys
import time

import numpy as np

from molfp_core import ORACLE_CALLS, reset_oracle_counter, upper_bound_over_S
from molfp_instance import generate
from molfp_matheuristic import matheuristic_P

TAILLES = [20, 30, 40]
CORRS = [0.0, 0.5]


def une(inst, cap, seed, geom, gate):
    reset_oracle_counter()
    t0 = time.time()
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6, seed=seed,
                       archive_cuts=True, ilp_budget=cap,
                       cut_diversify=False, geom_bound=geom,
                       geom_gate=gate)
    return {"lb": r.q_lb, "ub": r.q_ub, "ilp": r.ilp_calls,
            "lp": ORACLE_CALLS["lp"], "cuts": r.cert.get("n_cuts", 0),
            "geom": r.cert.get("geom"), "geom_ub": r.cert.get("geom_ub"),
            "gilp": r.cert.get("geom_ilp", 0),
            "gtours": r.cert.get("geom_tours", 0),
            "prouve": r.proved_optimal, "t": time.time() - t0}


def _sig(d):
    return (str(d["lb"]), None if d["ub"] is None else round(d["ub"], 9),
            d["ilp"], d["cuts"])


def main() -> int:
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 900
    n_gr = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    # `gate` : n'engager la seconde route que la ou la premiere ne ferme
    # pas. Le banc sait jouer les deux, parce que la question « la route
    # aide-t-elle ? » et la question « le declencheur evite-t-il ses
    # rechutes ? » ne se repondent pas par la meme mesure.
    gate = (len(sys.argv) <= 3) or sys.argv[3].lower() not in ("0", "non",
                                                              "false")
    graines = list(range(1, n_gr + 1))

    largeur = 112
    print("=" * largeur)
    print(f"SECONDE ROUTE (seuil avance sur le relache) - plafond {cap} "
          f"appels entiers, {n_gr} graines, "
          f"declencheur {'ACTIF' if gate else 'DESACTIVE'}")
    print("=" * largeur)
    print(f"{'n':>4}{'corr':>6}{'gr':>4}{'reprod':>8}"
          f"{'ecart sans':>12}{'ecart avec':>12}{'gain pts':>10}"
          f"{'statut':>9}{'tours':>7}{'ILP 2e':>8}"
          f"{'coupes s/a':>13}{'ILP s/a':>12}{'coherent':>10}")
    print("-" * largeur)

    gains, mieux, pire, egal, alea, ok_tout = [], 0, 0, 0, 0, True
    n_lignes = 0
    ferme = 0
    for n in TAILLES:
        m = max(3, n // 2 + 1)
        for corr in CORRS:
            for g in graines:
                n_lignes += 1
                inst = generate(n=n, m=m, p=3, seed=g, rhs_scale=1.0,
                                corr=corr)
                mS = upper_bound_over_S(inst)
                A = une(inst, cap, g, False, gate)
                A2 = une(inst, cap, g, False, gate)
                B = une(inst, cap, g, True, gate)
                B2 = une(inst, cap, g, True, gate)
                reprod = (_sig(A) == _sig(A2) and _sig(B) == _sig(B2))
                alea += 0 if reprod else 1

                def ecart(d):
                    hi = min(d["ub"] if d["ub"] is not None else np.inf, mS)
                    lo = float(d["lb"])
                    return (hi - lo) / max(1e-12, abs(hi)) * 100 \
                        if np.isfinite(hi) else np.nan

                ea, eb = ecart(A), ecart(B)
                # coherence : les deux encadrements doivent se recouper,
                # sinon l'une des deux bornes est fausse.
                lo = max(float(A["lb"]), float(B["lb"]))
                hi = min(min(A["ub"] or np.inf, mS), min(B["ub"] or np.inf, mS))
                coherent = lo <= hi + 1e-9
                ok_tout &= coherent
                if B["geom"] == "optimal":
                    ferme += 1
                d = ea - eb
                if reprod:
                    gains.append(d)
                    if d > 1e-9:
                        mieux += 1
                    elif d < -1e-9:
                        pire += 1
                    else:
                        egal += 1
                print(f"{n:>4}{corr:>6.2f}{g:>4}"
                      f"{('oui' if reprod else 'NON'):>8}"
                      f"{ea:>11.1f}%{eb:>11.1f}%{d:>+9.2f}"
                      f"{str(B['geom']):>9}{B['gtours']:>7}{B['gilp']:>8}"
                      f"{A['cuts']:>7}/{B['cuts']:<5}"
                      f"{A['ilp']:>6}/{B['ilp']:<5}"
                      f"{'ok' if coherent else 'KO':>10}", flush=True)

    print("-" * largeur)
    print(f"  REPRODUCTIBILITE : {n_lignes - alea}/{n_lignes} lignes stables")
    if gains:
        d = np.asarray(gains)
        print(f"  points d'ecart garanti gagnes : median {np.median(d):+.2f}   "
              f"moyen {d.mean():+.2f}   max {d.max():+.2f}   min {d.min():+.2f}")
    print(f"  MIEUX {mieux}   PIRE {pire}   EGAL {egal}")
    print(f"  SECONDE ROUTE CONVERGEE (max_R f atteint) : {ferme}/{n_lignes}")
    print(f"  COHERENCE des encadrements : "
          f"{'TOUT VALIDE' if ok_tout else 'ECHEC'}")
    print()
    print("  Lecture du cout. La seconde route se paie sur le MEME plafond")
    print("  d'appels entiers : la colonne « coupes » montre ce que la boucle")
    print("  de coupes a du ceder pour la financer. Un gain positif signifie")
    print("  donc qu'un seuil avance vaut mieux que les coupes abandonnees.")
    return 0 if ok_tout else 1


if __name__ == "__main__":
    sys.exit(main())
