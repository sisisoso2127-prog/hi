#!/usr/bin/env python3
"""
bench_zm.py
===========
MATCH sur le terrain de Zerdani & Moulai (2011) : phi LINEAIRE, criteres
FRACTIONNAIRES. Leur methode, reimplementee, contre la notre, sur les memes
instances.

L'UNITE DE COUT. Les secondes ne comparent rien ici : deux implementations,
un simplexe rationnel exact d'un cote, HiGHS de l'autre. On compte donc les
APPELS AU SOLVEUR EN NOMBRES ENTIERS, qui sont l'operation chere des deux
cotes -- chez eux les tests d'efficacite et la relaxation initiale, chez nous
tout. Cette unite est independante de la machine et du langage.

ELLE FAVORISE LEUR METHODE, ET C'EST VOULU. Leur algorithme tire l'essentiel
de son travail du TABLEAU du simplexe -- unicite de l'optimum lue dans
gamma, aretes lues dans les colonnes -- operations que cette unite ne compte
pas, alors qu'elle compte tout ce que nous faisons. Un desavantage assume :
si notre methode tient malgre cela, la conclusion est solide ; si elle ne
tient pas, il faut le dire.

CE QUI EST VERIFIE A CHAQUE INSTANCE, par enumeration exhaustive :
  * la valeur rendue par chaque methode vaut-elle bien l'optimum sur E ;
  * notre encadrement q_lb <= q* <= q_ub tient-il ;
  * leur methode a-t-elle mene son processus a terme, ou s'est-elle arretee
    sur le sommet fractionnaire que l'article laisse aux coupes de Gomory.

Usage :  python bench_zm.py [n_instances] [plafond_appels]
"""

import sys
from fractions import Fraction

import numpy as np

from molfp_core import ORACLE_CALLS, reset_oracle_counter
from molfp_enum import ground_truth
from molfp_instance import FracObj, MOILFP
from molfp_matheuristic import matheuristic_P
from zerdani_moulai import zerdani_moulai


def instance_zm(n: int, m: int, p: int, seed: int) -> MOILFP:
    """
    Classe de Zerdani & Moulai : criteres FRACTIONNAIRES, phi LINEAIRE.

    (A1) est assuree par construction (d_k >= 0, beta_k >= 1), donc les
    denominateurs valent au moins 1 sur tout le domaine. phi peut avoir des
    coefficients negatifs -- c'est le cas dans leur exemple, phi = 2x1 - 3x2 --
    et c'est ce qui rend le probleme non trivial : sans cela l'optimum sur E
    serait souvent atteint au bord evident.
    """
    rng = np.random.default_rng(seed)
    A = rng.integers(1, 6, size=(m, n))
    b = np.maximum(2, (1.4 * A.sum(axis=1)).astype(int))
    Z = [FracObj(num=rng.integers(0, 8, size=n), a=int(rng.integers(0, 4)),
                 den=rng.integers(0, 3, size=n), b=int(rng.integers(1, 4)))
         for _ in range(p)]
    f = FracObj(num=rng.integers(-6, 7, size=n), a=0,
                den=np.zeros(n, dtype=int), b=1)      # phi LINEAIRE
    inst = MOILFP(A=A.astype(int), b=b.astype(int), Z=Z, f=f,
                  name=f"zm_n{n}_m{m}_p{p}_s{seed}", seed=seed)
    inst.check_assumptions()
    return inst


def main() -> int:
    n_inst = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 400

    print("=" * 112)
    print(f"MATCH SUR LE TERRAIN DE ZERDANI & MOULAI - phi lineaire, "
          f"criteres fractionnaires, {n_inst} instances")
    print(f"cout compte : appels au solveur ENTIER (notre plafond : {cap})")
    print("=" * 112)
    print(f"{'instance':<20}{'|S|':>7}{'|E|':>6}{'phi*':>8}"
          f"{'  ZM: valeur':>13}{'statut':>22}{'coupes':>7}{'tests':>7}"
          f"{'  nous: valeur':>15}{'prouve':>8}{'ILP':>6}{'ok':>4}")
    print("-" * 112)

    zm_exact, zm_termine, nous_exact, nous_prouve = 0, 0, 0, 0
    zm_trouve_non_prouve = 0
    zm_ilp, nous_ilp, n_ok, total = [], [], 0, 0
    for i in range(n_inst):
        n = 3 + i % 4
        m = 2 + i % 3
        p = 2 + i % 2
        inst = instance_zm(n, m, p, 3000 + i)
        gt = ground_truth(inst, limit=200_000)
        phi_star = gt.q_star
        total += 1

        reset_oracle_counter()
        rz = zerdani_moulai(inst)
        zm_ilp.append(rz.ilp_calls)
        zm_ok = (rz.phi_opt == phi_star)
        zm_exact += int(zm_ok)
        zm_termine += int(rz.status == "optimal")
        zm_trouve_non_prouve += int(zm_ok and rz.status != "optimal")

        reset_oracle_counter()
        rn = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6, seed=0,
                            archive_cuts=True, ilp_budget=cap)
        nous_ilp.append(rn.ilp_calls)
        nous_ok = (rn.q_lb == phi_star)
        nous_exact += int(nous_ok)
        nous_prouve += int(rn.proved_optimal)
        encadre = (rn.q_lb <= phi_star
                   and (rn.q_ub is None or rn.q_ub >= float(phi_star) - 1e-9))
        coherent = encadre and (rz.phi_opt is None or rz.phi_opt <= phi_star)
        n_ok += int(coherent)

        vz = "-" if rz.phi_opt is None else str(rz.phi_opt)
        print(f"{inst.name:<20}{len(gt.S):>7}{len(gt.E):>6}{str(phi_star):>8}"
              f"{vz:>13}{rz.status:>22}{rz.coupes + rz.gomory:>6}"
              f"{rz.tests_efficacite:>7}"
              f"{str(rn.q_lb):>15}{str(rn.proved_optimal):>8}"
              f"{rn.ilp_calls:>6}{'ok' if coherent else 'KO':>4}", flush=True)

    print("-" * 112)
    print(f"  ZERDANI & MOULAI  : optimum trouve {zm_exact}/{total}   "
          f"processus mene a terme {zm_termine}/{total}   "
          f"(dont {zm_trouve_non_prouve} trouve mais NON certifie)   "
          f"appels entiers medians {int(np.median(zm_ilp))}")
    print(f"  NOTRE METHODE     : optimum trouve {nous_exact}/{total}   "
          f"optimalite PROUVEE {nous_prouve}/{total}   "
          f"appels entiers medians {int(np.median(nous_ilp))}")
    print(f"  COHERENCE (encadrement valide, valeurs <= optimum) : "
          f"{n_ok}/{total}")
    print()
    print("  Rappel : l'unite de cout ne compte pas les pivots du simplexe,")
    print("  d'ou leur methode tire l'essentiel de son travail. Elle leur est")
    print("  donc favorable, et volontairement.")
    print("  « processus mene a terme » distingue les instances ou leur")
    print("  algorithme s'acheve de celles ou il atteint le sommet")
    print("  fractionnaire que l'article laisse aux coupes de Gomory sans les")
    print("  detailler -- cas ou il peut avoir DEJA trouve l'optimum, comme")
    print("  sur leur propre exemple publie, sans pouvoir le certifier.")
    return 0 if n_ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
