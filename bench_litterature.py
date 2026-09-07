#!/usr/bin/env python3
"""
bench_litterature.py
====================
Notre methode sur le TERRAIN de Drici, Ouail & Moulai (2018), avec LEUR
protocole de generation et LEURS tailles.

POURQUOI CE BANC, ET CE QU'IL PEUT DIRE. Aucune des deux methodes voisines
ne resout notre probleme : Zerdani & Moulai (2011) maximisent une fonction
LINEAIRE sur l'ensemble efficace d'un MOILFP, Drici et al. (2018) une
fonction FRACTIONNAIRE sur celui d'un MOILP ; notre (P) est fractionnaire
des DEUX cotes. Une comparaison n'a donc de sens que sur le terrain de
chacun -- ici celui de Drici et al., dont la classe s'obtient en posant
d_k = 0 et b_k = 1.

LEUR PROTOCOLE, cite section 6 de l'article :
    coefficients de contraintes  : uniformes discrets sur [1, 30]
    second membre b              : uniforme discret sur [50, 100]
    coefficients C des criteres  : uniformes discrets sur [-10, 10]
    p et alpha (numerateur de f) : comme C
    q et beta                    : tels que q^T x + beta > 0 sur X
    10 problemes par taille (n, m, r)

Nous le reprenons a l'identique. Pour q et beta nous prenons q >= 0 et
beta >= 1, ce qui garantit q^T x + beta >= 1 > 0 sur X puisque x >= 0 :
c'est la condition des auteurs, obtenue par construction.

CE QUI EST COMPARABLE, ET CE QUI NE L'EST PAS. Leurs temps sont mesures en
MATLAB sur un MacBook Pro de 2017 ; les notres en Python avec HiGHS sur une
autre machine, des annees plus tard. Comparer les SECONDES n'aurait aucun
sens et nous ne le faisons pas. Ce qui se compare :

  * la TAILLE des instances traitees, a la meme regle de generation ;
  * la PROPORTION d'instances resolues avec optimalite PROUVEE ;
  * pour nous, le nombre d'appels au solveur entier, unite independante de
    la machine -- que les auteurs ne rapportent pas, leur mesure etant en
    noeuds et en coupes.

La colonne « CPU publie » est reproduite a titre indicatif seulement, et
sert a situer l'ordre de grandeur des tailles atteintes, non a departager
les implementations.

DIFFERENCE DE NATURE A GARDER EN TETE. Leur methode est EXACTE : elle rend
l'optimum ou rien. La notre encadre a tout instant. Sur une instance qu'elle
ne prouve pas, elle rend tout de meme une solution et un ecart garanti --
ce que leur branch and cut, interrompu, ne fournit pas. C'est la difference
que ce banc doit faire apparaitre, plus que des secondes.

Usage :  python bench_litterature.py [budget_s] [n_par_taille]
"""

import sys
import time

import numpy as np

from molfp_core import ORACLE_CALLS, reset_oracle_counter
from molfp_instance import FracObj, MOILFP
from molfp_matheuristic import matheuristic_P

# (n, m, r) et le temps CPU moyen publie par Drici et al., table 11, r = 3
TAILLES_DRICI = [
    (5, 5, 3, 0.21), (10, 5, 3, 0.44), (15, 5, 3, 0.70),
    (20, 5, 3, 1.23), (20, 10, 3, 1.45), (30, 10, 3, 8.98),
    (35, 15, 3, 7.45), (40, 15, 3, 10.82),
]


def instance_drici(n: int, m: int, r: int, seed: int) -> MOILFP:
    """Generation suivant le protocole de la section 6 de Drici et al."""
    rng = np.random.default_rng(seed)
    A = rng.integers(1, 31, size=(m, n))              # [1, 30]
    b = rng.integers(50, 101, size=m)                 # [50, 100]
    Z = [FracObj(num=rng.integers(-10, 11, size=n),   # [-10, 10]
                 a=0, den=np.zeros(n, dtype=int), b=1)
         for _ in range(r)]                           # criteres LINEAIRES
    f = FracObj(num=rng.integers(-10, 11, size=n),
                a=int(rng.integers(-10, 11)),
                den=rng.integers(0, 11, size=n),      # q >= 0
                b=int(rng.integers(1, 11)))           # beta >= 1 => q x + b > 0
    inst = MOILFP(A=A.astype(int), b=b.astype(int), Z=Z, f=f,
                  name=f"drici_n{n}_m{m}_r{r}_s{seed}", seed=seed)
    inst.check_assumptions(strict=False)
    return inst


def main() -> int:
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    print("=" * 104)
    print(f"NOTRE METHODE SUR LE TERRAIN DE DRICI ET AL. (2018) - "
          f"leur protocole, leurs tailles, {n_seeds} instances par taille")
    print(f"budget {budget:g} s par instance")
    print("=" * 104)
    print(f"{'n x m':>9}{'r':>3}{'prouve':>9}{'ecart garanti median':>23}"
          f"{'ILP med':>10}{'t med (s)':>11}{'CPU publie':>13}{'valide':>9}")
    print("-" * 104)

    tout_valide = True
    for n, m, r, cpu_pub in TAILLES_DRICI:
        prouves, ecarts, ilps, temps = 0, [], [], []
        valide = True
        for s in range(n_seeds):
            inst = instance_drici(n, m, r, 1000 + s)
            reset_oracle_counter()
            t0 = time.time()
            res = matheuristic_P(inst, time_budget=budget * 0.6,
                                 bound_budget=budget * 0.4, seed=0,
                                 archive_cuts=True)
            temps.append(time.time() - t0)
            ilps.append(res.ilp_calls)
            prouves += int(res.proved_optimal)
            # controle de surete : la borne doit encadrer la valeur rendue
            if res.q_ub is not None and res.q_lb is not None:
                valide &= res.q_ub >= float(res.q_lb) - 1e-9
            ecarts.append(res.gap * 100 if res.gap is not None else np.nan)
        tout_valide &= valide
        med = np.nanmedian(ecarts) if not all(np.isnan(ecarts)) else np.nan
        print(f"{n:>4} x{m:>3}{r:>3}{prouves:>6}/{n_seeds}"
              f"{med:>22.1f}%{int(np.median(ilps)):>10}"
              f"{np.median(temps):>11.2f}{cpu_pub:>13.2f}"
              f"{'ok' if valide else 'KO':>9}", flush=True)

    print("-" * 104)
    print(f"  VALIDITE q_lb <= q_ub : "
          f"{'TOUT VALIDE' if tout_valide else 'ECHEC'}")
    print()
    print("  LECTURE. La colonne « CPU publie » est le temps moyen rapporte par")
    print("  les auteurs (MATLAB, MacBook Pro, r = 3). Elle situe l'ordre de")
    print("  grandeur des tailles atteintes ; elle NE DEPARTAGE PAS les")
    print("  implementations, mesurees sur des machines et des langages")
    print("  differents a des annees d'intervalle.")
    print("  Ce qui se compare vraiment : leur methode est EXACTE et ne rend")
    print("  rien si on l'interrompt ; la notre encadre q* a tout instant, donc")
    print("  meme sur une instance non prouvee elle rend une solution ET un")
    print("  ecart garanti. C'est cette difference que la colonne « ecart")
    print("  garanti median » fait apparaitre.")
    return 0 if tout_valide else 1


if __name__ == "__main__":
    sys.exit(main())
