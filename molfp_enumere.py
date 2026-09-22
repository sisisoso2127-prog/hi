#!/usr/bin/env python3
"""
molfp_enumere.py
================
ENUMERER L'ENSEMBLE EFFICACE, AVEC CERTIFICAT DE COMPLETUDE.

Ce depot rend jusqu'ici un SOUS-ENSEMBLE de E : l'archive, dont chaque
point est certifie efficace (purete mesuree a 1 sur 8/8 instances) mais qui
n'en retient qu'environ 35 % des vecteurs criteres. C'etait ce que le sujet
demandait, et c'est ce que les indicateurs mesurent. Ce fichier repond a
l'autre question -- rendre E TOUT ENTIER -- avec les memes briques.

LE PRINCIPE, ET IL N'EST PAS DE NOUS. La coupe d'efficacite retire
{ x : Z(x) <= Z(a) } pour un `a` efficace certifie, et sa demonstration
etablit que tout point efficace de vecteur criteres DIFFERENT de Z(a) y
survit. Iterer -- resoudre, certifier, couper -- enumere donc les vecteurs
criteres efficaces un par un, et l'infaisabilite du relache CERTIFIE qu'il
n'en reste aucun. C'est exactement la methode de Sylva & Crema (2004), et
nous le disons ici comme le memoire le dit ailleurs : l'enumeration par
cette coupe est anterieure a ce travail. Ce qui lui appartient est le
CADRE FRACTIONNAIRE -- la coupe e_k rendue entiere par produit croise,
valide quand le denominateur varie avec x, la ou Sylva & Crema traitent des
couts lineaires et etendent aux couts rationnels par un ppcm.

CE QUE CE PROGRAMME GARANTIT.

  * Chaque point rendu est EFFICACE, certifie par la chaine de reparation
    (Th. 2) avant que sa coupe ne soit posee. Une coupe centree sur un
    point non certifie retirerait des points efficaces : la certification
    n'est pas une precaution, c'est la condition de validite.
  * Deux points rendus ont des vecteurs criteres DISTINCTS.
  * Si la boucle s'arrete sur une infaisabilite, la liste est COMPLETE au
    sens des vecteurs criteres : Z(E) tout entier.
  * Si elle s'arrete sur le plafond, la liste est un sous-ensemble
    correct de Z(E), et le programme le DIT au lieu de le taire.

CE QU'IL NE REND PAS. Les solutions alternatives : un vecteur criteres
atteint par plusieurs x ne donne qu'un representant, puisque la coupe
retire { Z(x) <= Z(a) } et donc tous ses jumeaux. Rendre E au sens des
SOLUTIONS demanderait, pour chaque vecteur retenu, une enumeration des
optima alternatifs sous Z(x) = Z(a) -- un travail separe, et qui n'est pas
fait ici.

CE QU'IL COUTE. Un programme entier par vecteur efficace, plus les
resolutions de la chaine de reparation. Le nombre de vecteurs efficaces
croit vite : sur le lot de ce depot, |Z(E)| atteint 218 a n=7 et 319 a
n=12. L'enumeration complete n'est donc pas une methode de resolution pour
les grandes instances -- c'est un oracle exact pour les petites, et un
generateur anytime pour les autres.

Usage :  python molfp_enumere.py                 demonstration + validation
         python molfp_enumere.py --valider N     valide sur N instances
"""

from __future__ import annotations

import sys
import time
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from molfp_core import ORACLE_CALLS, reset_oracle_counter, set_ilp_budget
from molfp_instance import MOILFP, generate
from molfp_oracle import ECutModel, repair_to_efficient


class Resultat(NamedTuple):
    points: List[np.ndarray]          # un representant par vecteur criteres
    vecteurs: List[tuple]             # les vecteurs criteres, dans l'ordre
    complet: bool                     # infaisabilite atteinte ?
    motif: str                        # pourquoi on s'est arrete
    ilp: int                          # appels au solveur entier
    secondes: float


def _poids(inst: MOILFP) -> np.ndarray:
    """Objectif de sondage : somme des numerateurs des criteres.

    La CORRECTION n'en depend pas -- n'importe quel point realisable du
    relache ferait l'affaire, et un objectif nul conviendrait. Celui-ci
    tend seulement a tomber pres du front, ce qui raccourcit la chaine de
    reparation et donc le nombre d'appels entiers.
    """
    w = np.zeros(inst.n)
    for z in inst.Z:
        w = w + np.asarray(z.num, dtype=float)
    return w


def enumere(inst: MOILFP,
            ilp_budget: Optional[int] = None,
            time_budget: float = 1e9,
            max_points: Optional[int] = None,
            verbose: bool = False) -> Resultat:
    """Enumere Z(E) par coupes d'efficacite successives."""
    t0 = time.time()
    reset_oracle_counter()
    if ilp_budget is not None:
        set_ilp_budget(ilp_budget)

    model = ECutModel(inst)
    w = _poids(inst)
    points: List[np.ndarray] = []
    vecteurs: List[tuple] = []
    vus = set()

    while True:
        if max_points is not None and len(points) >= max_points:
            return Resultat(points, vecteurs, False, "plafond de points",
                            ORACLE_CALLS["ilp"], time.time() - t0)
        reste = time_budget - (time.time() - t0)
        if reste <= 0:
            return Resultat(points, vecteurs, False, "budget de temps",
                            ORACLE_CALLS["ilp"], time.time() - t0)

        res = model.optimize(w, 0.0, maximize=True, time_limit=reste)

        if res.status == "infeasible":
            # LE CERTIFICAT. Plus aucun x ne survit aux coupes, donc plus
            # aucun vecteur criteres efficace n'est inedit.
            return Resultat(points, vecteurs, True, "relache vide",
                            ORACLE_CALLS["ilp"], time.time() - t0)
        if res.x is None:
            return Resultat(points, vecteurs, False,
                            f"solveur interrompu ({res.status})",
                            ORACLE_CALLS["ilp"], time.time() - t0)

        x = np.rint(np.asarray(res.x[:inst.n])).astype(int)
        a = repair_to_efficient(inst, x,
                                deadline=t0 + time_budget)
        if a is None:
            # La reparation n'a pas abouti dans le budget. Rendre `x` comme
            # s'il etait efficace romprait la seule garantie de ce
            # programme ; on s'arrete plutot.
            return Resultat(points, vecteurs, False,
                            "reparation non terminee dans le budget",
                            ORACLE_CALLS["ilp"], time.time() - t0)

        z = tuple(inst.criteria(a))
        if z in vus:
            # Impossible si les coupes sont exactes : un point de vecteur
            # deja enumere aurait ete retire. Si cela arrive, c'est un
            # defaut de coupe, et le taire ferait boucler le programme
            # en silence.
            return Resultat(points, vecteurs, False,
                            f"vecteur deja enumere : {z} -- coupe suspecte",
                            ORACLE_CALLS["ilp"], time.time() - t0)
        vus.add(z)
        points.append(a)
        vecteurs.append(z)
        if verbose:
            print(f"  {len(points):>4}  Z = {z}", flush=True)
        model.add_efficiency_cut(a)


# ---------------------------------------------------------------------------
# Validation contre l'enumeration par force brute
# ---------------------------------------------------------------------------

def valide(inst: MOILFP, verbose: bool = False) -> Tuple[bool, str]:
    """Compare Z(E) enumere aux vecteurs de E calcules par force brute."""
    from molfp_enum import ground_truth
    gt = ground_truth(inst)
    attendus = {tuple(inst.criteria(x)) for x in gt.E}

    r = enumere(inst, verbose=verbose)
    obtenus = set(r.vecteurs)

    if not r.complet:
        return False, f"enumeration incomplete : {r.motif}"
    manquants = attendus - obtenus
    en_trop = obtenus - attendus
    if manquants or en_trop:
        return False, (f"{len(manquants)} manquant(s), "
                       f"{len(en_trop)} en trop "
                       f"(attendus {len(attendus)}, obtenus {len(obtenus)})")
    return True, f"{len(obtenus)} vecteurs, {r.ilp} appels entiers"


LOT = [dict(n=5, m=3, p=2, seed=1, rhs_scale=1.2, corr=0.00),
       dict(n=5, m=3, p=3, seed=2, rhs_scale=1.5, corr=0.00),
       dict(n=6, m=4, p=2, seed=1, rhs_scale=1.5, corr=0.25),
       dict(n=6, m=4, p=3, seed=1, rhs_scale=1.2, corr=0.00),
       dict(n=6, m=4, p=3, seed=3, rhs_scale=1.5, corr=0.50),
       dict(n=7, m=4, p=2, seed=1, rhs_scale=1.2, corr=0.90),
       dict(n=7, m=4, p=3, seed=1, rhs_scale=1.2, corr=0.25),
       dict(n=8, m=5, p=3, seed=2, rhs_scale=1.0, corr=0.90)]


def main() -> int:
    n = None
    if "--valider" in sys.argv:
        i = sys.argv.index("--valider")
        n = int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else len(LOT)
    lot = LOT[:n] if n else LOT

    print("=" * 84)
    print("ENUMERATION DE Z(E) PAR COUPES D'EFFICACITE, "
          "CONTRE LA FORCE BRUTE")
    print("=" * 84)
    print(f"\n{'instance':<34}{'|Z(E)|':>8}{'appels':>9}{'secondes':>10}"
          f"   verdict")
    print("-" * 84)

    tout_bon = True
    for spec in lot:
        inst = generate(**spec)
        t0 = time.time()
        ok, detail = valide(inst)
        tout_bon &= ok
        etiq = (f"n={spec['n']} m={spec['m']} p={spec['p']} "
                f"s={spec['seed']} c={spec['corr']:.2f}")
        nb = detail.split()[0] if ok else "--"
        print(f"{etiq:<34}{nb:>8}{'':>9}{time.time() - t0:>10.1f}   "
              f"{'OK' if ok else 'ECHEC : ' + detail}", flush=True)

    print("-" * 84)
    print("OK partout" if tout_bon else "AU MOINS UN ECHEC")
    print("\nUn OK dit deux choses a la fois : aucun vecteur efficace n'a")
    print("ete manque, et aucun vecteur non efficace n'a ete rendu. La")
    print("seconde est la plus facile a casser -- il suffit de poser une")
    print("coupe autour d'un point que l'on n'a pas certifie.")
    return 0 if tout_bon else 1


if __name__ == "__main__":
    raise SystemExit(main())
