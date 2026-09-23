#!/usr/bin/env python3
"""
verify_lineaire.py
==================
LE CAS LINEAIRE EST UN CAS PARTICULIER, ET CELA SE VERIFIE.

La question posee etait : peut-on reprendre le code MATLAB de Sylva & Crema
-- ecrit pour un MOILP a criteres LINEAIRES -- et le convertir vers notre
cadre FRACTIONNAIRE ? La reponse est que la conversion est deja faite, et
qu'elle ne portait que sur un seul point.

CE QUI CHANGE, ET CE QUI NE CHANGE PAS. Toute la mecanique de Sylva & Crema
-- resoudre, certifier, exclure { x : Z(x) <= Z(a) }, recommencer jusqu'a
infaisabilite -- se transporte mot pour mot. Un seul element ne se
transporte pas : la coupe elle-meme. Avec des criteres lineaires,
Z_k(x) >= Z_k(a) + 1 est deja une ligne entiere. Avec des criteres
fractionnaires, Z_k(x) = (c_k x + a_k)/(d_k x + b_k) n'est pas lineaire, et
il faut le PRODUIT CROISE :

    e^a_k(x) = D(a) N_k(x) - N_k(a) D(x) >= 1

qui redevient une ligne entiere parce que N_k(a) et D(a) sont des
constantes. C'est cette substitution, et elle seule, qui separe les deux
cadres.

CE QUE CE FICHIER MONTRE. L'instance codee en dur dans `sylva_crema.m` --
la matrice C et la matrice A de ses lignes 8 a 10 -- reconstruite dans
notre cadre avec des denominateurs identiquement egaux a 1, puis enumeree
par `molfp_enumere`. Si notre code rend exactement l'ensemble que rend la
force brute sur cette instance, alors il traite bien le cas lineaire comme
un cas particulier, et il n'y a rien a porter depuis MATLAB.

CE QUE CE FICHIER NE MONTRE PAS. Que le code MATLAB est juste. Il n'a pas
ete execute -- `bintprog` a ete retire de MATLAB en R2016a -- et sa lecture
a releve cinq defauts bloquants. La comparaison reste a faire le jour ou il
tournera ; elle vaudrait, parce que les deux chemins ne partagent ni
langage, ni solveur, ni structure.
"""

from __future__ import annotations

import sys

import numpy as np

from molfp_instance import MOILFP, generate
from molfp_enum import ground_truth
from molfp_enumere import enumere

# sylva_crema.m, lignes 8 a 10, telles quelles
C_MATLAB = np.array([[3, 8, 1, 5, 7, 4],
                     [7, 5, 8, 7, 0, 7]])
A_MATLAB = np.array([[11, 1, 16, 13, 8, 4],
                     [6, 13, 0, 13, 1, 9]])
B_MATLAB = np.array([205, 136])


def instance_lineaire() -> MOILFP:
    """L'instance MATLAB dans notre cadre : denominateurs constants a 1."""
    n = C_MATLAB.shape[1]
    gabarit = generate(n=n, m=2, p=2, seed=1, rhs_scale=1.0, corr=0.0)
    Critere = gabarit.Z[0].__class__
    zero = np.zeros(n, dtype=int)
    Z = [Critere(num=C_MATLAB[k], den=zero, a=0, b=1)
         for k in range(C_MATLAB.shape[0])]
    return MOILFP(A=A_MATLAB, b=B_MATLAB, Z=Z,
                  f=Critere(num=C_MATLAB[0], den=zero, a=0, b=1),
                  name="sylva_crema_matlab", ub=gabarit.ub)


def main() -> int:
    inst = instance_lineaire()
    den = [int(z.denominator(np.zeros(inst.n, dtype=int))) for z in inst.Z]
    print("=" * 76)
    print("L'INSTANCE DE sylva_crema.m, ENUMEREE PAR NOTRE CADRE")
    print("=" * 76)
    print(f"\n  denominateurs des criteres : {den}  "
          f"(tous a 1 : criteres lineaires)")

    r = enumere(inst)
    attendus = {tuple(inst.criteria(x)) for x in ground_truth(inst).E}
    obtenus = set(r.vecteurs)

    print(f"  coupes d'efficacite : {len(r.vecteurs)} vecteurs, "
          f"complet = {r.complet}, {r.ilp} appels entiers")
    print(f"  force brute         : {len(attendus)} vecteurs")
    manque, trop = attendus - obtenus, obtenus - attendus
    print(f"  manquants : {len(manque)}   en trop : {len(trop)}")

    ok = r.complet and not manque and not trop
    print("\n" + ("OK -- le cas lineaire est traite comme un cas "
                  "particulier du cadre fractionnaire."
                  if ok else "ECHEC"))
    if ok:
        print("Il n'y a donc rien a porter depuis MATLAB : la seule piece")
        print("qui distingue les deux cadres est la coupe, et elle est")
        print("ecrite. Ce que le code MATLAB peut encore apporter est une")
        print("VERIFICATION CROISEE, pas une fonctionnalite.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
