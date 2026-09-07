"""
molfp_enum.py
=============
VERITE TERRAIN (ground truth) par enumeration exhaustive, en arithmetique
rationnelle EXACTE (fractions.Fraction). Sert uniquement a valider les autres
modules sur de petites instances : ne jamais l'utiliser comme methode de
resolution.

Regle methodologique : aucune heuristique ni methode exacte de ce projet ne
doit etre consideree comme correcte tant qu'elle n'a pas ete confrontee a ce
module sur un lot d'instances aleatoires.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np

from molfp_instance import MOILFP


# ----------------------------------------------------------------------------
# Enumeration de S
# ----------------------------------------------------------------------------

def enumerate_feasible(inst: MOILFP, limit: int = 2_000_000) -> List[np.ndarray]:
    """
    Enumere tous les points entiers de S = {x in Z^n_+ : Ax <= b}.

    ELAGAGE, ET LE BOGUE QU'IL A CACHE. Une premiere version elaguait ainsi :
    des qu'une somme partielle A[:, :j] @ x[:j] depassait b, elle coupait le
    sous-arbre et sortait de la boucle sur la valeur de x_j. Ce raisonnement
    suppose A >= 0 -- les sommes partielles ne peuvent alors que croitre. Le
    generateur garantit cette hypothese (A2), donc rien ne s'est jamais vu
    sur les instances tirees.

    Mais les instances de la LITTERATURE ont des coefficients NEGATIFS. Une
    variable ultERIEURE peut alors faire REDESCENDRE une ligne au-dessous de
    b, et l'elagage retire des points REALISABLES. Sur l'exemple de Drici,
    Ouail & Moulai (2018), il retirait tous les points a x1 = 2, dont
    (2,3,0,0) -- l'un des deux optima publies. La reference elle-meme etait
    donc fausse, precisement sur les instances servant a la validation
    externe.

    Elagage CORRECT. Pour chaque ligne i, on minore la contribution des
    variables encore libres par sum_{k >= j} min(0, A[i,k]) * ub[k]. Ajoutee
    a la somme partielle, elle donne le plus petit membre de gauche encore
    atteignable : si celui-ci depasse deja b, le sous-arbre est reellement
    infaisable. Et l'on ne sort de la boucle sur x_j (`break`) que si la
    colonne est >= 0 ; sinon on passe a la valeur suivante (`continue`), une
    valeur plus grande pouvant redevenir realisable.
    """
    n, m = inst.n, inst.m
    ub = inst.var_upper_bounds()
    A, b = inst.A, inst.b

    # contribution MINIMALE encore atteignable par les variables j..n-1
    neg = np.minimum(A, 0) * ub                       # m x n
    suffix_min = np.zeros((n + 1, m), dtype=int)
    for j in range(n - 1, -1, -1):
        suffix_min[j] = suffix_min[j + 1] + neg[:, j]

    out: List[np.ndarray] = []
    x = np.zeros(n, dtype=int)

    def rec(j: int, partial: np.ndarray) -> None:
        if len(out) > limit:
            raise MemoryError(f"Plus de {limit} points realisables : instance "
                              f"trop grande pour l'enumeration exhaustive.")
        if j == n:
            if np.all(partial <= b):
                out.append(x.copy())
            return
        col = A[:, j]
        montante = bool(np.all(col >= 0))
        for v in range(ub[j] + 1):
            new_partial = partial + v * col
            if np.any(new_partial + suffix_min[j + 1] > b):
                if montante:
                    break          # colonne >= 0 : les v suivants sont pires
                continue           # sinon un v plus grand peut redevenir bon
            x[j] = v
            rec(j + 1, new_partial)
        x[j] = 0

    rec(0, np.zeros(m, dtype=int))
    return out


# ----------------------------------------------------------------------------
# Filtrage de Pareto exact
# ----------------------------------------------------------------------------

def dominates(u: Tuple[Fraction, ...], v: Tuple[Fraction, ...]) -> bool:
    """u domine v (maximisation) : u >= v composante par composante, u != v."""
    ge = all(ui >= vi for ui, vi in zip(u, v))
    return ge and u != v


def pareto_filter(vectors: List[Tuple[Fraction, ...]]) -> List[int]:
    """Indices des vecteurs non domines. Comparaison exacte (Fractions)."""
    uniq = sorted(set(vectors), reverse=True)     # tri lexicographique decroissant
    nd_set = []
    for v in uniq:
        # un vecteur ne peut etre domine que par un vecteur lexico-superieur,
        # donc deja traite et conserve dans nd_set
        if not any(dominates(u, v) for u in nd_set):
            nd_set.append(v)
    nd = set(nd_set)
    return [i for i, v in enumerate(vectors) if v in nd]


# ----------------------------------------------------------------------------
# Verite terrain complete
# ----------------------------------------------------------------------------

@dataclass
class GroundTruth:
    S: List[np.ndarray]                       # tous les points realisables
    E: List[np.ndarray]                       # ensemble efficace exact
    ZE: List[Tuple[Fraction, ...]]            # vecteurs criteres des points de E
    q_star: Fraction                          # max f sur E   <-- reference de (P)
    x_star: np.ndarray                        # un argmax
    q_max_S: Fraction                         # max f sur S   <-- borne sup valide
    ideal: Tuple[Fraction, ...]
    nadir: Tuple[Fraction, ...]               # vrai nadir (calcule sur E)

    def summary(self) -> str:
        return (f"|S| = {len(self.S):>7}   |E| = {len(self.E):>6}   "
                f"|E|/|S| = {len(self.E)/max(1,len(self.S)):.4f}\n"
                f"q*      = {float(self.q_star):.6f}  (max f sur E)\n"
                f"max_S f = {float(self.q_max_S):.6f}  (borne sup, relaxation)\n"
                f"ecart relaxation = "
                f"{float((self.q_max_S - self.q_star)/abs(self.q_max_S))*100:.2f} %")


def ground_truth(inst: MOILFP, limit: int = 2_000_000) -> GroundTruth:
    """Calcule S, E, q* et les points ideal/nadir par force brute exacte."""
    S = enumerate_feasible(inst, limit=limit)
    Zvals = [inst.criteria(x) for x in S]

    idx_E = pareto_filter(Zvals)
    E = [S[i] for i in idx_E]
    ZE = [Zvals[i] for i in idx_E]

    fS = [inst.f.value(x) for x in S]
    q_max_S = max(fS)

    fE = [inst.f.value(x) for x in E]
    j = int(np.argmax([float(v) for v in fE]))
    # argmax exact (evite les egalites mal tranchees en flottant)
    best = max(fE)
    j = fE.index(best)

    p = inst.p
    ideal = tuple(max(z[k] for z in Zvals) for k in range(p))
    nadir = tuple(min(z[k] for z in ZE) for k in range(p))

    return GroundTruth(S=S, E=E, ZE=ZE, q_star=best, x_star=E[j],
                       q_max_S=q_max_S, ideal=ideal, nadir=nadir)


# ----------------------------------------------------------------------------
# Utilitaire de comparaison
# ----------------------------------------------------------------------------

def as_key(x: np.ndarray) -> Tuple[int, ...]:
    return tuple(int(v) for v in x)


def compare_sets(found: List[np.ndarray],
                 truth: List[np.ndarray]) -> Dict[str, object]:
    """Compare un ensemble de solutions trouvees a la verite terrain."""
    F = {as_key(x) for x in found}
    T = {as_key(x) for x in truth}
    return {
        "n_found": len(F),
        "n_truth": len(T),
        "faux_positifs": len(F - T),     # trouves mais NON efficaces  -> grave
        "manquants": len(T - F),         # efficaces non trouves       -> couverture
        "exact": F == T,
    }
