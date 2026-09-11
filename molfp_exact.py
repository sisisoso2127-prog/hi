"""
molfp_exact.py
==============
LA METHODE EXACTE DE REFERENCE, isolee.

Ce fichier ne contient qu'une chose : `solve_P`, l'hybride exact-exact --
Dinkelbach a parametre rationnel en boucle externe, oracle lineaire sur E en
boucle interne. C'est la methode a laquelle le memoire compare la sienne,
et non la sienne.

Elle vivait jusqu'ici dans molfp_oracle.py, c'est-a-dire dans le fichier qui
porte aussi l'oracle et le modele de coupes dont la METHODE HYBRIDE depend.
Un lecteur y trouvait donc deux methodes concurrentes melees a leur
infrastructure commune. Les trois sont maintenant separees :

    molfp_oracle.py     infrastructure PARTAGEE (ECutModel, oracle,
                        chaine de reparation)
    molfp_exact.py      la methode exacte de reference          <- ce fichier
    molfp_hybride.py    la matheuristique certifiee, contribution du memoire

Le code est deplace VERBATIM : aucune ligne n'est modifiee, de sorte que le
decoupage ne peut rien changer aux chiffres. C'est verifie et non suppose
(voir le message de commit).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Tuple

import numpy as np

from molfp_core import ORACLE_CALLS
from molfp_instance import MOILFP
from molfp_oracle import ECutModel, max_linear_over_E

# ----------------------------------------------------------------------------
# Hybride exact-exact : Dinkelbach + oracle lineaire sur E
# ----------------------------------------------------------------------------

@dataclass
class HybridResult:
    status: str
    q_star: Optional[Fraction]
    x_star: Optional[np.ndarray]
    outer_iterations: int          # iterations de Dinkelbach
    total_cuts: int
    ilp_calls: int
    time: float
    trace: List[Tuple[Fraction, float]] = field(default_factory=list)
    archive: List[np.ndarray] = field(default_factory=list)


def solve_P(inst: MOILFP,
            max_outer: int = 60,
            time_limit: float = 600.0,
            reuse_cuts: bool = True,
            model: Optional[ECutModel] = None,
            x0: Optional[np.ndarray] = None,
            verbose: bool = False) -> HybridResult:
    """
    Resout  (P)  max f(x) s.c. x in E  par l'hybride exact-exact.

    Boucle externe : Dinkelbach a parametre rationnel (Th. 3).
    Boucle interne : max_linear_over_E (oracle exact ci-dessus).

    reuse_cuts : les coupes de dominance ne dependent que de E, pas de la
    fonction objectif. Les conserver d'une iteration Dinkelbach a l'autre
    evite de reconstruire la relaxation a chaque fois -- c'est le point
    d'hybridation qui rend l'ensemble economique.

    AMORCAGE A CHAUD. `x0` et `model` permettent de demarrer ailleurs qu'au
    point efficace arbitraire obtenu depuis 0 :

      * `x0` doit etre un point EFFICACE CERTIFIE. Le Th. 3 ne suppose rien
        sur le point de depart sinon qu'il appartienne a l'ensemble sur
        lequel on optimise ; partir de x0 dans E donne q = f(x0) <= q*, et
        l'iteration reste croissante. Un x0 proche de l'optimum epargne des
        iterations externes, dont chacune coute un appel complet a l'oracle.
      * `model` peut arriver deja garni de coupes de dominance. Elles restent
        valides quel que soit l'objectif (Th. 4 ne depend que de E), donc des
        coupes produites par une autre methode sont directement reutilisables.

    C'est ce qui permet de brancher une metaheuristique en amont : voir
    `solve_P_warm`.
    """
    t0 = time.time()
    calls0 = ORACLE_CALLS["ilp"]
    f = inst.f
    if model is not None:
        R = model
    else:
        R = ECutModel(inst) if reuse_cuts else None

    if x0 is not None:
        x_cur = np.asarray(x0, dtype=int)
        archive = [x_cur]
    else:
        # amorcage : un point efficace quelconque
        first = max_linear_over_E(inst, np.zeros(inst.n), 0.0,
                                  model=R, time_limit=time_limit)
        if first.x_star is None and not first.incumbents:
            return HybridResult("empty", None, None, 0, 0,
                                ORACLE_CALLS["ilp"] - calls0, time.time() - t0)
        x_cur = first.x_star if first.x_star is not None else first.incumbents[0]
        archive = list(first.incumbents)

    q = f.value(x_cur)
    trace: List[Tuple[Fraction, float]] = []

    for it in range(1, max_outer + 1):
        if time.time() - t0 > time_limit:
            return HybridResult("limit", q, x_cur, it,
                                R.n_cuts if R else 0,
                                ORACLE_CALLS["ilp"] - calls0,
                                time.time() - t0, trace, archive)

        P_, Q_ = q.numerator, q.denominator
        coef = (Q_ * f.num - P_ * f.den).astype(float)
        const = float(Q_ * f.a - P_ * f.b)

        r = max_linear_over_E(inst, coef, const, model=R,
                              time_limit=time_limit - (time.time() - t0))
        archive.extend(r.incumbents)

        if r.x_star is None:
            return HybridResult("empty", q, x_cur, it,
                                R.n_cuts if R else 0,
                                ORACLE_CALLS["ilp"] - calls0,
                                time.time() - t0, trace, archive)

        Fq = r.value                      # borne INFERIEURE de F(q) en general
        trace.append((q, Fq))
        if verbose:
            print(f" [Dinkelbach {it}] q = {float(q):.6f}   F(q) = {Fq:.4f}   "
                  f"coupes = {r.n_cuts}   [{r.status}]")

        # ------------------------------------------------------------------
        # CORRECTION DE SURETE.
        # r.value n'est la valeur exacte de F(q) que si l'oracle a PROUVE
        # l'optimalite. Sinon ce n'est qu'une borne inferieure, et le test
        # "F(q) <= 0" ne prouve rien : c'est ainsi que l'on concluait a tort
        # a l'optimalite quand l'oracle atteignait sa limite de temps.
        #
        #   Fq > 0  : valide quel que soit le statut, car F(q) >= Fq > 0
        #             donc la racine n'est pas atteinte -> on continue.
        #   Fq <= 0 : conclusion valide UNIQUEMENT si r.status == 'optimal'.
        # ------------------------------------------------------------------
        if Fq > 1e-9:
            x_cur = r.x_star
            q = f.value(x_cur)
            continue

        if r.status == "optimal":         # racine atteinte et prouvee
            return HybridResult("optimal", q, x_cur, it,
                                R.n_cuts if R else 0,
                                ORACLE_CALLS["ilp"] - calls0,
                                time.time() - t0, trace, archive)

        # borne non prouvee : on ne peut rien conclure
        return HybridResult("limit", q, x_cur, it,
                            R.n_cuts if R else 0,
                            ORACLE_CALLS["ilp"] - calls0,
                            time.time() - t0, trace, archive)

    return HybridResult("limit", q, x_cur, max_outer,
                        R.n_cuts if R else 0,
                        ORACLE_CALLS["ilp"] - calls0, time.time() - t0,
                        trace, archive)
