"""
literature.py
=============
Temoins issus de la litterature, et positionnement exact de nos resultats
par rapport a elle.

--------------------------------------------------------------------------
TROIS PROBLEMES VOISINS, A NE PAS CONFONDRE
--------------------------------------------------------------------------
La litterature du domaine traite trois problemes distincts, et la
distinction porte sur DEUX axes independants : la nature des criteres du
programme multi-objectif, et la nature de la fonction d'utilite optimisee
sur son ensemble efficace.

                        criteres LINEAIRES        criteres FRACTIONNAIRES
                        (MOILP)                   (MOILFP)
    f lineaire          Jorge (2009),             Zerdani & Moulai (2011)
                        Chergui & Moulai (2008)   -> notre ORACLE interne
    f fractionnaire     Mahdi & Chaabane (2015),  <- NOTRE PROBLEME (P)
                        Drici et al. (2018)

Autrement dit :

  * Zerdani & Moulai (2011) optimisent une fonction LINEAIRE sur l'ensemble
    efficace d'un MOILFP. C'est exactement le sous-probleme que resout notre
    oracle a chaque iteration de Dinkelbach ;
  * Drici, Ouail & Moulai (2018) optimisent une fonction FRACTIONNAIRE sur
    l'ensemble efficace d'un MOILP, donc a criteres LINEAIRES ;
  * notre probleme (P) combine les deux difficultes : f fractionnaire ET
    criteres fractionnaires. Aucun des deux articles ne le couvre.

Consequence directe pour la comparaison : une comparaison frontale avec
Drici et al. n'a de sens que sur l'INTERSECTION des deux classes, c'est-a
dire sur des instances a criteres lineaires. Notre generateur les produit
comme cas particulier en posant d_k = 0 et b_k = 1.

--------------------------------------------------------------------------
TEST D'EFFICACITE D'ECKER & KOUADA (1975)
--------------------------------------------------------------------------
C'est le test employe par Drici et al. (2018), equation (5) :

        max   Theta = somme_i omega_i
        s.c.  C x = I omega + C xbar
              x dans D,  omega >= 0

Comme omega = C x - C xbar, la contrainte omega >= 0 impose
Z_i(x) >= Z_i(xbar) pour tout i, et Theta = somme_i (Z_i(x) - Z_i(xbar)).
Le point xbar est efficace si et seulement si Theta = 0.

RAPPORT AVEC NOTRE THEOREME 2. Notre test pose
e_k(x) = Dbar_k N_k(x) - Nbar_k D_k(x) et theta = somme_k e_k(x). Sur des
criteres LINEAIRES (d_k = 0, b_k = 1) on a Dbar_k = 1 et D_k(x) = 1, donc

        e_k(x) = N_k(x) - Nbar_k = Z_k(x) - Z_k(xbar),

et theta coincide EXACTEMENT avec le Theta d'Ecker & Kouada -- pas seulement
sur le verdict, sur la valeur entiere elle-meme. Notre theoreme 2 est donc
la generalisation stricte de ce test aux criteres fractionnaires, et
`cross_validate` ci-dessous le verifie numeriquement plutot que de le
postuler.

--------------------------------------------------------------------------
CE QUI N'EST PAS ENCORE IMPLEMENTE
--------------------------------------------------------------------------
La methode complete de Drici et al. est un branch and cut sur les relaxations
CONTINUES fractionnaires, et ses coupes dependent du TABLEAU DU SIMPLEXE :

        H_l = { j dans N_l : il existe i, cbar^i_j > 0 }
              union { j dans N_l : cbar^i_j = 0 pour tout i }
        coupe (8) :  somme_{j dans H_l} x_j >= 1

ou N_l est l'ensemble des variables hors base et cbar^i_j les couts reduits
du critere i a la base courante. La reproduire fidelement demande un simplexe
donnant acces au tableau, plus un simplexe fractionnaire (Martos 1975 ;
Cambini & Martein 1992) pour les sous-problemes (LFP)_l. `scipy.optimize`
n'expose ni l'un ni l'autre. C'est un chantier a part entiere, et le faire a
moitie produirait un temoin fautif -- ce que ce module refuse.

Usage :  python literature.py
"""

from __future__ import annotations

import numpy as np

from molfp_core import INF, efficiency_test, feasibility_rows, solve_milp
from molfp_enum import as_key, ground_truth
from molfp_instance import MOILFP, FracObj


# ----------------------------------------------------------------------------
# Instances a criteres lineaires : l'intersection des deux classes
# ----------------------------------------------------------------------------

def has_linear_criteria(inst: MOILFP) -> bool:
    """Les criteres sont-ils lineaires, i.e. l'instance est-elle un MOILP ?"""
    return all(not Zk.den.any() and Zk.b == 1 for Zk in inst.Z)


def generate_linear_criteria(n: int, m: int, p: int, seed: int,
                             coef_max: int = 20,
                             rhs_scale: float = 1.5) -> MOILFP:
    """
    Instance a criteres LINEAIRES et fonction d'utilite fractionnaire : la
    classe exacte traitee par Drici et al. (2018), obtenue comme cas
    particulier de notre structure en posant d_k = 0 et b_k = 1.
    """
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 10, size=(m, n))
    for j in range(n):
        if A[:, j].sum() == 0:
            A[rng.integers(0, m), j] = rng.integers(1, 10)
    b = np.maximum(1, (rhs_scale * A.sum(axis=1)).astype(int))

    Z = [FracObj(num=rng.integers(1, coef_max + 1, size=n),
                 a=int(rng.integers(0, 10)),
                 den=np.zeros(n, dtype=int), b=1)      # critere LINEAIRE
         for _ in range(p)]
    f = FracObj(num=rng.integers(1, coef_max + 1, size=n),
                a=int(rng.integers(0, 10)),
                den=rng.integers(0, 4, size=n),
                b=int(rng.integers(1, 10)))            # utilite FRACTIONNAIRE
    inst = MOILFP(A=A.astype(int), b=b.astype(int), Z=Z, f=f,
                  name=f"lin_n{n}_m{m}_p{p}_s{seed}", seed=seed)
    inst.check_assumptions()
    return inst


# ----------------------------------------------------------------------------
# Ecker & Kouada (1975), tel qu'employe par Drici et al. (2018) eq. (5)
# ----------------------------------------------------------------------------

def ecker_kouada(inst: MOILFP, xbar: np.ndarray) -> tuple:
    """
    Test d'efficacite d'Ecker & Kouada. Exige des criteres LINEAIRES.

    Variables : (x, omega) avec omega dans R^p_+.
    Contraintes : C x - omega = C xbar,  A x <= b,  0 <= x <= ub.
    Objectif   : max somme_i omega_i.

    Renvoie (Theta, efficace, x_dominant).
    """
    if not has_linear_criteria(inst):
        raise ValueError("Ecker & Kouada suppose des criteres lineaires "
                         "(MOILP) ; cette instance est un MOILFP.")
    n, p = inst.n, inst.p
    ub_x = inst.var_upper_bounds()
    nvar = n + p

    obj = np.concatenate([np.zeros(n), np.ones(p)])

    rows = []
    for i in range(inst.m):                       # A x <= b
        r = np.zeros(nvar); r[:n] = inst.A[i]
        rows.append((r, -INF, float(inst.b[i])))
    for k in range(p):                            # C x - omega = C xbar
        r = np.zeros(nvar)
        r[:n] = inst.Z[k].num
        r[n + k] = -1.0
        rhs = float(inst.Z[k].num @ xbar)
        rows.append((r, rhs, rhs))

    lb = np.zeros(nvar)
    # omega est borne par l'amplitude maximale du critere sur la boite
    ub = np.concatenate([ub_x.astype(float),
                         [float(inst.Z[k].num @ ub_x) + 1.0 for k in range(p)]])
    integrality = np.concatenate([np.ones(n), np.zeros(p)])   # omega continu

    res = solve_milp(obj, rows, lb, ub, integrality=integrality,
                     maximize=True, obj_const=0.0)
    if not res.ok:
        raise RuntimeError(f"Ecker & Kouada : statut {res.status}")
    theta = int(round(res.obj))
    return theta, theta == 0, (None if theta == 0 else res.x[:n])


# ----------------------------------------------------------------------------
# Validation croisee : Th. 2 contre Ecker & Kouada
# ----------------------------------------------------------------------------

def cross_validate(inst: MOILFP, sample: int = 40, rng=None) -> dict:
    """
    Sur des criteres lineaires, notre theta (Th. 2) doit valoir EXACTEMENT le
    Theta d'Ecker & Kouada. On compare les valeurs entieres, pas seulement les
    verdicts : c'est un controle bien plus serre.
    """
    rng = rng or np.random.default_rng(0)
    gt = ground_truth(inst, limit=200_000)
    truth = {as_key(x) for x in gt.E}

    pts = list(gt.E)
    if len(gt.S) > len(pts):
        idx = rng.choice(len(gt.S), size=min(sample, len(gt.S)), replace=False)
        pts += [gt.S[i] for i in idx]

    n_theta_diff = n_verdict_diff = n_truth_diff = 0
    for x in pts:
        ours = efficiency_test(inst, x)
        theta_ek, eff_ek, _ = ecker_kouada(inst, x)
        if int(ours.theta) != theta_ek:
            n_theta_diff += 1
        if bool(ours.efficient) != bool(eff_ek):
            n_verdict_diff += 1
        if bool(eff_ek) != (as_key(x) in truth):
            n_truth_diff += 1
    return {"n": len(pts), "E": len(gt.E), "S": len(gt.S),
            "theta_diff": n_theta_diff, "verdict_diff": n_verdict_diff,
            "ek_vs_truth_diff": n_truth_diff}


# ----------------------------------------------------------------------------
# L'exemple publie de Zerdani & Moulai (2011), section 5
# ----------------------------------------------------------------------------

def zerdani_moulai_example() -> MOILFP:
    """
    Instance de la section 5 de Zerdani & Moulai (2011), reproduite a
    l'identique :

        Z1 = (-x1 + 4) / (x2 + 1)
        Z2 = ( x1 - 4) / (-x2 + 3)
        Z3 = -x1 + x2
        S  = { x entier >= 0 : -x1 + 4 x2 <= 0,  x1 - x2/2 <= 4 }
        (P_E) : max  phi = 2 x1 - 3 x2   sur E(P)

    Deux ecarts avec notre generateur, tous deux instructifs.

    `A` comporte des coefficients NEGATIFS, donc (A2) ne tient pas et le
    calcul automatique des bornes n'a plus de sens : on les fournit
    explicitement. De -x1 + 4x2 <= 0 et x1 - x2/2 <= 4 on tire
    4 x2 <= x1 <= 4 + x2/2, donc x2 <= 8/7, soit x2 <= 1 et x1 <= 4.

    `Z2` a un denominateur a coefficient negatif, donc (A1) ne tient pas non
    plus. Mais (A1) n'est qu'une condition SUFFISANTE : les theoremes
    n'exigent que D_k(x) > 0 sur le domaine, ce que `check_assumptions(
    strict=False)` verifie ici par programmation lineaire. C'est exactement
    l'hypothese des auteurs.

    La fonction d'utilite phi etant LINEAIRE, cette instance releve du
    sous-probleme que resout notre oracle, pas de notre probleme (P) complet.
    On l'ecrit donc comme la fraction phi / 1.
    """
    inst = MOILFP(
        A=np.array([[-1, 4], [2, -1]]),          # -x1+4x2 <= 0 ; 2x1-x2 <= 8
        b=np.array([0, 8]),
        Z=[FracObj(np.array([-1, 0]), 4, np.array([0, 1]), 1),
           FracObj(np.array([1, 0]), -4, np.array([0, -1]), 3),
           FracObj(np.array([-1, 1]), 0, np.array([0, 0]), 1)],
        f=FracObj(np.array([2, -3]), 0, np.array([0, 0]), 1),
        name="zerdani_moulai_2011_sec5",
        ub=np.array([4, 1]),
    )
    inst.check_assumptions(strict=False)
    return inst


# resultats PUBLIES, section 5 de l'article
ZM_PUBLISHED_E = {(4, 1), (3, 0), (2, 0), (1, 0), (0, 0)}
ZM_PUBLISHED_XOPT = (3, 0)
ZM_PUBLISHED_PHIOPT = 6


def check_published_example() -> dict:
    """
    Reproduit l'exemple publie et confronte nos resultats aux leurs.

    C'est la seule validation EXTERNE actuellement disponible dans ce projet :
    elle ne compare pas les algorithmes, mais elle confronte nos sorties a un
    resultat publie, sur la classe de probleme exacte de l'article.
    """
    inst = zerdani_moulai_example()
    gt = ground_truth(inst, limit=10_000)
    ours_E = {tuple(int(v) for v in x) for x in gt.E}
    ours_xopt = tuple(int(v) for v in gt.x_star)
    ours_phi = gt.q_star

    from molfp_oracle import solve_P
    r = solve_P(inst, time_limit=60)

    return {
        "S": len(gt.S),
        "E_ours": ours_E,
        "E_published": ZM_PUBLISHED_E,
        "E_match": ours_E == ZM_PUBLISHED_E,
        "xopt_ours": ours_xopt,
        "xopt_match": ours_xopt == ZM_PUBLISHED_XOPT,
        "phi_ours": ours_phi,
        "phi_match": ours_phi == ZM_PUBLISHED_PHIOPT,
        "solve_P_status": r.status,
        "solve_P_value": r.q_star,
        "solve_P_match": (r.status == "optimal"
                          and r.q_star == ZM_PUBLISHED_PHIOPT),
        "solve_P_ilp": r.ilp_calls,
    }


def main() -> int:
    print("=" * 84)
    print("VALIDATION CROISEE - Th. 2 contre le test d'Ecker & Kouada (1975)")
    print("sur des instances a criteres LINEAIRES, classe de Drici et al. (2018)")
    print("=" * 84)
    print(f"{'instance':<22}{'|S|':>7}{'|E|':>6}{'testes':>8}"
          f"{'theta identique':>17}{'verdicts':>10}{'vs verite':>11}")
    print("-" * 84)

    rng = np.random.default_rng(0)
    all_ok = True
    for (n, m, p, s) in [(4, 3, 2, 1), (5, 3, 3, 1), (5, 4, 2, 2),
                         (6, 4, 3, 1), (6, 4, 2, 3), (7, 4, 3, 2)]:
        inst = generate_linear_criteria(n, m, p, s)
        r = cross_validate(inst, rng=rng)
        ok = (r["theta_diff"] == 0 and r["verdict_diff"] == 0
              and r["ek_vs_truth_diff"] == 0)
        all_ok &= ok
        print(f"{inst.name:<22}{r['S']:>7}{r['E']:>6}{r['n']:>8}"
              f"{('oui' if r['theta_diff'] == 0 else 'NON'):>17}"
              f"{('ok' if r['verdict_diff'] == 0 else 'KO'):>10}"
              f"{('ok' if r['ek_vs_truth_diff'] == 0 else 'KO'):>11}")

    print("-" * 84)
    print("RESULTAT : " + ("TOUT VALIDE" if all_ok else "ECHEC"))

    # ------------------------------------------------------------------
    print()
    print("=" * 84)
    print("EXEMPLE PUBLIE - Zerdani & Moulai (2011), section 5")
    print("=" * 84)
    z = check_published_example()
    print(f"|S| = {z['S']}")
    print(f"  E publie   : {sorted(z['E_published'])}")
    print(f"  E calcule  : {sorted(z['E_ours'])}")
    print(f"  -> identique : {'OUI' if z['E_match'] else 'NON'}")
    print(f"  x_opt publie {ZM_PUBLISHED_XOPT}, calcule {z['xopt_ours']}"
          f"   -> {'OUI' if z['xopt_match'] else 'NON'}")
    print(f"  phi_opt publie {ZM_PUBLISHED_PHIOPT}, calcule {z['phi_ours']}"
          f"   -> {'OUI' if z['phi_match'] else 'NON'}")
    print(f"  notre hybride solve_P : statut {z['solve_P_status']}, "
          f"valeur {z['solve_P_value']}, {z['solve_P_ilp']} ILP"
          f"   -> {'OUI' if z['solve_P_match'] else 'NON'}")
    pub_ok = (z["E_match"] and z["xopt_match"] and z["phi_match"]
              and z["solve_P_match"])
    all_ok &= pub_ok
    print()
    print("VALIDATION EXTERNE : " + ("CONFORME AU PUBLIE" if pub_ok
                                     else "DIVERGENCE"))
    print("Cette instance viole (A1) et (A2) -- coefficients negatifs dans A")
    print("et dans un denominateur -- mais non l'hypothese reellement")
    print("necessaire aux theoremes, D_k(x) > 0 sur le domaine, verifiee ici")
    print("par `check_assumptions(strict=False)`.")
    print()
    print("Lecture. Sur criteres lineaires, notre theta ne se contente pas de")
    print("rendre le meme VERDICT que le test d'Ecker & Kouada : il en rend la")
    print("meme VALEUR entiere. Le theoreme 2 est donc la generalisation")
    print("stricte de ce test aux criteres fractionnaires, ou Ecker & Kouada ne")
    print("s'applique plus.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
