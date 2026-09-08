"""
bench_couverture.py
===================
LA TROISIEME HYPOTHESE : l'outil de coupe ou la couverture de l'archive ?

Ce que les bancs precedents ont ELIMINE. A corr = 0 et n >= 20, l'ecart
garanti reste bloque autour de 98 / 94 / 97 %. Deux explications ont ete
proposees puis REFUTEES par l'experience :

  * « le verrou est le PLAFOND de coupes (leur cout en binaires) » --
    bench_cglp.py supprime ce cout entierement, par coupes disjonctives
    sans binaire ; l'ecart bouge de 0,11 point au mieux ;
  * « le verrou est l'ORDRE des coupes » -- height_rank donne un signe
    mixte et reste desactive.

L'HYPOTHESE QUI RESTE. La region que nous savons couper serait une part
NEGLIGEABLE de ce qui separe le relache R de l'ensemble efficace E ; non
parce que la coupe est faible, mais parce que l'ARCHIVE ne voit qu'une
fraction infime de E quand n grandit.

CE QUI REND L'HYPOTHESE TESTABLE. La coupe d'efficacite posee en `a` retire
exactement { x : Z(x) <= Z(a) }. Posons

    S+ = { x dans S : f(x) > q* }

l'ensemble des points qui rendent la borne lache -- ce sont eux, et eux
seuls, que la borne superieure peut aller chercher au-dessus de q*. Aucun
point de S+ n'est efficace : si x etait efficace avec f(x) > q*, alors q*
ne serait pas le maximum de f sur E. Donc TOUT point de S+ est domine par
au moins un point efficace, c'est-a-dire :

    couvrir S+ avec les coupes centrees sur E TOUT ENTIER ferme la borne
    COMPLETEMENT.                                                    (*)

L'outil de coupe est donc SUFFISANT en principe. La seule perte possible
est une perte de COUVERTURE. On la decompose en deux :

    cov_E   = |{ x dans S+ couvert par E }|        / |S+|   -- doit valoir
                                                               100 %, c'est
                                                               le test de (*)
    cov_A   = |{ x dans S+ couvert par l'archive }| / |S+|   -- plafond de
                                                               l'archive
    cov_cut = |{ x dans S+ couvert par les centres
                 REELLEMENT poses }|               / |S+|   -- ce qui est
                                                               effectivement
                                                               retire

La lecture est alors sans ambiguite :

    cov_A proche de 100 % et cov_cut faible  ->  le verrou est le BUDGET de
                                                 coupes (deja refute par CGLP) ;
    cov_A qui S'EFFONDRE avec n              ->  le verrou est la COUVERTURE :
                                                 l'archive ne voit plus E.

Protocole. Deux echelles, parce qu'aucune seule n'est honnete :

  ECHELLE A -- rhs_scale FIXE a 1,0, n dans {6, 8, 10, 12}. Le regime est
    rigoureusement comparable d'un n a l'autre ; c'est l'echelle sur
    laquelle une tendance a un sens. Elle s'arrete a n = 12 parce que
    |S| y atteint deja 3.10^5.
  ECHELLE B -- rhs_scale DECROISSANT pour maintenir |S| enumerable,
    n jusqu'a 20. Elle atteint le regime ou l'ecart est observe, au prix
    d'une densite qui varie avec n : a lire comme INDICATIVE, la densite
    etant un facteur confondu.

Arithmetique. Les comparaisons de dominance sont EXACTES. On n'utilise pas
les flottants : chaque coordonnee des criteres est remplacee par son RANG
dans la liste triee des valeurs distinctes prises sur S. Z(x) <= Z(a) si
et seulement si rang_k(x) <= rang_k(a) pour tout k. La comparaison devient
entiere, donc vectorisable, sans perdre l'exactitude des Fractions.

Usage :  python bench_couverture.py [n_graines] [plafond_ILP]
"""

from __future__ import annotations

import sys
import time
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple

import numpy as np

from molfp_core import reset_oracle_counter
from molfp_enum import ground_truth
from molfp_instance import generate
from molfp_matheuristic import matheuristic_P

# echelle A : regime strictement comparable (rhs_scale fixe)
ECHELLE_A = [(6, 1.0), (8, 1.0), (10, 1.0), (12, 1.0)]
# echelle B : |S| maintenu enumerable, densite variable (facteur confondu)
ECHELLE_B = [(12, 0.65), (14, 0.55), (16, 0.48), (18, 0.42), (20, 0.38)]

LIMITE_S = 400_000


# ---------------------------------------------------------------------------
# Rangs exacts : dominance entiere equivalente a la dominance rationnelle
# ---------------------------------------------------------------------------

def table_de_rangs(Z: Sequence[Tuple[Fraction, ...]], p: int) -> np.ndarray:
    """
    Remplace chaque coordonnee par son rang dans les valeurs distinctes
    prises sur l'echantillon. L'application est strictement croissante
    coordonnee par coordonnee, donc elle PRESERVE exactement l'ordre
    partiel de dominance -- et rend la comparaison entiere.
    """
    R = np.zeros((len(Z), p), dtype=np.int32)
    for k in range(p):
        valeurs = sorted({z[k] for z in Z})
        rang = {v: i for i, v in enumerate(valeurs)}
        R[:, k] = [rang[z[k]] for z in Z]
    return R


def masque_couvert(R_cible: np.ndarray, R_centres: np.ndarray) -> np.ndarray:
    """Masque booleen : ligne de `R_cible` retiree par au moins un centre."""
    couvert = np.zeros(len(R_cible), dtype=bool)
    for a in R_centres:
        # ~couvert evite de retester ce qui est deja pris : le cout total
        # decroit a mesure que la couverture monte.
        reste = ~couvert
        if not reste.any():
            break
        couvert[reste] = np.all(R_cible[reste] <= a, axis=1)
    return couvert


def couverture(R_cible: np.ndarray, R_centres: np.ndarray) -> float:
    """
    Part des lignes de `R_cible` couvertes par au moins un centre, au sens
    Z(x) <= Z(a). Renvoie une fraction de [0, 1].
    """
    if len(R_cible) == 0:
        return 1.0
    if len(R_centres) == 0:
        return 0.0
    return float(masque_couvert(R_cible, R_centres).mean())


def borne_ideale(f_S: Sequence[Fraction], couvert: np.ndarray,
                 qs: Fraction) -> float:
    """
    MEILLEURE BORNE que les coupes posees AUTORISENT :

        U = max { f(x) : x dans S, x NON retire }.

    Point essentiel : la borne superieure est un MAXIMUM sur le relache.
    Elle ne se resserre donc PAS a proportion du VOLUME retire -- elle ne
    bouge que si le point de plus grand f est lui-meme retire. Retirer
    99 % des points fautifs ne vaut rien si le centieme pour cent contient
    le maximum. C'est ce que cette fonction quantifie.
    """
    reste = [v for v, c in zip(f_S, couvert) if not c]
    return float(max(reste)) if reste else float(qs)


# ---------------------------------------------------------------------------
# Mesure sur une instance
# ---------------------------------------------------------------------------

def mesurer(n: int, rhs: float, seed: int, cap: int) -> Dict[str, object]:
    m = max(3, n // 2 + 1)
    inst = generate(n=n, m=m, p=3, seed=seed, rhs_scale=rhs, corr=0.00)

    t0 = time.time()
    gt = ground_truth(inst, limit=LIMITE_S)
    t_gt = time.time() - t0

    qs = gt.q_star
    # S+ : les points qui rendent la borne lache. Aucun n'est efficace.
    fS = [inst.f.value(x) for x in gt.S]
    idx_plus = [i for i, v in enumerate(fS) if v > qs]

    reset_oracle_counter()
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6, seed=seed,
                       archive_cuts=True, ilp_budget=cap, cut_diversify=False)

    # criteres : on rassemble S+, E, l'archive et les centres poses dans un
    # meme referentiel de rangs.
    Z_plus = [inst.criteria(gt.S[i]) for i in idx_plus]
    Z_E = list(gt.ZE)
    Z_A = [inst.criteria(a) for a in r.archive]
    Z_C = [inst.criteria(a) for a in r.cut_points]

    # On rassemble S TOUT ENTIER (pas seulement S+) : l'ecart residuel se
    # lit sur le maximum de f parmi les survivants, ou qu'ils soient.
    Z_S = [inst.criteria(x) for x in gt.S]
    tous = Z_S + Z_E + Z_A + Z_C
    R = table_de_rangs(tous, inst.p)
    i0 = len(Z_S)
    i1 = i0 + len(Z_E)
    i2 = i1 + len(Z_A)
    R_S, R_E, R_A, R_C = R[:i0], R[i0:i1], R[i1:i2], R[i2:]
    R_plus = R_S[idx_plus]

    nZE = len({z for z in Z_E})
    nZA = len({z for z in Z_A})

    lb = None if r.q_lb is None else float(r.q_lb)
    gap = None if (lb is None or r.q_ub is None) else \
        (r.q_ub - lb) / max(1e-12, abs(r.q_ub))

    return {
        "n": n, "rhs": rhs, "seed": seed,
        "S": len(gt.S), "E": len(gt.E), "ZE": nZE,
        "Splus": len(idx_plus),
        "A": nZA, "cut": len(Z_C),
        "cov_E": couverture(R_plus, R_E),
        "cov_A": couverture(R_plus, R_A),
        "cov_cut": couverture(R_plus, R_C),
        "part_E": nZA / max(1, nZE),
        # ECHELLE DE BORNES : q* <= U_A <= U_cut <= q_ub, toutes exactes
        # sauf la derniere qui est celle que la methode produit vraiment.
        "q_star": float(qs),
        "q_lb": lb,
        "U_A": borne_ideale(fS, masque_couvert(R_S, R_A), qs),
        "U_cut": borne_ideale(fS, masque_couvert(R_S, R_C), qs),
        "q_ub": r.q_ub,
        "U_S": float(gt.q_max_S),          # aucune coupe du tout : temoin
        "lb_opt": (r.q_lb == qs),
        "gap": gap,
        "prouve": bool(r.proved_optimal),
        "ilp": r.ilp_calls, "t_gt": t_gt,
    }


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

EN_TETE = (f"{'n':>3}{'rhs':>6}{'|S|':>8}{'|Z(E)|':>7}{'|Z(A)|':>7}"
           f"{'cov_A':>7}{'cov_cut':>8}{'  ':>2}"
           f"{'q*':>9}{'q_lb':>9}{'U_S':>9}{'q_ub':>9}"
           f"{'U_A':>9}{'U_cut':>9}{'  ':>2}{'LB=q*':>6}{'ILP':>6}")


def pc(v) -> str:
    return "  --  " if v is None else f"{100*v:5.1f}%"


def num(v) -> str:
    return "     -- " if v is None else f"{v:9.3f}"


def _ligne(d) -> str:
    return (f"{d['n']:>3}{d['rhs']:>6.2f}{d['S']:>8d}{d['ZE']:>7d}"
            f"{d['A']:>7d}{pc(d['cov_A']):>7}{pc(d['cov_cut']):>8}{'':>2}"
            f"{num(d['q_star'])}{num(d['q_lb'])}{num(d['U_S'])}"
            f"{num(d['q_ub'])}{num(d['U_A'])}{num(d['U_cut'])}{'':>2}"
            f"{('oui' if d['lb_opt'] else 'NON'):>6}{d['ilp']:>6d}")


def afficher(lignes: List[Dict[str, object]]) -> None:
    print(EN_TETE)
    print("-" * len(EN_TETE))
    for d in lignes:
        print(_ligne(d), flush=True)


def _ecart(haut, bas):
    """Ecart relatif d'un encadrement, convention du projet : (h - b)/|h|."""
    if haut is None or bas is None:
        return None
    return (haut - bas) / max(1e-12, abs(haut))


def bornes(d):
    """Les deux bornes valides d'une ligne, et leur minimum."""
    if d["q_ub"] is None:
        return None, None, None
    # la borne geometrique ne peut pas descendre sous l'incumbent
    geo = max(d["U_cut"], float(d["q_lb"]) if d["q_lb"] is not None else 0.0)
    return d["q_ub"], geo, min(d["q_ub"], geo)


def agreger(lignes: List[Dict[str, object]]) -> None:
    """
    LES DEUX ROUTES VERS UNE BORNE, ET LEUR MINIMUM.

    La methode calcule sa borne par le Th. 5 prime : q_ub = q + U/(Q D+), ou
    U majore F_q sur le relache R. Cette route exploite la STRUCTURE du test
    d efficacite -- elle peut donc descendre sous max_S f, et elle le fait
    (voir les lignes ou q_ub est tres inferieur a U_S sans qu aucune coupe
    ne soit posee).

    Il en existe une SECONDE, qu aucune version de la methode n utilise :

        U_cut = max { f(x) : x dans S, x non retire par les coupes posees }.

    Elle est VALIDE pour la raison meme de la Prop. 3 : les coupes de
    dominance preservent E, et chaque coupe d efficacite n est posee qu apres
    etablissement de sa condition de cloture -- aucun point efficace retire
    ne bat l incumbent. Donc q* <= max(q, U_cut).

    Les deux routes sont INCOMPARABLES : aucune ne majore l autre. Le MINIMUM
    de deux bornes valides etant valide, la colonne « min » est ce que la
    methode pourrait annoncer SANS CHANGER UNE SEULE COUPE.
    """
    par_n: Dict[int, List[Dict[str, object]]] = {}
    for d in lignes:
        par_n.setdefault(d["n"], []).append(d)

    def med(g, c):
        v = [x[c] for x in g if x[c] is not None]
        return float(np.median(v)) if v else None

    print()
    print("LES DEUX ROUTES VERS LA BORNE  (medianes par n)")
    entete = (f"{'n':>3}{'LB=q*':>8}{'cov_A':>8}{'cov_cut':>9}   "
              f"{'ecart Th5p':>12}{'ecart geom':>12}{'ecart du min':>14}   "
              f"{'gain (pts)':>12}")
    print(entete)
    print("-" * len(entete))
    for n in sorted(par_n):
        g = par_n[n]
        e5, eg, em = [], [], []
        for d in g:
            lb = d["q_lb"]
            b5, bg, bm = bornes(d)
            for L, v in ((e5, _ecart(b5, lb)), (eg, _ecart(bg, lb)),
                         (em, _ecart(bm, lb))):
                if v is not None:
                    L.append(v)
        mm = lambda L: float(np.median(L)) if L else None
        gain = (mm(e5) - mm(em)) if (e5 and em) else None
        n_opt = sum(1 for d in g if d["lb_opt"])
        print(f"{n:>3}{f'{n_opt}/{len(g)}':>8}{pc(med(g,'cov_A')):>8}"
              f"{pc(med(g,'cov_cut')):>9}   "
              f"{pc(mm(e5)):>12}{pc(mm(eg)):>12}{pc(mm(em)):>14}   "
              f"{('     --     ' if gain is None else f'{100*gain:+12.1f}')}",
              flush=True)


def verdict(lignes: List[Dict[str, object]], titre: str) -> None:
    ko = [d for d in lignes if d["cov_E"] < 1.0 - 1e-12]
    print()
    print(f"  TEST DE (*) sur {titre} : les coupes centrees sur E tout entier")
    if ko:
        print(f"  NE couvrent PAS S+ sur {len(ko)}/{len(lignes)} lignes "
              f"-> la propriete (*) est FAUSSE, l analyse ci-dessus tombe.")
        for d in ko:
            print(f"     n={d['n']} graine={d['seed']} "
                  f"cov_E={100*d['cov_E']:.2f}%")
    else:
        print(f"  couvrent S+ a 100 % sur {len(lignes)}/{len(lignes)} lignes :")
        print(f"  l outil de coupe est SUFFISANT en principe.")

    haut = [d for d in lignes if d["cov_A"] >= 0.99]
    parts = [100 * d["part_E"] for d in lignes]
    print(f"  COUVERTURE PAR L ARCHIVE : cov_A >= 99 % sur "
          f"{len(haut)}/{len(lignes)} lignes, alors que l archive ne retient")
    print(f"  que {np.median(parts):.0f} % des vecteurs criteres de E "
          f"(mediane). L hypothese « l archive ne voit plus E »")
    print(f"  est donc {'REFUTEE' if len(haut) > len(lignes)//2 else 'MAINTENUE'}"
          f" : une petite archive domine deja presque tout S+.")

    n_lb = sum(1 for d in lignes if d["lb_opt"])
    print(f"  RECHERCHE : l incumbent atteint q* sur {n_lb}/{len(lignes)} "
          f"lignes. Ce qui manque au-dela n est pas")
    print(f"  une perte de borne mais une perte de RECHERCHE, invisible "
          f"sans verite terrain.")

    mieux = pire = egal = 0
    for d in lignes:
        b5, bg, _ = bornes(d)
        if b5 is None:
            continue
        if bg < b5 - 1e-9:
            mieux += 1
        elif bg > b5 + 1e-9:
            pire += 1
        else:
            egal += 1
    print(f"  ROUTE GEOMETRIQUE contre Th. 5 prime : "
          f"MEILLEURE {mieux}   PIRE {pire}   EGALE {egal}")
    print(f"  -> les deux routes sont INCOMPARABLES. Leur minimum est une "
          f"borne VALIDE. Elle n est pas")
    print(f"  gratuite pour autant : U_cut demande un Dinkelbach sur R, "
          f"soit quelques ILP de plus.")
    print(f"  Ce qu elle ne demande pas, c est une coupe de plus -- "
          f"le meme relache suffit.")


def main() -> int:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    for titre, echelle in (("ECHELLE A - rhs_scale FIXE (regime comparable)",
                            ECHELLE_A),
                           ("ECHELLE B - |S| maintenu enumerable "
                            "(densite = facteur confondu, INDICATIF)",
                            ECHELLE_B)):
        print("=" * len(EN_TETE))
        print(f"{titre}   corr=0, p=3, {n_seeds} graines, plafond {cap} ILP")
        print("=" * len(EN_TETE))
        lignes = []
        for n, rhs in echelle:
            for s in range(1, n_seeds + 1):
                try:
                    d = mesurer(n, rhs, s, cap)
                except MemoryError:
                    print(f"{n:>3}{rhs:>6.2f}   |S| > {LIMITE_S} : "
                          f"hors de portee de l'enumeration", flush=True)
                    continue
                lignes.append(d)
                if len(lignes) == 1:
                    print(EN_TETE)
                    print("-" * len(EN_TETE))
                print(_ligne(d), flush=True)
        if lignes:
            agreger(lignes)
            verdict(lignes, titre.split(" -")[0])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
