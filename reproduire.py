#!/usr/bin/env python3
"""
reproduire.py
=============
UN SEUL FICHIER QUI REJOUE TOUS LES RESULTATS PUBLIES.

POURQUOI CE FICHIER EXISTE. Les bancs de ce depot ne portent pas, dans
leurs valeurs par defaut, les reglages avec lesquels leurs resultats ont
ete publies. Seules les legendes des tableaux les portent. Une campagne
d'archivage menee avec les defauts a produit quatre desaccords, et AUCUN
n'etait un chiffre perime :

  tab:mouvementc  sa legende dit « trois graines » ; le defaut en joue deux
  tab:unifie      elle declare trois plafonds ; le banc n'en joue qu'un
  tab:geom        elle declare le plafond 300 ; le defaut du banc est 900
  tab:geom        et le DECLENCHEUR, que la production desactive, est actif
                  par defaut dans le banc

Chacun de ces ecarts a failli faire declarer perime un tableau parfaitement
juste. L'inverse est tout aussi possible : un tableau reellement perime que
l'on croit confirme parce qu'on l'a rejoue avec les mauvais arguments. Ce
fichier supprime la question en ecrivant les arguments A COTE du tableau
qu'ils produisent.

CE QU'IL FAIT. Pour chaque entree : si le journal existe deja dans
results/, il passe ; sinon il joue le banc avec SES arguments, ecrit le
journal hors du depot, et ne l'y deplace qu'une fois le banc sorti -- en y
inscrivant un marqueur d'incompletude si le code de sortie n'est pas nul.
Un journal partiel qui ne se declare pas partiel sera compare a un tableau
comme s'il etait complet.

CE QU'IL NE FAIT PAS. Comparer. Il produit les journaux ; la comparaison
au texte publie reste un travail de lecture, que `check_sensib.py`
n'automatise que pour un seul tableau sur soixante.

ETAT DE VERIFICATION. La colonne `etat` de chaque entree dit ce qui a ete
etabli, et il faut la lire litteralement :

  verifie   le journal a ete confronte au tableau, cellule par cellule
  declare   les arguments viennent de la legende, la confrontation reste
            a faire
  defaut    ni l'un ni l'autre : le banc tourne avec ses defauts et rien
            n'etablit que le tableau publie en vienne

Usage :  python reproduire.py              tout ce qui manque
         python reproduire.py --liste      n'execute rien, montre l'etat
         python reproduire.py --refaire X  rejoue l'entree X meme archivee
         python reproduire.py --delai 7200 plafond de temps par banc
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, NamedTuple

RACINE = Path(__file__).resolve().parent
JOURNAUX = RACINE / "results"
DELAI_DEFAUT = 10800          # 3 heures par banc


class Entree(NamedTuple):
    nom: str                  # nom du journal, sans .out
    banc: str                 # fichier du banc
    args: List[str]           # arguments EXACTS
    tableaux: str             # ce que ce journal produit
    etat: str                 # verifie | declare | defaut
    note: str = ""


# ---------------------------------------------------------------------------
# LA TABLE. C'est le contenu de ce fichier ; le reste n'est que mecanique.
# ---------------------------------------------------------------------------
ENTREES: List[Entree] = [
    # -- confrontes au texte publie, cellule par cellule --------------------
    Entree("bench_cible_determ", "bench_cible_determ.py", [],
           "tab:cibledeterm", "verifie"),
    Entree("bench_lot_determ", "bench_lot_determ.py", [],
           "tab:lotdeterm", "verifie"),
    Entree("bench_echelle_determ", "bench_echelle_determ.py", [],
           "tab:echelledeterm", "verifie"),
    Entree("bench_litt_determ", "bench_litt_determ.py", [],
           "tab:littdeterm", "verifie"),
    Entree("bench_couverture", "bench_couverture.py", [],
           "tab:couverture", "verifie"),
    Entree("bench_unifie_coupe", "bench_unifie_coupe.py", [],
           "tab:unifiecoupe", "verifie"),
    Entree("bench_budgets", "bench_budgets.py", [],
           "tab:budgets", "verifie"),
    Entree("bench_livrable", "bench_livrable.py", ["AB"],
           "tab:livrableA, tab:livrableB", "verifie"),
    Entree("bench_sensibilite", "bench_sensibilite.py", ["3"],
           "tab:sensib", "verifie",
           "seul tableau couvert par un fascheur : check_sensib.py"),
    Entree("bench_voisinage", "bench_voisinage.py", [],
           "tab:voisinage", "verifie",
           "le tableau etait en conf{0}{1}{1} ; il est passe en "
           "conf{1}{1}{1} avec ce journal"),
    Entree("bench_mouvementc", "bench_mouvementc.py", ["300", "3"],
           "tab:mouvementc", "verifie",
           "TROIS graines : le defaut du banc en joue deux"),
    Entree("bench_unifie", "bench_unifie.py", ["3", "300"],
           "tab:unifie (plafond 300)", "verifie",
           "le tableau declare trois plafonds ; voir les deux entrees "
           "suivantes, dont les journaux se concatenent"),
    Entree("bench_unifie_120", "bench_unifie.py", ["3", "120"],
           "tab:unifie (plafond 120)", "verifie"),
    Entree("bench_unifie_60", "bench_unifie.py", ["3", "60"],
           "tab:unifie (plafond 60)", "verifie"),
    Entree("bench_confirm_lns", "bench_confirm_lns.py", ["3", "300"],
           "remarque « une mediane qui ment »", "verifie",
           "ce journal manquait, et le tableau qu'il porte etait perime "
           "de part en part"),

    # -- arguments lus dans la legende, confrontation a faire ---------------
    Entree("bench_geom", "bench_geom.py", ["300", "3", "0"],
           "tab:geom", "verifie",
           "plafond 300 (le defaut du banc est 900) et DECLENCHEUR "
           "DESACTIVE, comme la production"),
    Entree("bench_geom_declencheur", "bench_geom.py", ["300", "3"],
           "tab:porte", "declare",
           "le meme banc, declencheur ACTIF : c'est la comparaison que "
           "tab:porte publie, et non tab:geom"),
    Entree("bench_stable", "bench_stable.py", ["3"],
           "tab:vivierstable", "verifie",
           "2 h 54 de calcul ; les deux moities concordent"),
    Entree("bench_vivier", "bench_vivier.py", ["300,900"],
           "tab:alternance", "declare",
           "joue en deux morceaux, 300 puis 900 : le second depasse trois "
           "heures et les journaux se concatenent"),
    Entree("molfp_enumere", "molfp_enumere.py", ["--valider", "8"],
           "validation de l'enumeration de Z(E)", "verifie",
           "8/8 contre la force brute"),

    # -- defauts, mapping non etabli ----------------------------------------
    Entree("bench_budget", "bench_budget.py", [],
           "decomposition LB/UB citee en prose", "defaut",
           "+49,6 %, 1,7 point et q_ub ~ 103 confirmes ligne a ligne"),
    Entree("bench_recherche", "bench_recherche.py", [],
           "5/12 lignes ou l'incumbent progresse", "defaut"),
    Entree("bench_filter", "bench_filter.py", [], "-- rien de cite --",
           "defaut"),
    Entree("bench_archive", "bench_archive.py", [], "tab:famine", "defaut"),
    Entree("bench_cglp", "bench_cglp.py", [], "tab:cglp, tab:cglp900",
           "defaut", "le plafond publie est 300, pas celui du defaut"),
    Entree("bench_litterature", "bench_litterature.py", [],
           "tab:match, tab:drici", "defaut"),
    Entree("bench_zm", "bench_zm.py", [], "terrain Zerdani & Moulai",
           "defaut"),
    Entree("bench_scale", "bench_scale.py", [], "lot d'echelle", "defaut"),
    Entree("bench_quality", "bench_quality.py", [], "", "defaut"),
    Entree("bench_compare", "bench_compare.py", [], "", "defaut"),
    Entree("bench_determin", "bench_determin.py", [], "tab:determ",
           "defaut"),
    Entree("bench_improve", "bench_improve.py", [], "", "defaut"),
    Entree("bench_bigm", "bench_bigm.py", [], "", "defaut"),
    Entree("bench_bound", "bench_bound.py", [], "", "defaut"),

    # -- mesures de cette campagne ------------------------------------------
    Entree("bench_temps", "bench_temps.py", ["AB", "2"],
           "tab:temps", "verifie"),
    Entree("bench_confirm_beta", "bench_confirm_beta.py", ["3", "300"],
           "tab:confirmbeta", "verifie"),
    Entree("bench_vivier_taille", "bench_vivier_taille.py", ["3", "300"],
           "tab:vivier", "verifie"),
    Entree("bench_prix_tours", "bench_prix_tours.py", ["3", "300"],
           "remarque « prix des tours »", "verifie"),
    Entree("bench_route_borne", "bench_route_borne.py", ["3", "300"],
           "remarque « route porteuse »", "verifie"),
    Entree("bench_bimodal", "bench_bimodal.py", ["3", "300"],
           "remarque « bimodalite mesuree »", "verifie"),
]

MARQUEUR = """
{barre}
JOURNAL INCOMPLET -- banc interrompu (code {code}) apres {duree} s.
Ce qui precede est PARTIEL ; ce qui manque n'a pas ete joue. Ce marqueur
est dans le FICHIER et non seulement dans un log de lancement : un journal
partiel qui ne se declare pas partiel sera compare a un tableau comme s'il
etait complet.
{barre}
"""


def liste() -> None:
    larg = max(len(e.nom) for e in ENTREES) + 2
    print(f"{'journal':<{larg}}{'etat':<10}{'present':<9}commande")
    print("-" * 100)
    for e in ENTREES:
        present = "oui" if (JOURNAUX / f"{e.nom}.out").exists() else "NON"
        cmd = f"python3 {e.banc} {' '.join(e.args)}".rstrip()
        print(f"{e.nom:<{larg}}{e.etat:<10}{present:<9}{cmd}")
        if e.note:
            print(f"{'':<{larg}}{'':<19}-- {e.note}")
    par_etat = {}
    for e in ENTREES:
        par_etat[e.etat] = par_etat.get(e.etat, 0) + 1
    print("-" * 100)
    print("  ".join(f"{k} : {v}" for k, v in sorted(par_etat.items())))
    print("\n« verifie » ne veut pas dire « juste » : il veut dire que le")
    print("journal a ete confronte au tableau. Deux tableaux confrontes se")
    print("sont reveles faux, et c'est le but.")


def joue(e: Entree, delai: int, tmp: Path) -> int:
    part = tmp / f"{e.nom}.part"
    t0 = time.time()
    with open(part, "w") as f:
        code = subprocess.call(["timeout", str(delai), sys.executable,
                                str(RACINE / e.banc), *e.args],
                               stdout=f, stderr=subprocess.STDOUT,
                               cwd=RACINE)
    duree = int(time.time() - t0)
    if code != 0:
        with open(part, "a") as f:
            f.write(MARQUEUR.format(barre="=" * 78, code=code, duree=duree))
    shutil.move(str(part), str(JOURNAUX / f"{e.nom}.out"))
    return code


def main() -> int:
    args = sys.argv[1:]
    if "--liste" in args:
        liste()
        return 0
    delai = DELAI_DEFAUT
    if "--delai" in args:
        delai = int(args[args.index("--delai") + 1])
    refaire = set()
    while "--refaire" in args:
        i = args.index("--refaire")
        refaire.add(args[i + 1])
        del args[i:i + 2]

    JOURNAUX.mkdir(exist_ok=True)
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "reproduire"
    tmp.mkdir(parents=True, exist_ok=True)

    for e in ENTREES:
        cible = JOURNAUX / f"{e.nom}.out"
        if cible.exists() and cible.stat().st_size > 0 \
                and e.nom not in refaire:
            print(f"### {e.nom} : deja archive, passe")
            continue
        cmd = f"{e.banc} {' '.join(e.args)}".rstrip()
        print(f"### {e.nom} [{e.etat}] : {cmd}", flush=True)
        code = joue(e, delai, tmp)
        print(f"### {e.nom} : code={code}", flush=True)
    print("### TOUTES LES ENTREES TRAITEES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
