#!/usr/bin/env python3
"""
check_sensib.py
===============
LE TABLEAU DE SENSIBILITE CONTRE SON JOURNAL, CELLULE PAR CELLULE.

`tab:sensib` compte quarante-quatre lignes et quatre colonnes de mesure,
soit environ cent-soixante-quinze nombres. Personne ne les relit. C'est
precisement le genre de tableau ou une valeur perimee survit des mois : ce
memoire en a deja corrige deux series, et les deux fois le symptome etait
le meme -- un defaut change, une campagne non rejouee, un tableau qui
continue d'afficher l'ancien regime.

Ce programme rejoue la comparaison a la place du relecteur. Il lit le
journal de `bench_sensibilite.py` archive dans `results/`, lit le tableau
dans le document, et les confronte poste par poste.

CE QU'IL COMPARE. Le nombre de preuves sur la strate A, l'ecart median sur
la strate B, l'ecart a la production, et le compte apparie. Les lignes de
production sont comparees sur les deux premieres colonnes seulement : le
tableau y porte « --- » et « production » la ou le journal imprime
« +0.00 » et « 0/0 », et ce n'est pas une divergence.

CE QU'IL NE FAIT PAS. Il ne verifie pas que le journal est recent. Un
journal perime et un tableau perime concordent parfaitement. La seule
parade est de rejouer le banc, et c'est pour cela que le journal porte sa
date dans git plutot que dans le fichier.

APPARIEMENT. Par POSITION, pas par etiquette : le journal et le tableau
listent les memes facteurs dans le meme ordre, et comparer
« $(0{,}2\\,;0{,}4)$ » a « (0.2, 0.4) » demanderait un analyseur de plus,
donc une source de faux negatifs de plus. Les etiquettes sont tout de meme
rapprochees, en ne gardant que leurs chiffres et leurs lettres, et un
desaccord est signale sans etre fatal.

TEST PAR MUTATION. Un fascheur qu'on n'abime pas volontairement est un
fascheur dont on ignore s'il regarde. `mutation_sensib.py` fabrique un
journal qui concorde avec le tableau, verifie que ce fascheur se taise,
puis abime le tableau d'une cellule a la fois : ecart, preuves, delta,
compte apparie, etiquette, signe, cellule en gras, ligne de production
devenue ordinaire, ligne supprimee, et deux lignes echangees. Les douze
sont attrapees, l'echange comptant pour quatre divergences puisque
l'appariement est positionnel et ne se recale pas en silence. Un treizieme
temoin fait deborder la colonne des valeurs, parce que c'est par la que ce
fascheur a d'abord failli : voir le commentaire de `RE_LIGNE`.

Le meme programme joue aussi les deux mutations que ce fascheur NE PEUT
PAS voir, et verifie qu'il ne les voit effectivement pas : le renommage
d'un facteur, et la meme faute commise des deux cotes. Un test par
mutation qui ne montre que ses reussites laisse croire a une couverture
qu'il n'a pas.

Usage :  python check_sensib.py [journal] [document]
         python mutation_sensib.py      (depuis doc/)
"""

from __future__ import annotations

import re
import sys
from typing import List, Optional, Tuple

JOURNAL = "../results/bench_sensibilite.out"
DOCUMENT = "hybride.tex"

Ligne = Tuple[str, str, Optional[int], Optional[float], Optional[float],
              Optional[str], bool]
#     (facteur, valeur, preuves A, ecart B, delta B, mieux/pire, production)


# ---------------------------------------------------------------------------
# le journal
# ---------------------------------------------------------------------------

# On ANCRE sur la queue numerique au lieu de decouper a des colonnes
# fixes. Le banc ecrit la valeur en « {:>22} », qui ne tronque pas : un
# menu comme ('A', 'A', 'A', 'B', 'C') fait vingt-cinq caracteres et
# decale tout le reste de trois crans. Une premiere version de ce fichier
# decoupait a la colonne 52 ; elle laissait tomber EN SILENCE les trois
# lignes trop larges, et l'examen qui suivait comparait le tableau a un
# journal desaligne -- quatorze fausses divergences, dont pas une vraie.
RE_LIGNE = re.compile(
    r"^(.*?)"                          # facteur (30 car.) puis valeur
    r"\s+(\d+)/(\d+)"                  # preuves A / n
    r"\s+(-?[\d.]+|nan)"               # ecart A
    r"\s+(-?[\d.]+|nan)"               # ecart B
    r"\s+([+-][\d.]+|[+-]?nan)"        # delta B
    r"\s+(\d+/\d+)"                    # mieux/pire
    r"(.*)$")


def lire_journal(chemin: str) -> List[Ligne]:
    out: List[Ligne] = []
    facteur = ""
    for brut in open(chemin, encoding="utf-8", errors="replace"):
        ligne = brut.rstrip("\n")
        if not ligne or ligne[0] in "=-" or ligne.startswith("facteur"):
            continue
        m = RE_LIGNE.match(ligne)
        if not m:
            continue
        tete = m.group(1)
        nom, valeur = tete[:30].strip(), tete[30:].strip()
        if not valeur:          # une ligne de resume, pas une mesure
            continue
        if nom:
            facteur = nom
        # groupes : 1 tete, 2 preuves, 3 n, 4 ecart A, 5 ecart B,
        # 6 delta B, 7 mieux/pire, 8 la marque. Les colonnes A et B
        # se ressemblent assez pour qu'un decalage d'un cran passe
        # inapercu : il avait fait lire l'ecart de la strate A comme celui
        # de la strate B.
        prod = "production" in m.group(8)
        out.append((facteur, valeur, int(m.group(2)),
                    _f(m.group(5)), _f(m.group(6)), m.group(7), prod))
    return out


def _f(t: str) -> Optional[float]:
    try:
        return float(t)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# le tableau
# ---------------------------------------------------------------------------

def lire_tableau(chemin: str, etiquette: str) -> List[Ligne]:
    txt = open(chemin, encoding="utf-8").read()
    i = txt.find("\\label{" + etiquette + "}")
    if i < 0:
        raise SystemExit(f"etiquette {etiquette} introuvable dans {chemin}")
    corps = txt[i:txt.find("\\end{tabular}", i)]
    corps = corps[corps.find("\\midrule"):]

    out: List[Ligne] = []
    facteur = ""
    for rang in corps.split("\\\\"):
        rang = rang.replace("\\midrule", " ").replace("\\bottomrule", " ")
        if "&" not in rang:
            continue
        cells = [c.strip() for c in rang.split("&")]
        if len(cells) != 6:
            continue
        if cells[0]:
            m = re.search(r"\\multirow\{\d+\}\{\*\}\{(.*)\}\s*$", cells[0],
                          re.S)
            facteur = _texte(m.group(1) if m else cells[0])
        prod = "production" in cells[5]
        out.append((facteur, _texte(cells[1]), _preuves(cells[2]),
                    _nb(cells[3]), _nb(cells[4]),
                    None if prod else _texte(cells[5]), prod))
    return out


def _texte(c: str) -> str:
    c = re.sub(r"\\(?:mathbf|textbf|emph|texttt|text|mathit)\b", " ", c)
    for x in ("$", "{", "}", "\\,", "\;", "\\ ", "_"):
        c = c.replace(x, "")
    return c.strip()


def _preuves(c: str) -> Optional[int]:
    m = re.search(r"(\d+)\s*/\s*(\d+)", _texte(c))
    return int(m.group(1)) if m else None


def _nb(c: str) -> Optional[float]:
    t = _texte(c).replace("---", "").replace("−", "-")
    t = t.replace(",", ".").replace("+", "")
    m = re.search(r"-?\d+\.?\d*", t)
    return float(m.group(0)) if m else None


# ---------------------------------------------------------------------------

def chiffres(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def main() -> int:
    journal = sys.argv[1] if len(sys.argv) > 1 else JOURNAL
    document = sys.argv[2] if len(sys.argv) > 2 else DOCUMENT

    j = lire_journal(journal)
    t = lire_tableau(document, "tab:sensib")

    print("=" * 78)
    print(f"tab:sensib  ({document})  contre  {journal}")
    print(f"  journal : {len(j)} lignes     tableau : {len(t)} lignes")
    print("=" * 78)

    # Un balayage complet dure des heures. Refuser de repondre tant qu'il
    # n'est pas fini rendrait ce programme inutilisable pendant tout le
    # temps ou il servirait le plus. On compare donc le PREFIXE commun,
    # sous un code de retour distinct : un examen partiel n'est pas un
    # verdict, et rien ne doit pouvoir le lire comme tel.
    partiel = len(j) < len(t)
    if len(j) > len(t):
        print(f"\n!! le journal a PLUS de lignes que le tableau : "
              f"{len(j)} contre {len(t)}.")
        print("   Le banc a-t-il ete joue avec le meme jeu de facteurs ?")
        return 1
    if partiel:
        print(f"\n** EXAMEN PARTIEL : le journal s'arrete a la ligne "
              f"{len(j)} sur {len(t)}.")
        print("   Ce n'est pas un verdict, c'est une lecture anticipee.")
        t = t[:len(j)]

    ecarts = 0
    for k, (a, b) in enumerate(zip(j, t), 1):
        fa, va, pa, ea, da, ma, proda = a
        fb, vb, pb, eb, db, mb, prodb = b
        ou = []
        if chiffres(va) != chiffres(vb):
            ou.append(f"valeur  journal « {va} »  tableau « {vb} »")
        if proda != prodb:
            ou.append(f"ligne de production : journal {proda}, "
                      f"tableau {prodb}")
        if pa != pb:
            ou.append(f"preuves A  journal {pa}  tableau {pb}")
        if not _proche(ea, eb):
            ou.append(f"ecart B    journal {ea}  tableau {eb}")
        if not proda:
            if not _proche(da, db):
                ou.append(f"delta B    journal {da}  tableau {db}")
            if ma != mb:
                ou.append(f"mieux/pire journal {ma}  tableau {mb}")
        if ou:
            ecarts += len(ou)
            print(f"\n  ligne {k:>2}  {fb or fa} = {vb or va}")
            for o in ou:
                print(f"      {o}")

    print(f"\nTOTAL : {ecarts} divergence(s) sur "
          f"{len(j)} lignes comparees")
    if ecarts:
        return 1
    if partiel:
        print("Aucune divergence sur les lignes deja jouees. Il en reste "
              f"{len(lire_tableau(document, 'tab:sensib')) - len(j)}.")
        return 2
    print("Le tableau imprime est bien le journal archive.")
    return 0


def _proche(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) < 0.005


if __name__ == "__main__":
    raise SystemExit(main())
