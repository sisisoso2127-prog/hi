#!/usr/bin/env python3
"""
check_croise.py -- le memoire et l'article, sujet par sujet.

POURQUOI. Deux documents racontent le meme travail avec les memes chiffres.
Une remesure corrige l'un et oublie l'autre, et rien ne le signale : chacun
compile, chacun est coherent avec lui-meme, et seule une lecture en
parallele revele que l'un affirme +0,00 la ou l'autre affirme +0,01. C'est
arrive deux fois ici -- la valeur 77,2 restee dans l'article quand le
memoire portait 80,8, puis tout le passage sur la coupe generee.

valeurs.tex supprime la divergence pour les valeurs qu'on a PENSE a y
mettre. Ce programme couvre le cas general, sans rien prevoir.

CE QUI N'A PAS MARCHE, ET POURQUOI C'EST ECRIT ICI. La premiere version
appariait les PHRASES par recouvrement de vocabulaire. Elle a echoue a son
test de mutation : reinjecte, le vrai defaut n'etait pas vu, parce que les
deux documents ne disent pas la meme chose avec les memes mots -- la prose
de l'article est reecrite, pas recopiee, et les deux phrases fautives
n'avaient que 26 % de vocabulaire commun. Apparier des phrases suppose une
ressemblance qui n'existe pas.

CE QUI MARCHE. On n'apparie plus des phrases mais des SUJETS. Un sujet est
une expression technique distinctive -- « coupe generee », « seconde
route », « vivier stable » -- reperee automatiquement comme suite de mots
pleins presente dans les deux documents et rare dans chacun. Pour chaque
sujet, on collecte les nombres cites AUTOUR de lui de part et d'autre, et
on compare les deux ensembles. Deux documents qui parlent du meme banc
doivent en citer les memes chiffres, quels que soient les mots employes.

LECTURE, ET CE QU'IL FAUT EN ATTENDRE. Ce n'est PAS une barriere : sur les
documents sains il rend environ 55 sujets, dont la quasi-totalite sont des
divergences legitimes -- un document precise ce que l'autre resume. C'est un
outil de TRI. Son merite mesure : reinjecte, le vrai defaut du passage sur
la coupe generee apparait sous le sujet « gain coupe generee », avec ses
deux versions cote a cote. Il ramene une relecture croisee de cent-vingt
phrases contre soixante-dix a une liste qu'on parcourt en quelques minutes.

A lancer apres chaque remesure, et a lire -- non a faire passer.
"""

import re
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

VIDES = {"dans", "pour", "avec", "sans", "cette", "leur", "elle", "nous",
         "donc", "plus", "moins", "meme", "même", "tout", "tous", "entre",
         "alors", "quand", "point", "points", "ligne", "lignes", "valeur",
         "valeurs", "cela", "elles", "ceux", "celle", "celles", "etre",
         "être", "avoir", "faire", "deux", "trois", "quatre", "leurs",
         "ainsi", "aussi", "mais", "chaque", "autre", "autres", "notre",
         "nos", "que", "qui", "est", "sont", "une", "des", "les", "par"}
MIN_SUJET, MAX_SUJET = 2, 4      # longueur en mots pleins
RARETE = 0.12                    # un sujet doit tenir dans 12 % des phrases


def _phrases(path: str) -> List[str]:
    t = re.sub(r"(?<!\\)%.*", "", open(path, encoding="utf-8").read())
    t = re.sub(r"\\begin\{(table|figure|tabular|algorithm)\*?\}.*?"
               r"\\end\{\1\*?\}", " ", t, flags=re.S)
    t = re.sub(r"\s+", " ", t)
    return [p.strip() for p in re.split(r"(?<=[.!?;]) ", t) if 30 < len(p) < 500]


def _pleins(s: str) -> List[str]:
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = re.sub(r"\$[^$]*\$", " ", s)
    return [w for w in re.findall(r"[a-zà-ÿ]{3,}", s.lower()) if w not in VIDES]


def _sujets(phrase: str) -> Set[str]:
    m = _pleins(phrase)
    out = set()
    for n in range(MIN_SUJET, MAX_SUJET + 1):
        for i in range(len(m) - n + 1):
            out.add(" ".join(m[i:i + n]))
    return out


def _nombres(s: str) -> Set[str]:
    return set(re.findall(r"\d+\{,\}\d+", s))


def _index(path: str) -> Tuple[Dict[str, Set[str]], Dict[str, str], int]:
    """sujet -> nombres cites autour, et un exemple de phrase."""
    phrases = _phrases(path)
    nums: Dict[str, Set[str]] = defaultdict(set)
    vus: Dict[str, int] = defaultdict(int)
    exemple: Dict[str, str] = {}
    for p in phrases:
        n = _nombres(p)
        for s in _sujets(p):
            vus[s] += 1
            if n:
                nums[s] |= n
                exemple.setdefault(s, p)
    plafond = max(2, int(RARETE * len(phrases)))
    return ({s: v for s, v in nums.items() if vus[s] <= plafond},
            exemple, len(phrases))


def controle(p1: str, p2: str) -> List[Tuple[str, Set[str], Set[str], str, str]]:
    n1, e1, _ = _index(p1)
    n2, e2, _ = _index(p2)
    out = []
    for sujet in sorted(set(n1) & set(n2)):
        a, b = n1[sujet], n2[sujet]
        if a - b and b - a:
            out.append((sujet, a - b, b - a, e1[sujet], e2[sujet]))
    # un sujet long contient ses sous-suites : on ne garde que les plus longs
    gardes = []
    for s, da, db, xa, xb in sorted(out, key=lambda t: -len(t[0])):
        if not any(s in g[0] and s != g[0] for g in gardes):
            gardes.append((s, da, db, xa, xb))
    return gardes


def main() -> int:
    p1 = sys.argv[1] if len(sys.argv) > 1 else "hybride.tex"
    p2 = sys.argv[2] if len(sys.argv) > 2 else "article.tex"
    res = controle(p1, p2)
    print(f"=== {p1} contre {p2} : {len(res)} sujet(s) a verifier\n")
    fmt = lambda s: ", ".join(sorted(x.replace("{,}", ",") for x in s))
    for sujet, da, db, xa, xb in res:
        print(f"--- sujet « {sujet} »")
        print(f"    seulement dans {p1} : {fmt(da)}")
        print(f"       {xa[:150]}")
        print(f"    seulement dans {p2} : {fmt(db)}")
        print(f"       {xb[:150]}\n")
    print(f"TOTAL : {len(res)}")
    return 1 if res else 0


if __name__ == "__main__":
    sys.exit(main())
