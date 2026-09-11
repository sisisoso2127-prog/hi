"""
molfp_matheuristic.py
=====================
ALIAS DE COMPATIBILITE.

La methode hybride a ete deplacee dans `molfp_hybride.py`, qui ne contient
plus qu'elle. Ce fichier ne subsiste que pour les bancs et les scripts de
verification qui importaient `molfp_matheuristic` : ils fonctionnent sans
changement. Rien n'est redefini ici, tout est reexporte.
"""

from molfp_hybride import *          # noqa: F401,F403
import molfp_hybride as _h

# les noms prefixes d'un souligne ne passent pas par l'etoile
_select_pool = _h._select_pool       # noqa: F401
__all__ = [n for n in dir(_h) if not n.startswith("__")]
