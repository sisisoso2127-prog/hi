"""
diag_enum.py
============
OU L'ENUMERATION SE BLOQUE-T-ELLE : DANS LE NOMBRE D'ILP, OU DANS LEUR
TAILLE ?

Le sondage a etabli que l'enumeration par coupes ne finit pas sur les
fronts peuples -- 178 vecteurs sur 202 a n=12 p=3, 181 sur 1874 a p=4.
Restait a savoir POURQUOI, et deux explications menent a des remedes
opposes.

  « un ILP par vecteur »  -- le temps par appel reste constant, le cout
     total est lineaire en |Z(E)|. Remede : un sondage plus fin, qui
     raccourcisse la chaine de reparation.

  « un ILP QUI GROSSIT »  -- chaque coupe d'efficacite ajoute p binaires et
     une disjonction big-M, donc le modele s'alourdit a chaque vecteur et
     le cout devient quadratique. Remede : un schema ou le modele NE
     GROSSIT PAS -- decomposition en regions, un petit programme neuf par
     region.

Les deux se departagent en chronometrant SEPAREMENT l'ILP et la
reparation ; les confondre ne dirait pas lequel des deux croit.

CE QUE LA MESURE A RENDU (n=12, p=3, corr=0, 602 s, 123 vecteurs) :

  vecteurs      10     30     50     70     90    120
  ILP (s)     0,32   2,53   5,14   5,83   6,99   8,68
  reparation  0,02   0,02   0,03   0,03   0,02   0,01

  premier quart : ILP 0,957 s   reparation 0,038 s
  dernier quart : ILP 7,766 s   reparation 0,029 s

L'ILP est multiplie par HUIT entre le premier et le dernier quart ; la
reparation ne bouge pas et pese 0,4 % du total. La premiere explication est
donc refutee, et avec elle le remede qu'elle appelait : un sondage plus fin
ne peut pas gagner plus que les trois centiemes de seconde qu'il vise.
"""
import sys, time
sys.path.insert(0, "/home/user/hi")
import numpy as np
from molfp_core import ORACLE_CALLS, reset_oracle_counter
from molfp_instance import generate
from molfp_oracle import ECutModel, repair_to_efficient
from molfp_enumere import _poids

inst = generate(n=12, m=7, p=3, seed=1, rhs_scale=1.0, corr=0.00)
reset_oracle_counter()
model = ECutModel(inst)
w = _poids(inst)
vus, t0, temps = set(), time.time(), []

while time.time() - t0 < 600:
    ta = time.time()
    res = model.optimize(w, 0.0, maximize=True, time_limit=60)
    t_ilp = time.time() - ta
    if res.status == "infeasible" or res.x is None:
        print("fini :", res.status); break
    x = np.rint(np.asarray(res.x[:inst.n])).astype(int)
    tb = time.time()
    a = repair_to_efficient(inst, x, deadline=t0 + 600)
    t_rep = time.time() - tb
    if a is None:
        print("reparation interrompue"); break
    z = tuple(inst.criteria(a))
    if z in vus:
        print("vecteur deja vu"); break
    vus.add(z)
    model.add_efficiency_cut(a)
    temps.append((len(vus), t_ilp, t_rep, ORACLE_CALLS["ilp"]))
    if len(vus) % 10 == 0:
        print(f"  {len(vus):>4} vecteurs   ILP {t_ilp:6.2f} s   "
              f"reparation {t_rep:6.2f} s   appels {ORACLE_CALLS['ilp']}",
              flush=True)

print(f"\n{len(temps)} vecteurs en {time.time()-t0:.0f} s")
if len(temps) >= 20:
    d = len(temps) // 4
    for nom, tr in (("premier quart", temps[:d]), ("dernier quart", temps[-d:])):
        mi = sum(t[1] for t in tr) / len(tr)
        mr = sum(t[2] for t in tr) / len(tr)
        print(f"  {nom:<15} ILP moyen {mi:6.3f} s   reparation {mr:6.3f} s")
