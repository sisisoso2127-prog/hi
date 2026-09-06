"""
make_lot.py
===========
Ecrit sur disque un LOT D'INSTANCES FIGE, en JSON, destine a la comparaison
entre methodes.

Pourquoi figer le lot plutot que de le regenerer a la volee. Le generateur
est deterministe a graine donnee, mais il peut evoluer ; et surtout, une
methode concurrente reimplementee dans un autre langage doit pouvoir lire
exactement les memes instances. Le JSON est le format d'echange : il rend la
comparaison possible hors de ce depot.

Le lot couvre les regimes etablis par la campagne de difficulte :

  * `corr` in {0, 0.5, 0.9}  -- le levier structurel ; a `(n, m, p)` fixe,
    `A` et `b` sont identiques d'un niveau a l'autre, seul `E` change ;
  * `p` in {2, 3, 4}         -- le nombre de criteres, qui gouverne le cout
    des coupes (p binaires chacune) ;
  * deux strates de taille   -- `n <= 8` ou la verite terrain par enumeration
    est praticable, et `n >= 10` ou elle ne l'est pas. Le manifeste marque
    lesquelles, car les deux strates ne se valident pas de la meme facon.

Usage :  python make_lot.py [repertoire]
"""

from __future__ import annotations

import json
import os
import sys

from molfp_instance import generate

# strate A : verite terrain praticable (enumeration exhaustive)
SMALL = dict(ns=[5, 6, 7, 8], ps=[2, 3, 4], corrs=[0.00, 0.50, 0.90],
             seeds=[1, 2], rhs={5: 1.8, 6: 1.5, 7: 1.2, 8: 1.0})
# strate B : hors de portee de l'enumeration
LARGE = dict(ns=[10, 15, 20], ps=[3], corrs=[0.00, 0.50, 0.90],
             seeds=[1, 2], rhs={10: 1.0, 15: 1.0, 20: 1.0})


def build(spec: dict, ground_truth: bool) -> list:
    out = []
    for n in spec["ns"]:
        m = max(3, n // 2 + 1)
        for p in spec["ps"]:
            for corr in spec["corrs"]:
                for seed in spec["seeds"]:
                    inst = generate(n=n, m=m, p=p, seed=seed,
                                    rhs_scale=spec["rhs"][n], corr=corr)
                    out.append((inst, {
                        "name": inst.name, "n": n, "m": m, "p": p,
                        "corr": corr, "seed": seed,
                        "rhs_scale": spec["rhs"][n],
                        "ground_truth": ground_truth,
                    }))
    return out


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join("instances", "lot_v1")
    os.makedirs(root, exist_ok=True)

    entries = build(SMALL, True) + build(LARGE, False)
    for inst, meta in entries:
        inst.save(os.path.join(root, inst.name + ".json"))

    manifest = {
        "version": "lot_v1",
        "n_instances": len(entries),
        "n_with_ground_truth": sum(1 for _, m in entries if m["ground_truth"]),
        "note": ("Lot fige pour la comparaison entre methodes. Les instances "
                 "marquees ground_truth=true sont enumerables ; les autres ne "
                 "se valident que par les proprietes auto-certifiantes "
                 "(cf. verify_scale.py)."),
        "instances": [m for _, m in entries],
    }
    with open(os.path.join(root, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    print(f"{len(entries)} instances ecrites dans {root}/")
    print(f"  avec verite terrain : {manifest['n_with_ground_truth']}")
    print(f"  sans                : {len(entries) - manifest['n_with_ground_truth']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
