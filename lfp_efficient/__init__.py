"""Optimizing a linear fractional function over an integer efficient set.

Reference implementation of the exact algorithm of

    Leila Younsi-Abbaci, *Optimizing a linear fractional function over an
    integer efficient set*, Reliability: Theory & Applications, No 4 (40),
    Volume 11, March 2025.

Quick start::

    from lfp_efficient import Model, MOILP, FractionalObjective, LE
    from lfp_efficient import optimize_over_efficient_set

    D = Model(2).add([-2, 1], LE, 0).add([6, 1], LE, 21).add([-2, 4], LE, 6)
    problem = MOILP(D, criteria=[[1, -2], [-1, 4]])
    phi = FractionalObjective(U=[1, 1], V=[5, 1], alpha=-1, beta=-1)

    solution = optimize_over_efficient_set(problem, phi, verbose=True)
    print(solution.x, solution.value)       # (3, 3)  5/17

The package is pure Python with no third-party dependency and computes in
exact rational arithmetic.
"""

from .algorithm import IterationLog, Solution, optimize_over_efficient_set
from .criterion_space import Box, optimize_in_criterion_space, remove_everywhere
from .hybrid import optimize_hybrid
from .metaheuristic import (ParetoArchive, metaheuristic_incumbent,
                            optimize_hybrid_metaheuristic, pareto_local_search,
                            random_maximal_point)
from .edges import (alternative_optima_columns, clean_tableau_at,
                    edge_direction, explore_edges, max_step, max_step_in,
                    reduced_gradient, walk_edge)
from .efficiency import (EfficiencyTest, add_dominance_cut, add_sylva_crema_cut,
                         efficient_dominator,
                         best_with_same_criterion, has_linear_criteria,
                         lower_bounds, repair_to_efficient, spread_weights,
                         test_efficiency, weighted_sum_efficient)
from .enumeration import (Enumeration, FullEnumeration,
                          enumerate_efficient_set, enumerate_nondominated,
                          maximize_by_full_enumeration,
                          best_over_efficient_set_by_scan,
                          Certificate, certify_optimum)
from .milp import (MilpResult, solve_fractional_milp, solve_linear_milp,
                   solve_milp, solve_relaxation)
from .model import (EQ, GE, LE, Constraint, FractionalObjective, MOILFP,
                    MOILP, Model)
from .rational import F, fmt
from .front import Front, enumerate_front
from .subset import EfficientSubset, efficient_subset
from .tchebychev import (augmented_tchebychev_efficient, ideal_point,
                         tchebychev_incumbent)

__all__ = [
    "Model", "MOILP", "MOILFP", "FractionalObjective", "Constraint",
    "LE", "GE", "EQ",
    "optimize_over_efficient_set", "Solution", "IterationLog",
    "optimize_in_criterion_space", "Box", "remove_everywhere",
    "optimize_hybrid", "optimize_hybrid_metaheuristic",
    "metaheuristic_incumbent", "pareto_local_search", "ParetoArchive",
    "random_maximal_point",
    "test_efficiency", "EfficiencyTest", "lower_bounds",
    "add_sylva_crema_cut", "add_dominance_cut", "best_with_same_criterion",
    "repair_to_efficient", "efficient_dominator",
    "weighted_sum_efficient", "spread_weights",
    "has_linear_criteria",
    "augmented_tchebychev_efficient", "ideal_point", "tchebychev_incumbent",
    "reduced_gradient", "alternative_optima_columns", "max_step", "max_step_in",
    "edge_direction", "walk_edge", "explore_edges", "clean_tableau_at",
    "solve_milp", "solve_linear_milp", "solve_fractional_milp",
    "solve_relaxation", "MilpResult",
    "enumerate_efficient_set", "Enumeration",
    "enumerate_nondominated", "maximize_by_full_enumeration", "FullEnumeration",
    "best_over_efficient_set_by_scan", "certify_optimum", "Certificate",
    "efficient_subset", "EfficientSubset",
    "enumerate_front", "Front",
    "F", "fmt",
]

__version__ = "1.0.0"
