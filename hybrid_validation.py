# ============================================================
# FINAL HYBRID VALIDATION - GOOGLE COLAB VERSION
# Batch Ranking + Adaptive K + Targeted Diversification
# + Pareto Archive + Dinkelbach Decision Maker
# ============================================================

import numpy as np


# ============================================================
# BENCHMARK MODEL
# ============================================================

def feasible_solutions(n):
    solutions = []

    for x1 in range(n + 1):
        for x2 in range(n + 1 - x1):
            solutions.append(
                np.array([x1, x2], dtype=int)
            )

    return solutions


def objectives(x):
    x1 = int(x[0])
    x2 = int(x[1])

    return np.array(
        [
            2.0 * x1 + 2.0 * x2,
            1.0 * x1 + 4.0 * x2
        ],
        dtype=float
    )


# ============================================================
# DOMINANCE
# Both objectives are MAXIMIZED
# ============================================================

def dominates(fa, fb):
    return (
        np.all(fa >= fb)
        and np.any(fa > fb)
    )


def pareto_filter(points):
    result = []

    for x, f in points:
        dominated = False

        for _, g in points:
            if dominates(g, f):
                dominated = True
                break

        if not dominated:
            result.append(
                (
                    x.copy(),
                    f.copy()
                )
            )

    return result


def exact_pareto(n):
    all_points = []

    for x in feasible_solutions(n):
        all_points.append(
            (
                x.copy(),
                objectives(x)
            )
        )

    return pareto_filter(all_points)


# ============================================================
# SEARCH
# ============================================================

class Search:

    def __init__(self, n):
        self.n = n

        self.all_x = feasible_solutions(n)

        self.visited = set()

        self.archive = []

        self.oracle_calls = 0

        self.iterations = 0

    # --------------------------------------------------------
    # KEY
    # --------------------------------------------------------

    def key(self, x):
        return tuple(
            int(v) for v in x
        )

    # --------------------------------------------------------
    # ORACLE
    # --------------------------------------------------------

    def oracle(self, x):

        k = self.key(x)

        if k in self.visited:
            return None

        self.visited.add(k)

        self.oracle_calls += 1

        return objectives(x)

    # --------------------------------------------------------
    # ADD TO ARCHIVE
    # --------------------------------------------------------

    def add_archive(self, x, f):

        # If already dominated by archive, reject.
        for _, g in self.archive:

            if dominates(g, f):
                return False

        # Remove archive points dominated by new point.
        new_archive = []

        for y, g in self.archive:

            if not dominates(f, g):
                new_archive.append(
                    (
                        y.copy(),
                        g.copy()
                    )
                )

        self.archive = new_archive

        # Avoid duplicates.
        for y, _ in self.archive:

            if self.key(y) == self.key(x):
                return False

        self.archive.append(
            (
                x.copy(),
                f.copy()
            )
        )

        return True

    # --------------------------------------------------------
    # RAW CANDIDATES
    # --------------------------------------------------------

    def raw_candidates(self):

        return [
            x
            for x in self.all_x
            if self.key(x) not in self.visited
        ]

    # --------------------------------------------------------
    # FRONTIER SCORE
    # --------------------------------------------------------

    def frontier_score(self, x):

        f = objectives(x)

        if not self.archive:
            return float(np.sum(f))

        archive_f = np.array(
            [
                g
                for _, g in self.archive
            ],
            dtype=float
        )

        # Improvement potential.
        improvement_values = []

        for af in archive_f:

            diff = f - af

            value = -np.sum(
                np.minimum(diff, 0.0)
            )

            improvement_values.append(value)

        improvement = max(
            improvement_values
        )

        # Objective-space distance.
        mins = archive_f.min(axis=0)
        maxs = archive_f.max(axis=0)

        scale = np.maximum(
            maxs - mins,
            1.0
        )

        z = (
            f - mins
        ) / scale

        distances = []

        for af in archive_f:

            az = (
                af - mins
            ) / scale

            distances.append(
                np.linalg.norm(
                    az - z
                )
            )

        gap = min(distances)

        # Boundary preference.
        x1 = int(x[0])
        x2 = int(x[1])

        boundary = 0.0

        if (
            x1 == 0
            or x2 == 0
            or x1 + x2 == self.n
        ):
            boundary = 1.0

        # Trade-off preference.
        tradeoff = (
            -abs(x1 - x2)
            / max(self.n, 1)
        )

        return (
            2.0 * improvement
            + 4.0 * gap
            + 1.5 * boundary
            + tradeoff
        )

    # --------------------------------------------------------
    # RANK
    # --------------------------------------------------------

    def rank(self, candidates, k):

        scored = []

        for x in candidates:

            score = self.frontier_score(x)

            scored.append(
                (
                    score,
                    x.copy()
                )
            )

        scored.sort(
            key=lambda z: z[0],
            reverse=True
        )

        selected = []

        selected_f = []

        for score, x in scored:

            f = objectives(x)

            if not selected:

                selected.append(
                    (
                        score,
                        x.copy()
                    )
                )

                selected_f.append(
                    f.copy()
                )

            else:

                distances = []

                for sf in selected_f:

                    distances.append(
                        np.linalg.norm(
                            f - sf
                        )
                    )

                d = min(distances)

                adjusted_score = (
                    score
                    + 0.25 * d
                )

                selected.append(
                    (
                        adjusted_score,
                        x.copy()
                    )
                )

                selected_f.append(
                    f.copy()
                )

            if len(selected) >= k:
                break

        selected.sort(
            key=lambda z: z[0],
            reverse=True
        )

        return selected

    # --------------------------------------------------------
    # TARGETED DIVERSIFICATION
    # --------------------------------------------------------

    def targeted_diversification(self):

        candidates = self.raw_candidates()

        if not candidates:
            return None

        # Small archive:
        # choose a boundary point near the center.
        if len(self.archive) < 2:

            boundary = []

            for x in candidates:

                if (
                    x[0] == 0
                    or x[1] == 0
                    or x.sum() == self.n
                ):
                    boundary.append(x)

            if boundary:

                return min(
                    boundary,
                    key=lambda x:
                    abs(
                        int(x[0])
                        - int(x[1])
                    )
                )

            return max(
                candidates,
                key=self.frontier_score
            )

        # Sort archive by x1.
        ordered = sorted(
            self.archive,
            key=lambda p: int(p[0][0])
        )

        gaps = []

        for a, b in zip(
            ordered[:-1],
            ordered[1:]
        ):

            fa = a[1]
            fb = b[1]

            gap = np.linalg.norm(
                fb - fa
            )

            midpoint = (
                a[0].astype(float)
                + b[0].astype(float)
            ) / 2.0

            gaps.append(
                (
                    gap,
                    midpoint
                )
            )

        if not gaps:
            return None

        _, target = max(
            gaps,
            key=lambda z: z[0]
        )

        target_int = (
            np.rint(target)
            .astype(int)
        )

        return min(
            candidates,
            key=lambda x:
            np.linalg.norm(
                objectives(x)
                - objectives(target_int)
            )
        )

    # --------------------------------------------------------
    # RUN SEARCH
    # --------------------------------------------------------

    def run(self):

        # Initial solution.
        x0 = np.array(
            [0, 0],
            dtype=int
        )

        f0 = self.oracle(x0)

        self.add_archive(
            x0,
            f0
        )

        K = 3

        stagnation = 0

        max_stagnation = 3

        while True:

            self.iterations += 1

            raw = self.raw_candidates()

            if not raw:
                break

            batch_size = min(
                K,
                len(raw)
            )

            batch = self.rank(
                raw,
                batch_size
            )

            if not batch:
                break

            new_count = 0

            for _, x in batch:

                f = self.oracle(x)

                if f is None:
                    continue

                if self.add_archive(x, f):

                    new_count += 1

            if new_count > 0:

                stagnation = 0

                K = max(
                    3,
                    min(K, 24)
                )

            else:

                stagnation += 1

                if stagnation >= max_stagnation:

                    d = self.targeted_diversification()

                    if d is None:
                        break

                    f = self.oracle(d)

                    if f is None:
                        break

                    if self.add_archive(d, f):

                        stagnation = 0

                    K = min(
                        24,
                        K * 2
                    )

        return self.archive


# ============================================================
# DINKELBACH
# ============================================================

def N_value(f):

    return (
        2.0 * f[0]
        + 1.0 * f[1]
    )


def D_value(f):

    return (
        0.5 * f[0]
        - 0.1 * f[1]
        + 1.0
    )


def dinkelbach(
    archive,
    verbose=True,
    tol=1e-12,
    max_iter=50
):

    lam = 0.0

    last_x = None
    last_f = None
    last_phi = None
    last_residual = None

    for it in range(
        1,
        max_iter + 1
    ):

        values = []

        for x, f in archive:

            n = N_value(f)

            d = D_value(f)

            value = (
                n
                - lam * d
            )

            values.append(
                (
                    value,
                    x.copy(),
                    f.copy(),
                    n,
                    d
                )
            )

        best = max(
            values,
            key=lambda z: z[0]
        )

        _, x, f, n, d = best

        phi = n / d

        residual = (
            n
            - lam * d
        )

        last_x = x.copy()
        last_f = f.copy()
        last_phi = phi
        last_residual = residual

        if verbose:

            print(
                f"{it:3d} "
                f"lambda={lam:14.8f} "
                f"x={x} "
                f"N={n:14.6f} "
                f"D={d:14.6f} "
                f"Phi={phi:14.8f} "
                f"Residual={residual:14.4e}"
            )

        if abs(residual) <= tol:

            return (
                last_x,
                last_f,
                last_phi,
                it,
                last_residual
            )

        lam = phi

    return (
        last_x,
        last_f,
        last_phi,
        max_iter,
        last_residual
    )


# ============================================================
# UTILITY
# ============================================================

def keyset(points):

    return {
        tuple(
            int(v)
            for v in x
        )
        for x, _ in points
    }


# ============================================================
# RUN CASE
# ============================================================

def run_case(
    name,
    n,
    verbose=True
):

    print()
    print("=" * 100)
    print(
        name
        + " - BATCH RANKING + ADAPTIVE K + TARGETED DIVERSIFICATION"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # Exact Pareto
    # --------------------------------------------------------

    exact = exact_pareto(n)

    print(
        f"Feasible solutions : "
        f"{len(feasible_solutions(n))}"
    )

    print(
        f"Exact Pareto size  : "
        f"{len(exact)}"
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    search = Search(n)

    archive = search.run()

    # Final filtering.
    archive = pareto_filter(
        archive
    )

    # --------------------------------------------------------
    # Pareto validation
    # --------------------------------------------------------

    exact_keys = keyset(exact)

    archive_keys = keyset(
        archive
    )

    missing = (
        exact_keys
        - archive_keys
    )

    extra = (
        archive_keys
        - exact_keys
    )

    intersection = (
        exact_keys
        & archive_keys
    )

    recall = (
        len(intersection)
        / max(
            len(exact_keys),
            1
        )
    )

    precision = (
        len(intersection)
        / max(
            len(archive_keys),
            1
        )
    )

    print()
    print("PARETO VALIDATION")
    print("-" * 100)

    print(
        f"Exact Pareto size      : "
        f"{len(exact)}"
    )

    print(
        f"Archive size           : "
        f"{len(archive)}"
    )

    print(
        f"Pareto Recall          : "
        f"{recall:.6f}"
    )

    print(
        f"Pareto Precision       : "
        f"{precision:.6f}"
    )

    print(
        f"Missing                : "
        f"{missing}"
    )

    print(
        f"Extra                  : "
        f"{extra}"
    )

    # --------------------------------------------------------
    # Final archive
    # --------------------------------------------------------

    print()
    print("FINAL PARETO ARCHIVE")
    print("-" * 100)

    sorted_archive = sorted(
        archive,
        key=lambda p:
        (
            int(p[0][0]),
            int(p[0][1])
        )
    )

    for i, (x, f) in enumerate(
        sorted_archive,
        start=1
    ):

        print(
            f"{i:2d}. "
            f"x = {x}    "
            f"F(x) = {f}"
        )

    # --------------------------------------------------------
    # Dinkelbach
    # --------------------------------------------------------

    print()
    print("DINKELBACH")
    print("-" * 100)

    dm_x, dm_f, phi, dit, residual = dinkelbach(
        archive,
        verbose=verbose
    )

    # --------------------------------------------------------
    # Exact DM
    #
    # IMPORTANT:
    # The exact DM is obtained by Dinkelbach over the
    # COMPLETE feasible set, not hard-coded.
    # --------------------------------------------------------

    all_points = [
        (
            x.copy(),
            objectives(x)
        )
        for x in feasible_solutions(n)
    ]

    exact_lam = 0.0

    for _ in range(100):

        best = max(
            all_points,
            key=lambda p:
            N_value(p[1])
            - exact_lam * D_value(p[1])
        )

        exact_x = best[0]
        exact_f = best[1]

        exact_phi = (
            N_value(exact_f)
            / D_value(exact_f)
        )

        exact_residual = (
            N_value(exact_f)
            - exact_lam
            * D_value(exact_f)
        )

        if abs(exact_residual) <= 1e-12:
            break

        exact_lam = exact_phi

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    dm_ok = np.array_equal(
        dm_x,
        exact_x
    )

    phi_ok = (
        abs(
            phi
            - exact_phi
        )
        <= 1e-10
    )

    pareto_ok = (
        recall == 1.0
        and precision == 1.0
    )

    converged = (
        abs(residual)
        <= 1e-12
    )

    abs_phi_gap = abs(
        phi
        - exact_phi
    )

    relative_phi_gap = (
        abs_phi_gap
        / max(
            abs(exact_phi),
            1e-12
        )
    )

    print()
    print("HYBRID FINAL VALIDATION")
    print("-" * 100)

    print(
        f"Exact DM x             : "
        f"{exact_x}"
    )

    print(
        f"Exact DM F(x)          : "
        f"{exact_f}"
    )

    print(
        f"Exact Phi*             : "
        f"{exact_phi:.10f}"
    )

    print(
        f"Hybrid DM x            : "
        f"{dm_x}"
    )

    print(
        f"Hybrid DM F(x)         : "
        f"{dm_f}"
    )

    print(
        f"Hybrid Phi             : "
        f"{phi:.10f}"
    )

    print(
        f"Absolute Phi Gap       : "
        f"{abs_phi_gap:.10e}"
    )

    print(
        f"Relative Phi Gap       : "
        f"{relative_phi_gap:.10e}"
    )

    print(
        f"Same DM solution       : "
        f"{dm_ok}"
    )

    print(
        f"Dinkelbach iterations  : "
        f"{dit}"
    )

    print(
        f"Final residual         : "
        f"{residual:.4e}"
    )

    # --------------------------------------------------------
    # Computational effort
    # --------------------------------------------------------

    print()
    print("COMPUTATIONAL EFFORT")
    print("-" * 100)

    print(
        f"Oracle calls           : "
        f"{search.oracle_calls}"
    )

    print(
        f"Search iterations      : "
        f"{search.iterations}"
    )

    print(
        f"Visited solutions      : "
        f"{len(search.visited)}"
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    overall = (
        pareto_ok
        and dm_ok
        and phi_ok
        and converged
    )

    print()
    print("=" * 100)

    print(
        f"{name} FINAL RESULT"
    )

    print("=" * 100)

    print(
        f"Complete Pareto recovery : "
        f"{pareto_ok}"
    )

    print(
        f"Decision-Maker correct   : "
        f"{dm_ok}"
    )

    print(
        f"Phi correct              : "
        f"{phi_ok}"
    )

    print(
        f"Dinkelbach converged     : "
        f"{converged}"
    )

    print(
        f"Overall                  : "
        f"{overall}"
    )

    print(
        "RESULT:",
        "PASS" if overall else "FAIL"
    )

    print("=" * 100)

    return overall


# ============================================================
# MAIN - COLAB VERSION
# ============================================================

print()
print("#" * 100)
print("FINAL HYBRID VALIDATION")
print("#" * 100)

medium_result = run_case(
    "MEDIUM_1",
    8,
    verbose=True
)

large_result = run_case(
    "LARGE_1",
    20,
    verbose=True
)

print()
print("#" * 100)
print("GLOBAL RESULT")
print("#" * 100)

print(
    "MEDIUM_1:",
    "PASS" if medium_result else "FAIL"
)

print(
    "LARGE_1 :",
    "PASS" if large_result else "FAIL"
)

both_result = (
    medium_result
    and large_result
)

print(
    "BOTH    :",
    "PASS" if both_result else "FAIL"
)

print("#" * 100)
