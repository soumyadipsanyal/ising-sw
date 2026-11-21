"""Compare a sampler with exact enumeration (Exercises 5.10 and 5.11)."""

import numpy as np

from . import exact
from .stats import deviation


def validate(run, L_values, T_values, n, seed=None, periodic=True, **kwargs):
    """Run the sampler at every (L, T) and compare e and c_V with exact values.

    run    the sampler's run function (metropolis.run or cluster.run)
    n      sweeps for Metropolis, cluster steps for the cluster algorithm
    """
    rows = []
    for i, L in enumerate(L_values):
        for j, T in enumerate(T_values):
            run_seed = None if seed is None else seed + 1000 * i + j
            result = run(L, T, n, seed_value=run_seed, periodic=periodic, **kwargs)
            e_exact, c_exact = exact.thermodynamics(L, T, periodic)
            e, e_err = result.energy()
            c, c_err = result.specific_heat()
            rows.append({
                "L": L, "T": T,
                "e_exact": e_exact, "e": e, "e_err": e_err, "e_dev": deviation(e, e_err, e_exact),
                "c_exact": c_exact, "c": c, "c_err": c_err, "c_dev": deviation(c, c_err, c_exact),
            })
    return rows


def markdown_table(rows):
    """The validation rows as a Markdown table; deviations are in error bars."""
    lines = [
        "| L | T | e exact | e Monte Carlo | dev | c_V exact | c_V Monte Carlo | dev |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['L']} | {r['T']:.4f} | {r['e_exact']:.5f} | {r['e']:.5f} ± {r['e_err']:.5f} | "
            f"{r['e_dev']:+.1f} | {r['c_exact']:.5f} | {r['c']:.5f} ± {r['c_err']:.5f} | {r['c_dev']:+.1f} |"
        )
    return "\n".join(lines)


def summary(rows):
    """One line: how many results lie within two error bars of the exact value."""
    devs = np.array([[r["e_dev"], r["c_dev"]] for r in rows]).ravel()
    within = int(np.sum(np.abs(devs) <= 2))
    return f"{within} of {devs.size} results lie within two error bars of the exact value."