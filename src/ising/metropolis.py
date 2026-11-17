"""Local Metropolis algorithm for the Ising model (Krauth Alg. 5.7, Exercise 5.10).

Each step picks a random site k and proposes to flip its spin s.  With h the
sum of the neighbouring spins, the energy change is dE = 2 s h, and the move
is accepted with probability min(1, exp(-beta dE)).  The two improvements
asked for in Exercise 5.10 are built in:

* only a few values of dE occur, so exp(-beta dE) is read from a table;
* when dE <= 0 the move is always accepted, so no random number is drawn.

E and M are updated after every accepted flip, never recomputed, and a
sweep is N attempted flips.
"""

import numpy as np
from numba import njit

from .lattice import energy, magnetisation, neighbour_table, random_spins
from .stats import Result


@njit(cache=True)
def seed(value):
    """Seed the random number generator used inside the compiled samplers."""
    np.random.seed(value)


def acceptance_table(beta):
    """Acceptance probability indexed by s*h + 4, for s*h = -4, ..., 4.

    Entries with s*h <= 0 are 1 and are never read: those moves are accepted
    without drawing a random number.
    """
    sh = np.arange(-4, 5)
    return np.where(sh > 0, np.exp(-2.0 * beta * sh), 1.0)


@njit(cache=True)
def _metropolis(spins, nbr, acc, n_blocks, block_len, E, M, series):
    n_sites = spins.shape[0]
    blocks = np.zeros((n_blocks, 5))
    record = series.shape[0] > 0
    accepted = 0
    t = 0
    for b in range(n_blocks):
        for _ in range(block_len):
            for _ in range(n_sites):
                k = np.random.randint(0, n_sites)
                h = 0
                for d in range(4):
                    neighbour = nbr[k, d]
                    if neighbour >= 0:
                        h += spins[neighbour]
                s = spins[k]
                sh = s * h
                if sh <= 0 or np.random.random() < acc[sh + 4]:
                    spins[k] = -s
                    E += 2 * sh
                    M -= 2 * s
                    accepted += 1
            e = E / n_sites
            m = M / n_sites
            blocks[b, 0] += e
            blocks[b, 1] += e * e
            blocks[b, 2] += abs(m)
            blocks[b, 3] += m * m
            blocks[b, 4] += m * m * m * m
            if record:
                series[t, 0] = E
                series[t, 1] = M
            t += 1
        for c in range(5):
            blocks[b, c] /= block_len
    return blocks, E, M, accepted


def run(L, T, n_sweeps, n_burn=None, n_blocks=64, seed_value=None, periodic=True,
        spins=None, record=False):
    """Run local Metropolis on an L x L lattice at temperature T.

    n_sweeps   measured sweeps, split into n_blocks equal blocks
    n_burn     sweeps discarded first (default: a tenth of n_sweeps, at least 1000)
    spins      starting configuration (default: random); it is not modified
    record     also return E and M after every sweep in info["series"]
    """
    beta = 1.0 / T
    nbr = neighbour_table(L, L, periodic)
    n_sites = L * L
    rng = np.random.default_rng(seed_value)
    if seed_value is not None:
        seed(seed_value)
    spins = random_spins(n_sites, rng) if spins is None else np.array(spins, dtype=np.int8)
    acc = acceptance_table(beta)
    E, M = energy(spins, nbr), magnetisation(spins)
    no_series = np.zeros((0, 2), dtype=np.int64)

    if n_burn is None:
        n_burn = max(1000, n_sweeps // 10)
    if n_burn > 0:
        _, E, M, _ = _metropolis(spins, nbr, acc, 1, n_burn, E, M, no_series)

    block_len = max(1, n_sweeps // n_blocks)
    series = np.zeros((n_blocks * block_len, 2), dtype=np.int64) if record else no_series
    blocks, E, M, accepted = _metropolis(spins, nbr, acc, n_blocks, block_len, E, M, series)

    if E != energy(spins, nbr) or M != magnetisation(spins):
        raise RuntimeError("tracked E or M disagrees with the final configuration")
    info = {"algorithm": "metropolis", "sweeps": n_blocks * block_len, "burn_in": n_burn,
            "acceptance": accepted / (n_blocks * block_len * n_sites)}
    if record:
        info["series"] = series
    return Result(blocks, beta, n_sites, spins, info)
