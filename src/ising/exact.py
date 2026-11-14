"""Exact results for small lattices.

Gray-code enumeration of every configuration gives the number of
configurations N(E, M) with energy E and magnetisation M (Krauth Alg. 5.2
and 5.3, Exercises 5.3 and 5.6).  Thermodynamic averages then follow by
summing over E and M (Alg. 5.4, Exercise 5.4).  These exact values are the
reference that every Monte Carlo result in this project is tested against.
"""

from pathlib import Path

import numpy as np
from numba import njit

from .lattice import n_bonds, neighbour_table

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "exact"


@njit(cache=True)
def _gray_enumerate(nbr, n_sites, nb):
    """Count configurations by (E, M) while visiting all 2**n_sites of them.

    Consecutive configurations of the binary-reflected Gray code differ by
    one spin: at step i it is the spin at the position of the lowest set bit
    of i.  E and M are updated locally, so each step costs O(1).
    """
    spins = -np.ones(n_sites, dtype=np.int8)  # start with all spins down
    E = -nb
    M = -n_sites
    counts = np.zeros((nb + 1, n_sites + 1), dtype=np.int64)
    counts[(E + nb) // 2, (M + n_sites) // 2] += 1
    for i in range(1, 1 << n_sites):
        k = 0
        j = i
        while (j & 1) == 0:
            j >>= 1
            k += 1
        h = 0
        for d in range(4):
            neighbour = nbr[k, d]
            if neighbour >= 0:
                h += spins[neighbour]
        s = spins[k]
        E += 2 * s * h
        M -= 2 * s
        spins[k] = -s
        counts[(E + nb) // 2, (M + n_sites) // 2] += 1
    return counts


def energy_magnetisation_counts(L, periodic=True, use_cache=True):
    """Return (E, M, N) where N[a, b] counts configurations with energy E[a]
    and magnetisation M[b] on an L x L lattice.

    L = 6 takes several minutes, so its table is cached in data/exact/.
    """
    bc = "periodic" if periodic else "open"
    path = DATA_DIR / f"counts_L{L}_{bc}.npz"
    if use_cache and path.exists():
        data = np.load(path)
        return data["E"], data["M"], data["N"]
    nbr = neighbour_table(L, L, periodic)
    n_sites = L * L
    nb = n_bonds(nbr)
    N = _gray_enumerate(nbr, n_sites, nb)
    E = -nb + 2 * np.arange(nb + 1)
    M = -n_sites + 2 * np.arange(n_sites + 1)
    if use_cache and L >= 5:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, E=E, M=M, N=N)
    return E, M, N


def density_of_states(L, periodic=True, use_cache=True):
    """Return the energies E that occur and their counts N(E)."""
    E, _, N = energy_magnetisation_counts(L, periodic, use_cache)
    g = N.sum(axis=1)
    keep = g > 0
    return E[keep], g[keep]


def _boltzmann_weights(E, g, beta):
    """Normalised weights g(E) exp(-beta E) / Z, computed with logarithms."""
    E = np.asarray(E, dtype=float)
    g = np.asarray(g, dtype=float)
    keep = g > 0
    logw = np.full(E.shape, -np.inf)
    logw[keep] = np.log(g[keep]) - beta * E[keep]
    logw -= logw.max()
    w = np.exp(logw)
    return w / w.sum()


def thermodynamics(L, T, periodic=True):
    """Exact energy per spin and specific heat per spin at temperature T."""
    E, g = density_of_states(L, periodic)
    beta = 1.0 / T
    w = _boltzmann_weights(E, g, beta)
    n_sites = L * L
    E_mean = np.sum(w * E)
    E_var = np.sum(w * (E - E_mean) ** 2)
    return E_mean / n_sites, beta**2 * E_var / n_sites


def magnetisation_moments(L, T, periodic=True):
    """Exact <|m|>, <m^2>, <m^4> and the Binder cumulant at temperature T.

    The Binder cumulant uses Krauth's convention B = (3 - <m^4>/<m^2>^2) / 2,
    which is 0 for Gaussian m (high T) and 1 for m = +-1 (low T).
    """
    E, M, N = energy_magnetisation_counts(L, periodic)
    beta = 1.0 / T
    n_sites = L * L
    wE = _boltzmann_weights(E, N.sum(axis=1), beta)
    g = N.sum(axis=1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_config = np.where(g > 0, wE / g, 0.0)
    P_M = per_config @ N.astype(float)  # probability of each magnetisation
    m = M / n_sites
    abs_m = np.sum(P_M * np.abs(m))
    m2 = np.sum(P_M * m**2)
    m4 = np.sum(P_M * m**4)
    return {"abs_m": abs_m, "m2": m2, "m4": m4, "binder": 0.5 * (3.0 - m4 / m2**2)}


def onsager_magnetisation(T):
    """Spontaneous magnetisation of the infinite 2D lattice (zero above T_c)."""
    T = np.asarray(T, dtype=float)
    x = 1.0 - np.sinh(2.0 / T) ** -4
    return np.where(x > 0, np.abs(x) ** 0.125, 0.0)