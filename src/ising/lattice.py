"""Square-lattice geometry (Krauth, Exercise 5.1).

Sites of an Lx x Ly lattice are numbered n = x + Lx * y, with column
x = n % Lx and row y = n // Lx.  Every program in the package works only
with the neighbour table, so changing the lattice means changing this file.
"""

import numpy as np

RIGHT, UP, LEFT, DOWN = 0, 1, 2, 3

T_C = 2.0 / np.log(1.0 + np.sqrt(2.0))  # Onsager's critical temperature for J = 1


def neighbour_table(Lx, Ly=None, periodic=True):
    """Return the neighbour table nbr[n, k] of an Lx x Ly square lattice.

    Column k holds the neighbour to the right (k = 0), above (1), to the
    left (2) and below (3).  With open boundaries a missing neighbour is -1.
    On a periodic lattice with Lx = 2 the right and left neighbours are the
    same site, so that pair is joined by two bonds, as in Krauth's tables.
    """
    if Ly is None:
        Ly = Lx
    n_sites = Lx * Ly
    nbr = np.full((n_sites, 4), -1, dtype=np.int64)
    steps = ((1, 0), (0, 1), (-1, 0), (0, -1))
    for n in range(n_sites):
        x, y = n % Lx, n // Lx
        for k, (dx, dy) in enumerate(steps):
            xx, yy = x + dx, y + dy
            if periodic:
                xx %= Lx
                yy %= Ly
            elif not (0 <= xx < Lx and 0 <= yy < Ly):
                continue
            nbr[n, k] = xx + Lx * yy
    return nbr


def n_bonds(nbr):
    """Number of bonds; each bond is counted once, through its right or upper end."""
    return int(np.sum(nbr[:, RIGHT] >= 0) + np.sum(nbr[:, UP] >= 0))


def energy(spins, nbr):
    """Energy E = -sum over bonds of s_i s_j, with J = 1."""
    s = spins.astype(np.int64)
    E = 0
    for k in (RIGHT, UP):
        j = nbr[:, k]
        has = j >= 0
        E -= int(np.sum(s[has] * s[j[has]]))
    return E


def magnetisation(spins):
    """Total magnetisation M = sum of the spins."""
    return int(np.sum(spins, dtype=np.int64))


def random_spins(n_sites, rng=None):
    """Independent random spins, stored as int8."""
    rng = np.random.default_rng() if rng is None else rng
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=n_sites)