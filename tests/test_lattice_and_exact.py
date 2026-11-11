import numpy as np

from ising.lattice import DOWN, LEFT, RIGHT, UP, energy, magnetisation, n_bonds, neighbour_table


def test_neighbour_table_periodic_3x3():
    nbr = neighbour_table(3, 3, periodic=True)
    assert list(nbr[0]) == [1, 3, 2, 6]  # right, up, left, down
    for n in range(9):
        assert nbr[nbr[n, RIGHT], LEFT] == n
        assert nbr[nbr[n, UP], DOWN] == n


def test_neighbour_table_open_boundaries():
    nbr = neighbour_table(3, 3, periodic=False)
    assert list(nbr[0]) == [1, 3, -1, -1]
    assert list(nbr[8]) == [-1, -1, 7, 5]
    assert n_bonds(nbr) == 12


def test_ground_state_energy():
    for L, periodic, bonds in ((4, True, 32), (4, False, 24)):
        nbr = neighbour_table(L, L, periodic)
        up = np.ones(L * L, dtype=np.int8)
        assert energy(up, nbr) == -bonds
        assert energy(-up, nbr) == -bonds
        assert magnetisation(up) == L * L