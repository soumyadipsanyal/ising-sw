import numpy as np
import pytest

from ising import exact
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


def test_density_of_states_2x2_and_4x4_match_krauth_table_5_2():
    E, g = exact.density_of_states(2, use_cache=False)
    assert dict(zip(E.tolist(), g.tolist())) == {-8: 2, 0: 12, 8: 2}
    E, g = exact.density_of_states(4, use_cache=False)
    expected = {-32: 2, -24: 32, -20: 64, -16: 424, -12: 1728, -8: 6688, -4: 13568, 0: 20524}
    table = dict(zip(E.tolist(), g.tolist()))
    for e, count in expected.items():
        assert table[e] == count
        assert table[-e] == count
    assert g.sum() == 2**16


def test_gray_code_matches_brute_force_with_open_boundaries():
    L = 3
    nbr = neighbour_table(L, L, periodic=False)
    E, M, N = exact.energy_magnetisation_counts(L, periodic=False, use_cache=False)
    brute = np.zeros_like(N)
    nb = n_bonds(nbr)
    for c in range(2 ** (L * L)):
        spins = np.array([1 if (c >> i) & 1 else -1 for i in range(L * L)], dtype=np.int8)
        brute[(energy(spins, nbr) + nb) // 2, (magnetisation(spins) + L * L) // 2] += 1
    assert np.array_equal(N, brute)


def test_bipartite_symmetry_exercise_5_5():
    """N(E) = N(-E) on bipartite lattices (4x4 torus) but not on the 3x3 torus."""
    E, g = exact.density_of_states(4, use_cache=False)
    assert np.array_equal(g, g[::-1])
    E, g = exact.density_of_states(3, use_cache=False)
    assert not np.array_equal(g, g[::-1])


@pytest.mark.parametrize("T", [1.5, 2.0, 2.5, 3.5])
def test_specific_heat_is_derivative_of_energy_exercise_5_4(T):
    h = 1e-4
    e_plus, _ = exact.thermodynamics(4, T + h)
    e_minus, _ = exact.thermodynamics(4, T - h)
    _, c = exact.thermodynamics(4, T)
    assert (e_plus - e_minus) / (2 * h) == pytest.approx(c, rel=1e-6)


def test_cached_6x6_table_if_present():
    path = exact.DATA_DIR / "counts_L6_periodic.npz"
    if not path.exists():
        pytest.skip("6x6 table not computed yet")
    E, g = exact.density_of_states(6)
    table = dict(zip(E.tolist(), g.tolist()))
    assert g.sum() == 2**36
    assert (table[-72], table[-64], table[-60], table[-56]) == (2, 72, 144, 1620)


def test_binder_cumulant_limits():
    assert exact.magnetisation_moments(4, 0.5)["binder"] == pytest.approx(1.0, abs=1e-6)
    assert exact.magnetisation_moments(4, 1e4)["binder"] < 0.1