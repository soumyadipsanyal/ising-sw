import numpy as np
import pytest

from ising import exact, metropolis
from ising.lattice import energy, magnetisation, neighbour_table


def test_acceptance_table():
    beta = 0.4
    acc = metropolis.acceptance_table(beta)
    assert acc[4 + 2] == pytest.approx(np.exp(-4 * beta))  # s*h = 2, dE = 4
    assert acc[4 + 4] == pytest.approx(np.exp(-8 * beta))  # s*h = 4, dE = 8
    assert np.all(acc[:5] == 1.0)


@pytest.mark.parametrize("sampler, n", [(metropolis.run, 200_000)])
@pytest.mark.parametrize("L, T", [(2, 2.0), (4, 2.0), (4, 3.0)])
def test_samplers_match_exact_values(sampler, n, L, T):
    """Energy, specific heat and Binder cumulant within four error bars of exact."""
    result = sampler(L, T, n, seed_value=11 * L + int(10 * T))
    e_exact, c_exact = exact.thermodynamics(L, T)
    b_exact = exact.magnetisation_moments(L, T)["binder"]
    for (value, error), target in ((result.energy(), e_exact),
                                   (result.specific_heat(), c_exact),
                                   (result.binder(), b_exact)):
        assert abs(value - target) < 4 * error + 1e-12


@pytest.mark.parametrize("sampler", [metropolis.run])
def test_recorded_series_matches_final_configuration(sampler):
    result = sampler(8, 2.3, 2000, seed_value=5, record=True)
    series = result.info["series"]
    nbr = neighbour_table(8)
    assert series[-1, 0] == energy(result.spins, nbr)
    assert series[-1, 1] == magnetisation(result.spins)


def test_starting_configuration_is_not_modified():
    start = np.ones(16, dtype=np.int8)
    metropolis.run(4, 3.0, 100, spins=start, seed_value=1)
    assert np.all(start == 1)