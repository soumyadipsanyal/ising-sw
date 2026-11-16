import numpy as np
import pytest

from ising.stats import Result, deviation, jackknife


def test_jackknife_of_a_mean_is_the_standard_error():
    rng = np.random.default_rng(0)
    blocks = rng.normal(size=(50, 5))
    value, error = jackknife(blocks, lambda b: b[0])
    assert value == pytest.approx(blocks[:, 0].mean())
    assert error == pytest.approx(blocks[:, 0].std(ddof=1) / np.sqrt(50))


def test_specific_heat_and_binder_from_identical_blocks():
    """Identical blocks: the estimates follow from the formulas, and the errors vanish."""
    e, m = -1.5, 0.5
    row = [e, e * e + 0.01, abs(m), m * m, m**4]
    result = Result(np.tile(row, (10, 1)), beta=0.5, n_sites=16,
                    spins=np.ones(16, dtype=np.int8))
    c, c_err = result.specific_heat()
    assert c == pytest.approx(0.5**2 * 16 * 0.01)
    assert c_err == pytest.approx(0.0, abs=1e-12)
    B, _ = result.binder()
    assert B == pytest.approx(1.0)  # <m^4> / <m^2>^2 = 1 when |m| never changes


def test_deviation_counts_error_bars():
    assert deviation(1.2, 0.1, 1.0) == pytest.approx(2.0)