"""Error bars from block averages.

The samplers split a run into equal blocks and return, for each block, the
block means of e, e^2, |m|, m^2 and m^4 (e = E/N, m = M/N).  When a block is
much longer than the autocorrelation time, block means are nearly
independent, so their spread gives honest error bars.  Quantities that are
not plain averages (specific heat, Binder cumulant) get their error bars from
the jackknife over blocks.
"""

from dataclasses import dataclass, field

import numpy as np

E1, E2, ABS_M, M2, M4 = range(5)
COLUMNS = ("e", "e^2", "|m|", "m^2", "m^4")


def jackknife(blocks, estimator):
    """Estimate and jackknife error of estimator(mean of each column)."""
    n = blocks.shape[0]
    total = blocks.sum(axis=0)
    estimate = estimator(total / n)
    leave_one_out = np.array([estimator((total - blocks[i]) / (n - 1)) for i in range(n)])
    error = np.sqrt((n - 1) / n * np.sum((leave_one_out - leave_one_out.mean()) ** 2))
    return float(estimate), float(error)


@dataclass
class Result:
    """Output of one Monte Carlo run."""

    blocks: np.ndarray  # shape (n_blocks, 5): block means of e, e^2, |m|, m^2, m^4
    beta: float
    n_sites: int
    spins: np.ndarray  # final configuration, to restart from
    info: dict = field(default_factory=dict)

    def energy(self):
        """Energy per spin <e> and its error."""
        return jackknife(self.blocks, lambda b: b[E1])

    def specific_heat(self):
        """Specific heat per spin, beta^2 N (<e^2> - <e>^2), and its error."""
        return jackknife(self.blocks, lambda b: self.beta**2 * self.n_sites * (b[E2] - b[E1] ** 2))

    def abs_magnetisation(self):
        """<|m|> and its error."""
        return jackknife(self.blocks, lambda b: b[ABS_M])

    def binder(self):
        """Binder cumulant B = (3 - <m^4>/<m^2>^2) / 2, Krauth's convention."""
        return jackknife(self.blocks, lambda b: 0.5 * (3.0 - b[M4] / b[M2] ** 2))


def deviation(value, error, exact):
    """Distance from the exact value in units of the error bar."""
    return (value - exact) / error if error > 0 else np.inf