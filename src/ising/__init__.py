"""Monte Carlo algorithms for the 2D Ising model (Krauth, chapter 5)."""

from .lattice import T_C, energy, magnetisation, neighbour_table, random_spins

__all__ = ["T_C", "energy", "magnetisation", "neighbour_table", "random_spins"]