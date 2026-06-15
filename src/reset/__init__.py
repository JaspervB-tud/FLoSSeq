# src/__init__.py

# This file marks the directory as a Python package.

from .solution import Solution, Solution_shm
from importlib.metadata import version as _pkg_version

__all__ = ["Solution", "Solution_shm"]

__version__ = _pkg_version("reset-bio")
