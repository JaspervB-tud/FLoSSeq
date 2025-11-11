# src/__init__.py

# This file marks the directory as a Python package.
# You can use this file to initialize the package or define package-level imports.

# Example: Importing modules or functions to make them accessible at the package level
# from .module_name import some_function, SomeClass
from .solution import Solution, SolutionAverage

__all__ = ["Solution", "SolutionAverage"]

__version__ = "0.1.0"
