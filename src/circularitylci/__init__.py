"""Alias package so `circularitylci` can be imported as well as `circularity_lci`.

Re-exports everything from :mod:`circularity_lci` without mutating built-in
module attributes, which keeps import tooling and PyPI installs safe.
"""
from circularity_lci import *  # noqa: F401,F403
from circularity_lci import __all__  # noqa: F401
