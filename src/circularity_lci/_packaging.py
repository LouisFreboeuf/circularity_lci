"""Helpers to locate packaged data files independent of the CWD.

Data files shipped under ``circularity_lci/data`` (e.g. the BAFU elementary
flows mapping CSV) must be resolvable after the package is installed, from any
working directory. This module centralises that resolution.
"""

from __future__ import annotations

from pathlib import Path


_DATA_DIR = Path(__file__).resolve().parent / "data"


def get_data_path(*parts: str) -> Path:
    """Return an absolute filesystem path to a packaged data file.

    ``parts`` are path segments relative to the ``circularity_lci/data``
    directory, e.g. ``get_data_path("elementary_flows_mapping.csv")``.

    Resolution is based on the location of this module, so it works after the
    package is installed regardless of the current working directory.
    """
    return _DATA_DIR.joinpath(*parts)
