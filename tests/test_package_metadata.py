"""Check that the installed circularitylci distribution is consistent with the repo.

Runs against whatever `circularity_lci` is importable:
- In the regular build/test workflow this is the local (editable) install.
- In the release-verification job it is the package actually published on
  PyPI (https://pypi.org/project/circularitylci/), so a failed publish or a
  forgotten version bump is caught immediately.
"""

import importlib.metadata
import tomllib
from pathlib import Path


def test_installed_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    repo_version = tomllib.load(pyproject.open("rb"))["project"]["version"]
    installed_version = importlib.metadata.version("circularitylci")
    assert installed_version == repo_version, (
        f"Installed circularitylci is {installed_version} but the repo "
        f"declares {repo_version}. If this is the release-verification run, "
        "the PyPI package does not match this repository state."
    )


def test_published_package_exposes_canonical_api():
    import circularity_lci

    for name in (
        "BurdenFreeAnalyzer",
        "BiosphereFlowManager",
        "CircularityCalculator",
        "MultiLCACalculator",
        "CircularityDatabaseAnalyzer",
        "LCIAMethodBuilder",
        "ProgressTracker",
    ):
        assert hasattr(circularity_lci, name), f"Missing public API: {name}"
