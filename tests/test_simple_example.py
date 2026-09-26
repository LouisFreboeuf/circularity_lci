"""Execute examples/simple_example.ipynb end-to-end and check key results.

The notebook is self-contained (no ecoinvent credentials needed): it builds a
Brightway project from scratch and runs the full pipeline
(BurdenFreeAnalyzer -> BiosphereFlowManager -> CircularityCalculator -> LCA).
"""

import re

import nbformat
import pytest
from nbclient import NotebookClient

NOTEBOOK_PATH = "examples/simple_example.ipynb"

EXPECTED_EFFICIENCIES = {
    "kilogram": 0.906208,
    "cubic meter": 1.000000,
    "megajoule": 0.946843,
}


@pytest.fixture(scope="module")
def executed_notebook():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute(cwd=".")
    return nb


def _results_df_text(nb):
    for cell in reversed(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") != "execute_result":
                continue
            text = out.get("data", {}).get("text/plain", "")
            if "efficiency (\u03b7+)" in text:
                return text
    pytest.fail("No results table found in notebook outputs")


def _efficiencies_from_df_text(text):
    """Parse the pandas-rendered table chunk containing the efficiency columns."""
    lines = text.splitlines()
    efficiencies = {}
    in_efficiency_chunk = False
    for line in lines:
        if "efficiency (\u03b7+)" in line:
            in_efficiency_chunk = True
            continue
        if in_efficiency_chunk:
            match = re.match(r"^(kilogram|cubic meter|megajoule)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
            if match:
                efficiencies[match.group(1)] = float(match.group(4))
    return efficiencies


def test_notebook_runs(executed_notebook):
    assert executed_notebook is not None


def test_circularity_efficiencies(executed_notebook):
    text = _results_df_text(executed_notebook)
    efficiencies = _efficiencies_from_df_text(text)
    assert set(efficiencies) == set(EXPECTED_EFFICIENCIES), (
        f"Parsed units {sorted(efficiencies)} != expected {sorted(EXPECTED_EFFICIENCIES)}\n{text}"
    )
    for unit, expected in EXPECTED_EFFICIENCIES.items():
        got = efficiencies[unit]
        assert got == pytest.approx(expected, abs=1e-4), (
            f"efficiency for '{unit}': got {got}, expected {expected}"
        )
