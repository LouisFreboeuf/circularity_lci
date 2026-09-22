# Circularity LCI

A Python package for **circularity** and **life cycle assessment (LCA)** analysis built on top of [Brightway](https://brightway.dev/).

It provides tools for analyzing burden-free activities, duplicating technosphere flows to biosphere flows, calculating circularity indicators (aligned with ISO 59020), building LCIA methods based on circularity variables, and running them across case studies and entire LCA databases. Long-running calculations can be tracked with a built-in progress tracker.

## Installation

```bash
pip install circularitylci
```

The package depends on the Brightway ecosystem (`bw2data`, `bw2calc`, `bw2io`) plus `numpy`, `pandas`, `matplotlib`, `lxml`, and `stats_arrays`, which are installed automatically.

## Usage

The canonical import name is `circularity_lci` (the package is also mirrored
under the alias `circularitylci`, so `import circularitylci` works too).

```python
import circularity_lci as clci
# or, equivalently:
import circularitylci as clci

from circularity_lci import (
    BurdenFreeAnalyzer,
    BiosphereFlowManager,
    CircularityCalculator,
    MultiLCACalculator,
    CircularityDatabaseAnalyzer,
    LCIAMethodBuilder,
    ProgressTracker,
)
```

Verbose progress output from long-running operations goes through the
standard `logging` module; control it with the usual configuration, e.g.
`logging.basicConfig(level=logging.INFO)`.

Worked examples are available in the [`examples/`](examples) directory:

- `examples/simple_example.ipynb` &mdash; minimal end-to-end run.
- `examples/case_study.ipynb` &mdash; a full case study.
- `examples/main.ipynb` &mdash; the main workflow walkthrough.

## Features

- **Burden-free activity analysis** &mdash; identify and handle burden-free exchanges.
- **Biosphere flow management** &mdash; duplicate a list of technosphere flows to biosphere flows.
- **Circularity indicators** &mdash; compute circularity efficiency variables and indicators.
- **LCIA method builder** &mdash; create circularity LCIA methods (`create_circularity_lcia_methods`).
- **Multi-LCA calculator** &mdash; run LCAs across many activities or databases.
- **Database analyzer** &mdash; apply the workflow to a whole LCA database.
- **Progress tracking** &mdash; persist progress for long-running LCA projects.

## Development

```bash
git clone https://github.com/LouisFreboeuf/circularity_lci
cd circularity_lci
pip install -e ".[dev]"
```

Run tests with `pytest` from the repository root.

## License

MIT &mdash; see [LICENSE](LICENSE).

## Citation

If you use this software, please cite it (see [CITATION.cff](CITATION.cff)).
