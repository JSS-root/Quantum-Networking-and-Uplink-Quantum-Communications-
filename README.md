# Uplink QKD Viability

This repository contains Python code and supporting data for modelling the viability of satellite-uplink quantum key distribution (QKD), with emphasis on finite-key optimisation and link-level key-rate modelling for BBM92-style entanglement-based protocols.

The repository is being prepared alongside an associated manuscript draft and is intended to support reproducible figures, tables, and future archival citation through GitHub, PyPI, arXiv, and DOI-backed software archiving.

## Status

This codebase is under active preparation. The repository should remain private until the manuscript and software metadata have been reviewed.

## Repository Layout

```text
uplink-qkd/
|-- .github/
|   ├-- ISSUE_TEMPLATE/
|   |   ├-- bug_report.md              Bug-report template
|   |   ├-- reproducibility.md         Reproducibility-issue template
|   ├-- workflows/
|       ├-- tests.yml                  GitHub Actions test workflow
|-- data/
|   ├-- processed/                     Processed optimisation outputs used by figure scripts
|-- docs/
|   ├-- reproducibility.md             Notes on reproducibility and known data gaps
|-- notebooks/                         Exploratory and table-generation notebooks
|-- scripts/
|   ├-- figures/                       Manuscript and supplementary figure scripts
|-- src/
|   ├-- uplink_qkd/                    Reusable Python package code
|       ├-- __init__.py                Public package interface
|       ├-- finite_key.py              Finite-key BBM92 optimisation routines
|       ├-- rates.py                   Coincidence and raw-overpass key-rate models
├-- tests/                             Lightweight regression and import tests
├-- CITATION.cff                       Software citation metadata
├-- LICENSE                            Project license
├-- MANIFEST.in                        Source-distribution file manifest
├-- README.md                          Project overview
├-- pyproject.toml                     Python packaging and tool configuration
```

## Dependencies and Requirements

The package requires Python `3.10` or newer.

Core Python dependencies:

- `numpy >= 1.24`
- `scipy >= 1.10`
- `matplotlib >= 3.7`
- `tqdm >= 4.65`

Development and testing dependencies:

- `pytest >= 8.0`
- `ruff >= 0.5`
- `build >= 1.2`

Notebook support:

- `jupyter >= 1.0`

## Installation

For development from a local checkout:

```bash
python -m pip install -e ".[dev,notebooks]"
```

For a minimal local install:

```bash
python -m pip install -e .
```

## Quick Start

```python
import numpy as np
from uplink_qkd import smart_optimise

m = 1e6
qber = 0.05
eps_qkd = 1e-6
t = np.log2(10**8)
error_correction_efficiency = 1.19

secret_fraction = smart_optimise(
    m,
    qber,
    eps_qkd,
    t,
    error_correction_efficiency,
    granularity=50,
)
print(secret_fraction)
```

## Reproducing Outputs

Figure scripts are in `scripts/figures/`. Scripts that use the processed CSV files should be run from the repository root so relative paths resolve correctly.

Some supplementary scripts require external or generated loss-profile data that is not currently present in this repository. See [docs/reproducibility.md](docs/reproducibility.md) for the current data inventory and remaining reproducibility tasks.

## Citation

If you use this software, please cite the associated paper and software record. Citation metadata is provided in [CITATION.cff](CITATION.cff) and should be updated with the final arXiv identifier and DOI once available.

## Related Work

This repository focuses on satellite-uplink geometry and complements related work on finite-key optimisation and satellite QKD configurations, including the Mathematica implementation in [JSS-root/BBM92_Finite_Key](https://github.com/JSS-root/BBM92_Finite_Key).

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).
