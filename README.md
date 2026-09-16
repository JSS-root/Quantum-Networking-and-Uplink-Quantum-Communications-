# Uplink QKD Viability Analysis

This repository contains Python code and supporting data for modelling the viability of entanglement-based satellite-uplink quantum key distribution (QKD), with emphasis on finite-key optimisation and link-level key-rate modelling for BBM92-style entanglement-based protocols.

The repository supports an associated manuscript and is intended to support reproducible figures, tables, and future archival citation through GitHub, PyPI, arXiv, and DOI-backed software archiving.

## Release version

Version 1.1.0 release: September 2026.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22795420.svg)](https://doi.org/10.5281/zenodo.22795420)

## Repository Layout

```text
uplink-qkd/
├── .github/
|   ├── ISSUE_TEMPLATE/
|   |   ├── bug_report.md              Bug-report template
|   |   └── reproducibility.md         Reproducibility-issue template
|   └── workflows/
|       └── tests.yml                  GitHub Actions test workflow
├── data/
|   └── processed/                     Processed optimisation outputs used by figure scripts
├── docs/
|   └── reproducibility.md             Notes on reproducibility and known data gaps
├── notebooks/                         Exploratory and table-generation notebooks
├── scripts/
|   └── figures/                       Manuscript and supplementary figure scripts
├── src/
|   └── uplink_qkd/                    Reusable Python package code
|       ├── __init__.py                Public package interface
|       ├── finite_key.py              Finite-key BBM92 optimisation routines
|       └── rates.py                   Coincidence and raw-overpass key-rate models
├── tests/                             Lightweight regression and import tests
├── CITATION.cff                       Software citation metadata
├── LICENSE                            Project license
├── MANIFEST.in                        Source-distribution file manifest
├── README.md                          Project overview
└── pyproject.toml                     Python packaging and tool configuration
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

To install and use the Python version of this repository:

- Clone the repository.
- Navigate to the repository root folder.
- Install the package locally using `pip`.
- Run the desired Python module, figure script, or notebook.

In Terminal (macOS), Bash (Linux/Unix), or Cygwin/Command Prompt (Windows), run:

```bash
git clone https://github.com/JSS-root/uplink-qkd.git
cd uplink-qkd
python -m pip install -e .
```

For development, including testing and notebook support, run:

```bash
python -m pip install -e ".[dev,notebooks]"
```

## Development Roadmap

`UplinkQKD` is still a work in progress. Future and ongoing developments are listed below:

- Dedicated documentation
- Implement downlink version (https://arxiv.org/abs/2602.11833)
- Integration with SatQuMA (https://github.com/cnqo-qcomms/SatQuMA) and SatQuMA_UI (https://github.com/JSS-root/SatQuMA_UI)
- Multiscale search (coarse grid, refine) or adaptive sampling would dramatically speed up computation runtime.
- Efficient postprocessing for improved key rates and noise robustness.


## Quick Start

The smart_optimise function performs a coarse brute-force search followed by local refinement over the optimisation parameters to compute the secret-key-length ratio for a given block size and QBER. 

We recommend starting with granularity=50 for fast exploration. The speed/accuracy trade-off for higher granularity values can be increased up to granularity=300 for publication-quality results. Example below:

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

## Attribution

If you use this software, please cite as follows:

```
@software{Sidhu_quantum_networking_2026,
  author    = {Sidhu, Jasminder S.},
  title     = {{Quantum Networking and Uplink Quantum Communications}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22795420},
  url       = {[https://doi.org/10.5281/zenodo.22795420](https://doi.org/10.5281/zenodo.22795420)}
}
```

Citation metadata can also be seen in [CITATION.cff](CITATION.cff).

## Author Contact

Enquiries and requests can be sent to: 
[Jasminder S. Sidhu](mailto:jsmdrsidhu%40gmail.com)

## Related Work

This repository focuses on satellite-uplink geometry and complements related work on finite-key optimisation and satellite QKD configurations, including the Mathematica implementation in [JSS-root/BBM92_Finite_Key](https://github.com/JSS-root/BBM92_Finite_Key).

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).
