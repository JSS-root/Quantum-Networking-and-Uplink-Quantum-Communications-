# Uplink QKD Viability

This repository contains Python code and supporting data for modelling the viability of satellite-uplink quantum key distribution (QKD), with emphasis on finite-key optimisation and link-level key-rate modelling for BBM92-style entanglement-based protocols.

The repository is being prepared alongside an associated manuscript draft and is intended to support reproducible figures, tables, and future archival citation through GitHub, PyPI, arXiv, and DOI-backed software archiving.

## Status

This codebase is under active preparation. The repository should remain private until the manuscript and software metadata have been reviewed.

## Repository Layout

```text
src/uplink_qkd/      Reusable Python package code
scripts/figures/     Scripts used to reproduce manuscript and supplementary figures
data/processed/      Processed optimisation outputs used by figure scripts
notebooks/           Exploratory and table-generation notebooks
tests/               Lightweight regression and import tests
docs/                Notes for reproducibility and scientific context
```

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
