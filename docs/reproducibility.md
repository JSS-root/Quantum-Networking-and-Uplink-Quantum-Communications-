# Reproducibility Notes

## Current Included Artifacts

- `src/uplink_qkd/finite_key.py`: finite-key optimisation routines based on Lim et al.
- `src/uplink_qkd/rates.py`: coincidence and raw-overpass key-rate models.
- `data/processed/*.csv`: processed optimisation outputs used by manuscript figure scripts.
- `notebooks/*.ipynb`: table-generation notebooks.
- `scripts/figures/*.py`: scripts for manuscript and supplementary figures.

## Known Data Gaps

The script `scripts/figures/figS4.py` references a loss-profile directory named `15deg_500000m_0m_0.25m/`, which is not currently included in this repository. Before public release, either add the required data, add a generator script, or update the script to point to an archived data dependency.

## Release Checklist

- Confirm the final repository name and GitHub URL.
- Confirm the open-source license.
- Update `CITATION.cff` with final paper metadata, arXiv identifier, and DOI.
- Add the final manuscript/software DOI once available.
- Verify all scripts run from a clean checkout.
- Create a GitHub release and archive it with Zenodo or an equivalent DOI provider.
- Consider publishing the Python package to PyPI after API names have stabilised.
