# Contributing

This repository is under active development and twinned with the reference article: implementation for an associated research manuscript:

Optimised finite key in satellite up-link quantum communication with entangled photons, T. Jaeken, F. Redza, D. Oi, A. Fedrizzi, and J. S. Sidhu.

Collaborations welcome. Please review ongoing and planned development roadmap.  

## Development Setup

```bash
python -m pip install -e ".[dev,notebooks]"
pytest
ruff check .
```

## Expectations

- Keep scientific changes traceable to the model, manuscript, or documented assumptions.
- Add tests for reusable package behaviour where practical.
- Keep generated outputs separate from source code.
- Document any external data needed to reproduce figures or tables.
