# Contributing

This repository is currently being prepared as the reference implementation for an associated research manuscript. Until public release, contributions should be coordinated with the PI.

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
