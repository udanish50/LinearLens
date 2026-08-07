# Contributing

Contributions are welcome when they preserve three principles:

1. **Observational behavior:** analysis must not silently mutate model parameters or replace forward outputs.
2. **Method clarity:** publication-grounded behavior and implementation-specific choices must be distinguished.
3. **Privacy:** no confidential research data, human-subject microdata, secrets, or private checkpoints may be committed.

Before opening a pull request:

```bash
python -m pip install -e '.[dev]'
ruff format .
ruff check .
pytest
python scripts/validate_release.py
```
