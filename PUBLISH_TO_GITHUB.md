# Publish to GitHub

The repository includes `scripts/publish_github.sh`, which checks the currently authenticated GitHub account before creating or updating the target repository.

Recommended target:

```text
https://github.com/udanish50/LinearLens
```

Before publishing, run:

```bash
ruff format .
ruff check .
pytest
python scripts/validate_release.py
```
