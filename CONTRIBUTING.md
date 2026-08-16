# Contributing

Contributions should be focused, tested, and compatible with every supported
Python version and platform affected by the change.

Before submitting a change, run:

```bash
ruff check src tests
ruff format --check src tests
mypy
pytest -q
python scripts/asset_hashes.py check
rimkit robots verify
```

Do not commit notebooks, runtime build products, absolute local paths, secrets,
or pickle-based public inputs. Changes to robot assets must preserve licensing
and provenance records. Changes to retargeting behavior should include focused
unit or integration tests.
