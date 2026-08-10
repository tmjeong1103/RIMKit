# Development

CoRe supports Python 3.10 through 3.13 on macOS and Linux. Building from source
requires a C++17 compiler, CMake, and Ninja.

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,web,video]"
core-retarget backend --require-native
```

## Checks

Run the normal product test suite and static checks before submitting a change:

```bash
pytest -q
ruff check src tests
ruff format --check src tests
mypy
python scripts/asset_hashes.py check
core-retarget robots verify
```

Run one bundled motion end to end:

```bash
core-retarget run \
  examples/motions/kimodo/soma_rp_v11/stand_walk_run_stop.npz \
  --robot g1 \
  --output runs/development-check \
  --video \
  --thumbnail
```

Generate the complete final example gallery with:

```bash
python scripts/generate_example_outputs.py \
  --output runs/example-outputs \
  --gallery-dir docs/media/final \
  --resume
```

## Packages and container

```bash
python -m build
docker build --tag core-retarget:dev .
```

Do not commit local environments, build outputs, runtime result directories,
absolute machine paths, secrets, or pickle-based motion files. New robot assets
must include their license, source record, and file hashes.
