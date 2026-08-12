# Development

CoRe supports Python 3.10 through 3.13 on macOS and Linux. Building from source
requires a C++17 compiler, CMake, and Ninja.

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,web]"
core-retarget backend --require-native
```

The `web` extra includes rendering and PyTorch support, so both Kimodo `.npz`
and GEM-X `.pt` inputs are available in the development server.

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

Validate either source format directly:

```bash
core-retarget validate \
  examples/motions/kimodo/soma_rp_v11/stand_walk_run_stop.npz
core-retarget validate examples/motions/gem-x/rapid_stepping.pt --fps 30
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

Run a bundled GEM-X motion by passing its `.pt` sampling rate explicitly:

```bash
core-retarget run \
  examples/motions/gem-x/rapid_stepping.pt \
  --robot g1 \
  --fps 30 \
  --output runs/development-check-gemx \
  --video \
  --thumbnail
```

Generate the complete Kimodo gallery with:

```bash
python scripts/generate_example_outputs.py \
  --source-set kimodo \
  --output runs/example-outputs \
  --gallery-dir docs/media/final \
  --resume
```

Generate the equivalent GEM-X gallery at 30 Hz with:

```bash
python scripts/generate_example_outputs.py \
  --source-set gem-x \
  --output runs/gem-x-example-outputs \
  --gallery-dir docs/media/final \
  --resume
```

## Packages and container

```bash
python -m build
docker build --tag core-retarget:dev .
```

Do not commit local environments, build outputs, runtime result directories,
`docs/media/size_modified/`, absolute machine paths, secrets, or untrusted
pickle payloads. Only release-intended media with an explicit review status
belongs under `docs/media/final/`. New robot assets must include their license,
source record, and file hashes.
