# Migrating from CoRe 0.1

RIMKit 0.2 changes the canonical package and command names while retaining
compatibility for the public CoRe 0.1 entry points.

| CoRe 0.1 | RIMKit 0.2 |
|---|---|
| `pip install core-retarget` | `pip install rimkit` |
| `core-retarget ...` | `rimkit ...` |
| `python -m core_retarget ...` | `python -m rimkit ...` |
| `from core_retarget import Retargeter` | `from rimkit import Retargeter` |

The old top-level import and command remain available as compatibility aliases.
Internal modules under `core_retarget.*` were never part of the stable public
API and should be imported from `rimkit.*`.

The rebrand does not change the CoRe numerical pipeline or result format. The
default method remains `core`; it can also be selected explicitly with
`rimkit run --method core ...`.
