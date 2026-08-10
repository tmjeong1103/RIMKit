# Architecture

CoRe exposes one contact-aware retargeting pipeline through three interfaces:

- the `Retargeter` Python API
- the `core-retarget` command-line interface
- the FastAPI browser application started by `core-retarget serve`

All interfaces call the same `run_retarget_pipeline()` implementation. The web
adapter adds upload validation, bounded job execution, progress streaming, and
allowlisted artifact delivery without changing the motion algorithm.

## Pipeline

The external data flow is:

```text
SOMA motion -> DMR -> contact-aware refinement -> robot motion and preview
```

The runner uses the following artifact boundaries:

| Stage | Artifact | Responsibility |
|---:|---|---|
| 1 | `1_contacts.npz` | Validate the SOMA input and derive foot-contact state. |
| 2 | `2_dmr.npz` | Retarget source body targets to the selected robot. |
| 3 | `3_initial_collision.npz` | Apply initial arm self-collision refinement. |
| 4 | `4_target_trajectories.npz` | Extract root, ankle, sole, and toe trajectories. |
| 5 | `5_ara.npz` | Adjust the root trajectory and grounding offset. |
| 6 | `6_fpa_targets.npz` | Build contact-aware foot-placement targets. |
| 7 | `7_fpa_ik.npz` | Solve foot-placement IK and grounding. |
| 8 | `8_final.npz` | Apply final arm self-collision refinement. |
| 9 | `9_diagnostics.npz` | Recompute final trajectory diagnostics. |

After Stage 9, CoRe writes a `core-robot-motion-v1` NPZ containing direct
MuJoCo `qpos`, timestamps, joint names, contact data, and source/model hashes.
The final archive contains no object arrays and is validated with
`allow_pickle=False` before publication.

## Package layout

```text
core_retarget/
├── motion/       # SOMA loading, validation, contacts, and source targets
├── robots/       # robot registry and immutable model metadata
├── mujoco/       # model, kinematics, collision, and IK adapters
├── stages/       # retargeting and refinement stages
├── pipeline/     # orchestration, events, and result contracts
├── export/       # versioned robot-motion output
├── render/       # MP4 and thumbnail generation
├── web/          # FastAPI browser interface
└── assets/       # robot models, meshes, and scene wrappers
```

Algorithm modules do not depend on the CLI or web application. Robot-specific
model information is resolved through the registry and immutable profiles
rather than being constructed by interface code.

## Runtime backends

The packaged native extension accelerates MuJoCo signed-distance queries and
body-position IK. `backend="auto"` selects it when available; `native` requires
it, and `python` selects the portable Python backend. A run records the selected
backend and dependency versions in its manifest.

## Model assets

Robot XML files and meshes are immutable package data. CoRe scene wrappers live
separately under `assets/scenes`, and runtime code never rewrites vendor model
directories. Asset hashes can be checked with:

```bash
core-retarget robots verify
```
