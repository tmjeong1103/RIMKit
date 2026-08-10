# SOMA input format

CoRe accepts NumPy NPZ files containing Kimodo SOMA77 motion.

General SOMA loading and `validate` accept one or more frames; this checks only
the storage contract. The complete `run` pipeline and `Retargeter.preflight()`
require at least two frames because their common contact/FPA path computes
temporal differences. The standalone K1 and H1 DMR stage can accept one frame,
while G1, H2, and R1 DMR itself requires at least two for contact-aware
preprocessing.

## Required arrays

| Key | Shape | Meaning |
|---|---|---|
| posed_joints | (T, 77, 3) | Global SOMA joint positions |
| global_rot_mats | (T, 77, 3, 3) | Global SOMA joint rotations |

## Optional arrays

| Key | Shape | Meaning |
|---|---|---|
| local_rot_mats | (T, 77, 3, 3) | Local rotations |
| root_positions | (T, 3) | Source root position |
| smooth_root_pos | (T, 3) | Smoothed root position |
| global_root_heading | (T, 2) | Horizontal root heading |
| foot_contacts | (T, 4) or (T, 6) | Foot contact labels |
| fps | scalar | Sampling rate |

The six-channel Kimodo contact order is left heel, left toe, left toe-end,
right heel, right toe, right toe-end.

The contact preprocessor treats the toe and toe-end labels as the primary
source, fills only eligible gaps from source-foot geometry, and builds
the confidence ramps consumed by contact-aware DMR. G1, H2, and R1 invoke this
stage automatically; callers can inspect the same immutable result directly:

```python
from core_retarget.motion import build_contact_schedule, load_soma_motion

motion = load_soma_motion("motion.npz")
contacts = build_contact_schedule(motion)
```

FPS is resolved from a user override first, then an NPZ scalar, then the
Kimodo example default of 30 Hz with a warning.

Files are loaded with NumPy pickle support disabled. Object arrays, non-finite
values, mismatched frame counts, invalid shapes, and excessive file sizes are
rejected before any model is loaded.

The Kimodo examples are converted to MuJoCo Z-up at runtime with a +90 degree
rotation about the X axis. Positions and global rotations are both
left-multiplied by this world rotation; rotations are not conjugated or
reprojected. This conversion must be recorded in run.json.

The source JOI layer constructs 22 body targets directly from those global
arrays. In particular, `base` uses the
midpoint of `LeftLeg` and `RightLeg`, while retaining the `Hips` rotation, and
`neck` uses the shoulder midpoint with the `Neck1` rotation. No stance-width,
hip-width, or robot-relative normalization is applied before DMR.

The equivalent source-JOI Python entry point is:

```python
from core_retarget.motion import extract_soma_joi, load_soma_motion

motion = load_soma_motion("motion.npz")
source_joi = extract_soma_joi(motion)
```
