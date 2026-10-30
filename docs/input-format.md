# SOMA source-motion formats

RIMKit accepts two source containers and selects the adapter from the filename
extension:

| Extension | Producer | Source representation |
|---|---|---|
| `.npz` | Kimodo | Evaluated SOMA77 global joint positions and rotations |
| `.pt` | GEM-X | SOMA body parameters plus static-contact logits |

Both adapters produce the same immutable SOMA77 in-memory motion before DMR.
The complete `run` pipeline and `Retargeter.preflight()` require at least two
frames because their common contact/FPA path computes temporal differences.

Use the container-aware entry points when accepting either format:

```python
from rimkit.motion import load_source_motion, validate_source_motion

summary = validate_source_motion("motion.pt")
source = load_source_motion("motion.pt")
motion = source.motion
contacts = source.build_contact_schedule()

print(summary.provider)          # "gem-x"
print(summary.container_format)  # "pt"
```

## Kimodo `.npz`

General Kimodo loading and `validate` accept one or more frames; this checks
only the storage contract. The standalone K1 and H1 DMR stage can accept one
Kimodo frame, while the complete pipeline and contact-aware temporal paths
require at least two.

### Required arrays

| Key | Shape | Meaning |
|---|---|---|
| `posed_joints` | `(T, 77, 3)` | Global SOMA joint positions |
| `global_rot_mats` | `(T, 77, 3, 3)` | Global SOMA joint rotations |

### Optional arrays

| Key | Shape | Meaning |
|---|---|---|
| `local_rot_mats` | `(T, 77, 3, 3)` | Local rotations |
| `root_positions` | `(T, 3)` | Source root position |
| `smooth_root_pos` | `(T, 3)` | Smoothed root position |
| `global_root_heading` | `(T, 2)` | Horizontal root heading |
| `foot_contacts` | `(T, 4)` or `(T, 6)` | Foot contact labels |
| `fps` | scalar | Sampling rate |

The six-channel Kimodo contact order is left heel, left toe, left toe-end,
right heel, right toe, right toe-end. The contact preprocessor treats toe and
toe-end labels as the primary source, fills only eligible gaps from source-foot
geometry, and builds the confidence ramps used by contact-aware stages.

Kimodo FPS is resolved from a user override first, then the `.npz` scalar, then
30 Hz with a warning. `.npz` files are opened with NumPy pickle support disabled.
Object arrays, non-finite values, mismatched frame counts, invalid shapes, and
excessive file sizes are rejected before a robot model is loaded.

The input positions and global rotations are left-multiplied by a +90-degree X
rotation to convert them to MuJoCo Z-up. Rotations are not conjugated or
reprojected. The source JOI layer then derives its 22 body targets directly
from those global arrays. In particular, `base` uses the midpoint of `LeftLeg`
and `RightLeg` while retaining the `Hips` rotation, and `neck` uses the shoulder
midpoint with the `Neck1` rotation. No stance-width, hip-width, or
robot-relative normalization is applied before DMR.

The Kimodo-specific low-level entry points remain available:

```python
from rimkit.motion import build_contact_schedule, extract_soma_joi, load_soma_motion

motion = load_soma_motion("motion.npz")
contacts = build_contact_schedule(motion)
source_joi = extract_soma_joi(motion)
```

The regression suite preserves exact source/contact and JOI outputs for the
bundled Kimodo reference motions. Those baselines do not extend to GEM-X input
or guarantee contact quality for arbitrary motions.

## GEM-X `.pt`

Install PyTorch support for command-line or Python use with the `gemx` extra:

```bash
python -m pip install -e ".[gemx]"
```

The `web` extra includes PyTorch because the browser upload path accepts both
source formats.

The primary GEM-X payload contract is:

```text
body_params_global/
  body_pose              # (T, 228) or (T, 76, 3)
  global_orient          # (T, 3), with an accepted singleton joint axis
  transl                 # (T, 3)
net_outputs/
  static_conf_logits     # (T, 6) or (1, T, 6)
```

For compatibility with GEM-X result variants, body parameters may instead be
stored at `net_outputs.pred_body_params_global`; contact logits may also be
nested at `net_outputs.model_output.static_conf_logits`. All three body
parameters and the six static-contact logits are required. The prediction
fallback may retain a leading singleton inference-batch axis `(1, T, ...)`,
which the adapter removes before shape validation. The contact-logit order is
left ankle, left foot, right ankle, right foot, left wrist, right wrist; RIMKit
uses the left/right foot channels as its toe-contact source.

`.pt` input is loaded on CPU with
`torch.load(..., map_location="cpu", weights_only=True)`. RIMKit never falls
back to unrestricted pickle deserialization. A PyTorch release that cannot
perform weights-only loading is rejected, as are non-mapping roots, unknown
shapes, non-real or non-finite values, missing fields, oversized files, and
excessive frame counts.

GEM-X `.pt` does not carry an FPS field. A user `--fps`/`fps_override` value takes
precedence; otherwise RIMKit uses 30 Hz and records a warning. GEM-X input always
requires at least two frames.

The adapter follows this fixed preprocessing order:

1. Evaluate body parameters with the packaged fixed SOMA77 bind rig.
2. Rotate positions and global rotations by +90 degrees about X into Z-up.
3. Fuse the static toe-contact logits with toe height and velocity geometry.
4. Estimate a time-varying support floor from the toe contacts and subtract it
   from every SOMA77 joint for each frame.

The normalized motion then enters the same typed stage pipeline, with the
GEM-X source profile selected for the chosen robot.

## Output is always robot-motion `.npz`

The source extension does not change the result format. RIMKit writes
`final/robot_motion.npz` using the `core-robot-motion-v1` contract, validates
it with `allow_pickle=False`, and stores no Python objects or pickle payloads.
The manifest records the input container, source provider, source hash, and FPS
so the selected preprocessing path is explicit.

Robot-motion **schema version 2** retains that container format and preserves
production CATO metadata. `cato_segment_frames` is an integer `(M, 2)` contiguous
half-open frame partition; `cato_segments` is its `(M+1,)` boundary vector,
normalized using the existing `round(frame_index / frame_count, 8)` convention.
The alpha dimension is `M`. These coarse macro segments differ from
`left_contact_segments` / `right_contact_segments`, which encode per-foot
source intended-contact intervals. None of these arrays measures a policy's
realized collision/force contacts.

The renderer and exporter share `rimkit.motion.cato_segments` without changing
the original algorithm. Pipeline export carries the exact preview partition
computed from **stage-7 FPA qpos**, rather than recomputing it after final
collision correction. `cato_segments_provenance` is a Unicode JSON scalar with
algorithm/version, feature weights, smoothing, config, input source hash,
contact source, RIMKit commit, worktree state, and implementation hash. The
manifest repeats that record as `cato_macro_segmentation`.

`contact_labels`, `contact_confidence`, `contact_availability`, `flight_labels`
and per-foot intervals are preserved. `contact_annotation_status` defaults to
`unreviewed`; a metadata export does not approve annotations. Legacy schema-1
archives contain no production macro partition and cannot silently select a
literal CATO partition when used scientifically. A separate metadata upgrade
may reuse existing labels and the unchanged algorithm, preserving the original
archive and its provenance. ReadyPose preparation is outside this source-motion
artifact and has no automatically supplied contact labels.
