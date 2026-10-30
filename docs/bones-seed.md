# BONES-SEED SOMA BVH conversion

Convert SOMA Uniform or SOMA Proportional BVH into the evaluated SOMA77 NPZ
accepted by the existing RIMKit pipeline. The source BVH is never modified.

Project default: use motions from `soma_uniform/bvh/` for BONES-SEED conversion,
retargeting, and batch processing. Use `soma_proportional/bvh/` only when
explicitly requested. This preference was established on 2026-10-27 after
comparing the same walking motion on K1.

For contact-aware timing test selection, use original clips no longer than
20 seconds that travel across the floor and have naturally stationary starts
and ends. The current screen measures the first and last 0.4 seconds with
mocap-noise tolerance; it does not trim, freeze, pad, or stitch clips. Select
one representative per action after normalizing actor/take, mirror, direction,
speed, performance style, and explicit version/variation labels.

Prop or partner mentions do not exclude a motion: kicking trash, carrying an
imaginary object, sweeping while walking, and opening an imaginary door are
allowed as pantomime. Exclude actions that need a physical support surface or
apparatus, such as stairs, ladders, jumping down from a box, or sitting on
furniture. A nonzero `content_props` tag alone is not an exclusion rule.
The revised local catalogs and selection audits are in
`runs/bones_seed_selection_v3/`; robot/contact quality still needs validation
after retargeting.

Run the commands below from the RIMKit repository with its dependencies installed
(including the `video` extra for MP4 rendering).

```bash
PYTHONPATH=src python scripts/convert_bones_seed_bvh.py \
  /home/rilab/datasets/bones-seed/soma_uniform/bvh/221012/walk_ff_loop_360_001__A053.bvh \
  runs/bones_seed_demo/walk_soma77_30fps.npz \
  --target-fps 30 --ground

PYTHONPATH=src MUJOCO_GL=egl python -m rimkit run \
  runs/bones_seed_demo/walk_soma77_30fps.npz \
  --robot k1 --output runs/bones_seed_demo/k1 \
  --backend python --video --thumbnail
```

The converter:

- Validates joint names and parents against SOMA77, tolerating the additional
  `Root` wrapper and different sibling traversal orders.
- Evaluates FK with the BVH's own offsets and intrinsic Euler channel order.
  BONES-SEED position channels contain absolute local positions and replace
  `OFFSET`; adding both would incorrectly double the pelvis height.
- Composes the wrapper root before selecting the 77 SOMA joints.
- Converts centimeters to meters and retains Y-up. The existing NPZ loader
  applies the single conversion to MuJoCo Z-up.
- Applies the same joint-frame correction as Kimodo's `to_standard_tpose`:
  `global_standard = global_bvh @ global_rot_offsets.T`. The small, packaged
  numeric offset asset has its source and hashes in `assets/soma/SOURCE.yaml`.
  No PyTorch or parametric SOMA body model is needed at conversion time.
- Optionally resamples positions by interpolation and rotations by SLERP at
  the original timestamps, preserving playback speed. Omitting `--target-fps`
  retains the source's frame time. A shorter final partial interval is omitted.
- Optionally subtracts the first-percentile toe height with `--ground`, suitable
  for the approximately flat-ground walking example.
- Saves numeric/Unicode arrays with no pickle and includes source SHA256,
  conversion settings, frame counts, and calibration hash in `conversion_metadata`.

The resulting file contains `posed_joints (T,77,3)`,
`global_rot_mats (T,77,3,3)`, `root_positions (T,3)`, `fps`, `joint_names`, and
`conversion_metadata`. Outputs are never overwritten. By default, contacts are
absent and RIMKit uses its geometric contact inference. `--estimate-contacts`
instead writes six named source foot-point support estimates, their source
name, and JSON provenance. This optional estimator uses foot-point height and
3D speed relative to each point's low-quantile floor proxy; it records thresholds,
smoothing, channel names, implementation SHA, and original BVH SHA. These are
kinematic estimates, not measured forces or human-validated contact labels.
The K1 CATO-WM batch uses this explicit source-estimation mode so its contacts
have a recorded origin. The manifest calls this the `kimodo` provider because that selects the
NPZ adapter; actual BONES-SEED origin is recorded in `conversion_metadata`.

Run or resume the selected 173-motion K1 batch with
`PYTHONPATH=src python scripts/retarget_bones_seed_k1.py --workers 12` after
building the standard native accelerator. Use `--backend python` when the
extension is unavailable. The output defaults to the sibling CATO-WM project's
`resources/motions/bones-seed-k1/`, with per-motion source NPZ, full K1 stages,
final NPZ, video/thumbnail, and 30/50 Hz MimicKit tables. Completed output hashes
are checked before skipping a motion. The CATO-WM packaging script writes the
pinned additive catalog, per-motion WM configurations, and video gallery.

The adapter intentionally targets the BONES-SEED/SOMA BVH conventions. Other
BVH dialects can differ in position-channel semantics and joint frames. The
K1 preview is playback of the retargeted kinematic trajectory, rather than a
demonstration of a dynamically balanced controller.

```bash
PYTHONPATH=src python -m pytest -q tests/unit/test_bvh.py \
  tests/unit/test_soma.py tests/unit/test_soma_joints.py
```
