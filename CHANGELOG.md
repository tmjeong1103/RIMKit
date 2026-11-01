# Changelog

All notable changes to RIMKit will be documented in this file.

## Unreleased

- Accept BONES-SEED SOMA `.bvh` uploads in the browser demo. The server converts
  each upload with the gallery settings and deletes the raw BVH; no BONES-SEED
  motion is bundled.
- Solve DMR arms in a separate stage from the realized shoulders, with each
  elbow hinge aligned to the source bend plane, and limit arm joints to
  15 rad/s so a singular shoulder cannot swing them by over a radian in one
  frame. Sign-continuous quaternion orientation smoothing is the default for
  every source.
- Preserve fast arm motion: stage-3 collision refinement now smooths only its
  arm corrections instead of the whole arm trajectory. Regenerate the Kimodo,
  GEM-X, and BONES-SEED gallery media.
- Add BONES-SEED SOMA BVH conversion to the SOMA77 `.npz` contract, with SOMA
  Uniform as the recommended source, optional kinematic foot-contact
  estimates, and `--source-set bones-seed` batch generation from a licensed
  local dataset copy. No BONES-SEED motion data is bundled.
- Export robot-motion schema version 2 with the CATO macro-segment partition
  computed from stage-7 FPA qpos, its provenance, and the intended-contact
  annotation status; share one segmentation implementation with the renderer.
- Reduce Hugging Face Space preview cost with a 15 FPS cap and 640×360 default
  while preserving local-demo defaults and the full source timeline in exported
  robot-motion data.
- Add Fourier GR3, AgiBot A3 T3.0, and Booster T2 support for Kimodo and GEM-X;
  migrate T1 to the latest pelvis-root Booster model with position-only wrist
  targets; preserve the original T1/T2 vendor joint order in exported motions.
- Add Asimov-1 and AgiBot X2-Ultra assets and verified Kimodo/GEM-X retargeting
  profiles, expanding the public robot registry to thirteen targets.
- Rebrand the toolkit as RIMKit, make `rimkit` the canonical Python package
  and CLI, retain `core_retarget` and `core-retarget` compatibility entry
  points, and expose CoRe as the currently available method.

## 0.1.0 - 2026-08-12

- Provide contact-aware SOMA motion retargeting for eleven humanoid robots.
- Expose a shared Python API, command-line interface, and browser demo.
- Package the native C++ MuJoCo backend and a portable Python backend.
- Export versioned, pickle-free robot-motion `.npz` files and optional MP4/PNG
  previews.
- Include eight Kimodo and eight GEM-X example motions, plus an eleven-robot
  qualitative result gallery.
- Provide a Docker image for the hosted Hugging Face Space.
