# Changelog

All notable changes to RIMKit will be documented in this file.

## Unreleased

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
