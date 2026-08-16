# CoRe

CoRe is the contact-aware whole-body retargeting method currently available in
RIMKit. It converts SOMA77 human motion into a robot-specific MuJoCo trajectory
through a complete optimization and collision-refinement pipeline.

<p align="center">
  <img src="../media/CoRe_overview.png" alt="CoRe method overview" width="100%">
</p>

## Run the method

```bash
rimkit run SOURCE_MOTION \
  --method core \
  --robot g1 \
  --output runs/core-g1 \
  --video \
  --thumbnail
```

`core` is the default method identifier, so `--method core` may be omitted.

## Inputs and targets

- Kimodo SOMA77 `.npz` and GEM-X SOMA `.pt` inputs
- Eleven bundled humanoid robot targets
- Command-line, Python, and browser interfaces
- Safe robot-motion NPZ, manifest, and optional MP4/PNG output

## Research basis

CoRe combines the contact-aware refinement method from Humanoids 2025 with the
optimization-based rig unification introduced at IROS 2025. See the main
README for the complete citations.
