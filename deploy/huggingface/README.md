---
title: RIMKit Humanoid Retargeting
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
suggested_hardware: cpu-basic
short_description: Contact-aware humanoid retargeting for seventeen robots
---

# RIMKit

Contact-aware SOMA human-motion retargeting for G1, H1, H2, R1, K1, Apollo,
Oli, N1, GR3, ADAM Lite, T1, T2, PM01, Asimov-1, X2-Ultra, A3 T3.0, and IGRIS-C.

Upload a Kimodo SOMA `.npz`, a GEM-X SOMA `.pt`, or a BONES-SEED SOMA Uniform
`.bvh` from your own licensed dataset copy, choose a humanoid from the
seventeen-model target selector, and run RIMKit's complete DMR → CoRe pipeline. The
extension selects the source adapter; a `.bvh` is converted to the SOMA77 `.npz`
contract on upload. The demo returns the same safe final robot-motion `.npz`,
manifest, and MP4 preview for every format. BONES-SEED results: Motion Data by
[Bones Studio](https://bones.studio/).

Browser previews default to 640×360 at up to 15 FPS for responsive CPU
rendering. Downloaded robot-motion data preserves the original source timeline.

The service runs the compiled C++ MuJoCo backend. Public jobs are processed one
at a time, completed artifacts expire after 30 minutes, and uploaded files are
stored only on the Space's ephemeral disk.

- [RIMKit repository](https://github.com/tmjeong1103/RIMKit)
- [Robot Intelligence Lab](https://sites.google.com/view/sungjoon-choi/home)
- [CoRe method paper](https://doi.org/10.1109/Humanoids65713.2025.11203055)
