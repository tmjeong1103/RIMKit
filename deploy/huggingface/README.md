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
short_description: Contact-aware humanoid retargeting for eleven robots
---

# RIMKit

Contact-aware SOMA human-motion retargeting for G1, H1, H2, R1, K1, Apollo,
Oli, N1, ADAM Lite, T1, and PM01.

Upload a Kimodo SOMA `.npz` or GEM-X SOMA `.pt`, choose a humanoid from the
eleven-model target selector, and run RIMKit's complete DMR → CoRe pipeline. The
extension selects the source adapter. The demo returns the same safe final
robot-motion `.npz`, manifest, and MP4 preview for either format.

The service runs the compiled C++ MuJoCo backend. Public jobs are processed one
at a time, completed artifacts expire after 30 minutes, and uploaded files are
stored only on the Space's ephemeral disk.

- [RIMKit repository](https://github.com/tmjeong1103/RIMKit)
- [Robot Intelligence Lab](https://sites.google.com/view/sungjoon-choi/home)
- [CoRe method paper](https://doi.org/10.1109/Humanoids65713.2025.11203055)
