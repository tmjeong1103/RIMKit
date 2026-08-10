---
title: CoRe Humanoid Retargeting
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

# CoRe

Contact-aware SOMA human-motion retargeting for G1, H1, H2, R1, K1, Apollo,
Oli, N1, ADAM Lite, T1, and PM01.

Upload a SOMA `.npz`, choose a humanoid from the eleven-model target selector,
and run the complete DMR → CoRe pipeline. The demo returns the final
robot-motion NPZ, manifest, and MP4 preview.

The service runs the compiled C++ MuJoCo backend. Public jobs are processed one
at a time, completed artifacts expire after 30 minutes, and uploaded files are
stored only on the Space's ephemeral disk.

- [CoRe repository](https://github.com/tmjeong1103/CoRe)
- [Robot Intelligence Lab](https://sites.google.com/view/sungjoon-choi/home)
- [CoRe paper](https://doi.org/10.1109/Humanoids65713.2025.11203055)
