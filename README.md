# CoRe

<h2 align="center">Contact-Aware Motion Retargeting for Humanoid Robots</h2>

<p align="center">
  <a href="https://github.com/tmjeong1103/CoRe/actions/workflows/ci.yml"><img src="https://github.com/tmjeong1103/CoRe/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://huggingface.co/spaces/robotaemoon/CoRe"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-FFD21E.svg" alt="Hugging Face demo"></a>
  <a href="https://doi.org/10.1109/Humanoids65713.2025.11203055"><img src="https://img.shields.io/badge/Paper-Humanoids%202025-b31b1b.svg" alt="Humanoids 2025 paper"></a>
  <a href="https://doi.org/10.1109/IROS60139.2025.11246607"><img src="https://img.shields.io/badge/Paper-IROS%202025-b31b1b.svg" alt="IROS 2025 paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="Apache-2.0 license"></a>
</p>

<p align="center">
  <img src="docs/media/CoRe_overview.png" alt="CoRe overview: SOMA-based source motion through motion retargeting and contact-aware refinement to robot motion" width="100%">
</p>

CoRe transforms [SOMA](https://github.com/NVlabs/SOMA-X) human motion into
contact-aware whole-body motion for humanoid robots. It bundles eleven targets
from Unitree Robotics, ROBOTIS, Apptronik, LimX Dynamics, Fourier Intelligence,
PNDbotics, Booster Robotics, and ENGINEAI behind one Python and command-line
interface, with robot-motion `.npz` and video export ready out of the box.

Source motions can be supplied as [Kimodo](https://github.com/nv-tlabs/kimodo)
`.npz` files or [GEM-X](https://github.com/NVlabs/GEM-X) `.pt` files. CoRe
selects the source adapter from the extension and normalizes both formats to the
same SOMA77 stage contract before retargeting.

The pipeline builds on
[robust robot motion retargeting](https://tmjeong1103.github.io/RMR/)
and
[contact-aware motion refinement](https://tmjeong1103.github.io/CoRe-page/).

## Highlights

- Contact-aware whole-body retargeting with collision refinement and grounding
- One pipeline for eleven bundled humanoid robot models
- Switch robots with a single `--robot` argument
- Compiled C++ MuJoCo kernels with a portable Python fallback
- Browser, command-line, and Python interfaces
- Kimodo `.npz` and GEM-X `.pt` source-motion adapters
- Sixteen ready-to-run source motions: eight Kimodo `.npz` and eight GEM-X `.pt`
- Eight Kimodo gallery motions with 88 rendered robot results

## Demo

Try CoRe without installing it. The hosted browser interface accepts Kimodo
`.npz` and GEM-X `.pt` SOMA motions, retargets them to any of the eleven bundled
humanoid robots, previews the final motion, and provides the safe robot-motion
`.npz` and manifest for download.

[**Launch the live demo on Hugging Face →**](https://huggingface.co/spaces/robotaemoon/CoRe)

After cloning the repository, run the same interface locally:

```bash
python -m pip install -e ".[web]"
core-retarget serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000). See
[Local web server options](#local-web-server-options) for host, port, upload,
and job-storage settings.

## Result videos

The eight bundled Kimodo motions are retargeted to all eleven supported
humanoid robots using the complete CoRe pipeline. Every result below was
rendered with the same shared lighting setup. Players follow the public robot
order:
**G1, H1, H2, R1, K1, Apollo, Oli, N1, ADAM Lite, T1, PM01**.

> [!IMPORTANT]
> These complete pipeline outputs have been visually reviewed for this release,
> and their release metadata records `review_status=reviewed`. Validate any
> newly generated motion in simulation before physical deployment.

Each motion uses one wide player row. Scroll horizontally to compare all
eleven robots. Reproducible MP4/PNG files remain under
[`docs/media/final`](docs/media/final).

<details open>
<summary><b>Stand, walk, run, stop — all 11 robots</b></summary>

<br>

<div style="width: 100%; overflow-x: auto;">
<table style="display: block; overflow-x: auto; white-space: nowrap;">
  <tr>
    <td align="center"><b>G1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/stand_walk_run_stop/g1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/stand_walk_run_stop/h1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H2</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/stand_walk_run_stop/h2.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>R1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/stand_walk_run_stop/r1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>K1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/stand_walk_run_stop/k1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Apollo</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/stand_walk_run_stop/apollo.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Oli</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/stand_walk_run_stop/oli.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>N1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/stand_walk_run_stop/n1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>ADAM Lite</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/stand_walk_run_stop/adam.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>T1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/stand_walk_run_stop/t1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>PM01</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/stand_walk_run_stop/pm01.mp4" width="240" controls preload="metadata"></video></td>
  </tr>
</table>
</div>
</details>

<details open>
<summary><b>March in place with contacts — all 11 robots</b></summary>

<br>

<div style="width: 100%; overflow-x: auto;">
<table style="display: block; overflow-x: auto; white-space: nowrap;">
  <tr>
    <td align="center"><b>G1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/march_in_place_contacts/g1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/march_in_place_contacts/h1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H2</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/march_in_place_contacts/h2.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>R1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/march_in_place_contacts/r1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>K1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/march_in_place_contacts/k1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Apollo</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/march_in_place_contacts/apollo.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Oli</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/march_in_place_contacts/oli.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>N1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/march_in_place_contacts/n1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>ADAM Lite</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/march_in_place_contacts/adam.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>T1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/march_in_place_contacts/t1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>PM01</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/march_in_place_contacts/pm01.mp4" width="240" controls preload="metadata"></video></td>
  </tr>
</table>
</div>
</details>

<details>
<summary><b>Alternating lunges with contacts — all 11 robots</b></summary>

<br>

<div style="width: 100%; overflow-x: auto;">
<table style="display: block; overflow-x: auto; white-space: nowrap;">
  <tr>
    <td align="center"><b>G1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/alternating_lunges_contacts/g1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/alternating_lunges_contacts/h1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H2</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/alternating_lunges_contacts/h2.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>R1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/alternating_lunges_contacts/r1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>K1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/alternating_lunges_contacts/k1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Apollo</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/alternating_lunges_contacts/apollo.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Oli</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/alternating_lunges_contacts/oli.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>N1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/alternating_lunges_contacts/n1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>ADAM Lite</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/alternating_lunges_contacts/adam.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>T1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/alternating_lunges_contacts/t1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>PM01</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/alternating_lunges_contacts/pm01.mp4" width="240" controls preload="metadata"></video></td>
  </tr>
</table>
</div>
</details>

<details>
<summary><b>Backward walk with contacts — all 11 robots</b></summary>

<br>

<div style="width: 100%; overflow-x: auto;">
<table style="display: block; overflow-x: auto; white-space: nowrap;">
  <tr>
    <td align="center"><b>G1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/backward_walk_contacts/g1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/backward_walk_contacts/h1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H2</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/backward_walk_contacts/h2.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>R1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/backward_walk_contacts/r1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>K1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/backward_walk_contacts/k1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Apollo</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/backward_walk_contacts/apollo.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Oli</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/backward_walk_contacts/oli.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>N1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/backward_walk_contacts/n1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>ADAM Lite</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/backward_walk_contacts/adam.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>T1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/backward_walk_contacts/t1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>PM01</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/backward_walk_contacts/pm01.mp4" width="240" controls preload="metadata"></video></td>
  </tr>
</table>
</div>
</details>

<details>
<summary><b>Foot walk and stop — all 11 robots</b></summary>

<br>

<div style="width: 100%; overflow-x: auto;">
<table style="display: block; overflow-x: auto; white-space: nowrap;">
  <tr>
    <td align="center"><b>G1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/foot_walk_stop/g1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/foot_walk_stop/h1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H2</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/foot_walk_stop/h2.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>R1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/foot_walk_stop/r1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>K1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/foot_walk_stop/k1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Apollo</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/foot_walk_stop/apollo.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Oli</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/foot_walk_stop/oli.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>N1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/foot_walk_stop/n1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>ADAM Lite</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/foot_walk_stop/adam.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>T1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/foot_walk_stop/t1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>PM01</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/foot_walk_stop/pm01.mp4" width="240" controls preload="metadata"></video></td>
  </tr>
</table>
</div>
</details>

<details>
<summary><b>Jump and land with contacts — all 11 robots</b></summary>

<br>

<div style="width: 100%; overflow-x: auto;">
<table style="display: block; overflow-x: auto; white-space: nowrap;">
  <tr>
    <td align="center"><b>G1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/jump_land_contacts/g1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/jump_land_contacts/h1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H2</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/jump_land_contacts/h2.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>R1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/jump_land_contacts/r1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>K1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/jump_land_contacts/k1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Apollo</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/jump_land_contacts/apollo.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Oli</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/jump_land_contacts/oli.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>N1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/jump_land_contacts/n1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>ADAM Lite</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/jump_land_contacts/adam.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>T1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/jump_land_contacts/t1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>PM01</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/jump_land_contacts/pm01.mp4" width="240" controls preload="metadata"></video></td>
  </tr>
</table>
</div>
</details>

<details>
<summary><b>Side steps right with contacts — all 11 robots</b></summary>

<br>

<div style="width: 100%; overflow-x: auto;">
<table style="display: block; overflow-x: auto; white-space: nowrap;">
  <tr>
    <td align="center"><b>G1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/side_steps_right_contacts/g1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/side_steps_right_contacts/h1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H2</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/side_steps_right_contacts/h2.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>R1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/side_steps_right_contacts/r1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>K1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/side_steps_right_contacts/k1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Apollo</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/side_steps_right_contacts/apollo.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Oli</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/side_steps_right_contacts/oli.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>N1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/side_steps_right_contacts/n1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>ADAM Lite</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/side_steps_right_contacts/adam.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>T1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/side_steps_right_contacts/t1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>PM01</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/side_steps_right_contacts/pm01.mp4" width="240" controls preload="metadata"></video></td>
  </tr>
</table>
</div>
</details>

<details>
<summary><b>Slow walk with firm steps — all 11 robots</b></summary>

<br>

<div style="width: 100%; overflow-x: auto;">
<table style="display: block; overflow-x: auto; white-space: nowrap;">
  <tr>
    <td align="center"><b>G1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/slow_walk_firm_steps/g1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/slow_walk_firm_steps/h1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H2</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/slow_walk_firm_steps/h2.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>R1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/slow_walk_firm_steps/r1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>K1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/slow_walk_firm_steps/k1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Apollo</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/slow_walk_firm_steps/apollo.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Oli</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/slow_walk_firm_steps/oli.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>N1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/slow_walk_firm_steps/n1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>ADAM Lite</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/slow_walk_firm_steps/adam.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>T1</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/slow_walk_firm_steps/t1.mp4" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>PM01</b><br><video src="https://raw.githubusercontent.com/tmjeong1103/CoRe/main/docs/media/final/slow_walk_firm_steps/pm01.mp4" width="240" controls preload="metadata"></video></td>
  </tr>
</table>
</div>
</details>

This Kimodo gallery is reproducible from the bundled `.npz` motions with
`scripts/generate_example_outputs.py`; its checksummed manifest is
[`docs/media/final/index.json`](docs/media/final/index.json).

## Supported robots

| # | Manufacturer | Robot | Robot ID |
|---:|---|---|---|
| 1 | Unitree Robotics | G1 29-DOF | `g1` |
| 2 | Unitree Robotics | H1 | `h1` |
| 3 | Unitree Robotics | H2 | `h2` |
| 4 | Unitree Robotics | R1 | `r1` |
| 5 | ROBOTIS | K1 | `k1` |
| 6 | Apptronik | Apollo | `apollo` |
| 7 | LimX Dynamics | Oli | `oli` |
| 8 | Fourier Intelligence | N1 | `n1` |
| 9 | PNDbotics | ADAM Lite | `adam` |
| 10 | Booster Robotics | T1 | `t1` |
| 11 | ENGINEAI | PM01 | `pm01` |

Switch the target humanoid by changing a single `--robot` argument.

## Supported platforms

CoRe officially supports macOS and Ubuntu Linux with Python 3.10 through
3.13. The native backend, test suite, and package installation are validated
on macOS; GitHub Actions runs the Python 3.10–3.13 test matrix, native backend,
headless rendering, and Docker deployment checks on Ubuntu.

## Installation

Installing from source requires a C++17 compiler; the package builds the
native MuJoCo kernels during installation.

```bash
git clone https://github.com/tmjeong1103/CoRe.git
cd CoRe

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install CLI rendering plus both Kimodo .npz and GEM-X .pt input support.
python -m pip install -e ".[gemx,video]"

# Confirm that the compiled backend is available.
core-retarget backend --require-native
```

### Local web server options

The local browser demo listens on
[http://127.0.0.1:8000](http://127.0.0.1:8000) by default. Uploaded motions and
results stay on the local machine under `runs/web`. Customize the server with:

```bash
core-retarget serve \
  --host 127.0.0.1 \
  --port 8000 \
  --runs-dir runs/web \
  --max-upload-mb 256
```

The local demo executes one retargeting job at a time to keep MuJoCo and CPU
usage predictable. Additional submissions wait in the local queue.

<details>
<summary><b>Deploy as a Hugging Face Space</b></summary>

CoRe includes a production-oriented Docker Space image with the native C++
backend, headless MuJoCo rendering, bounded public queue, and automatic result
expiration. [Open the live CoRe demo](https://huggingface.co/spaces/robotaemoon/CoRe)
or see the [Hugging Face deployment guide](docs/huggingface-space.md).

</details>

## Quick start

CoRe selects the source adapter from the filename extension. Run a bundled
Kimodo motion on Unitree G1 with:

```bash
core-retarget run \
  examples/motions/kimodo/soma_rp_v11/stand_walk_run_stop.npz \
  --robot g1 \
  --output runs/kimodo-g1 \
  --video \
  --thumbnail
```

Run a bundled GEM-X motion with the same command. GEM-X `.pt` does not store its
sampling rate, so pass the rate used to generate the motion; the bundled GEM-X
examples are 30 Hz:

```bash
core-retarget run \
  examples/motions/gem-x/dance.pt \
  --robot g1 \
  --fps 30 \
  --output runs/gemx-g1 \
  --video \
  --thumbnail
```

The generated results are written below the selected output root:

```text
runs/kimodo-g1/stand_walk_run_stop/g1/
├── final/robot_motion.npz
├── preview/final.mp4
├── preview/final.png
└── manifest.json

runs/gemx-g1/dance/g1/
├── final/robot_motion.npz
├── preview/final.mp4
├── preview/final.png
└── manifest.json
```

To target another robot, change `--robot g1` to any ID in the supported-robots
table above.

<details>
<summary><b>Compute backend selection</b></summary>

CoRe uses the compiled C++ backend automatically when it is available. Every
result manifest records the requested and selected backend. Use
`--backend native` to require C++, or `--backend python` to run the portable
Python backend explicitly:

```bash
core-retarget backend
core-retarget run SOURCE_MOTION --robot g1 --output runs --backend native
```

</details>

<details>
<summary><b>Generate all 8 Kimodo gallery motions for all 11 robots</b></summary>

```bash
python scripts/generate_example_outputs.py \
  --output runs/example-outputs \
  --gallery-dir docs/media/final
```

Add `--resume` to continue an interrupted batch.

</details>

## Python API

The Python API uses the same extension-based dispatch and output format as the
CLI. Run a Kimodo `.npz` with its embedded or default FPS:

```python
from core_retarget import Retargeter, RunConfig

kimodo_result = Retargeter(
    "g1",
    RunConfig(robot="g1", backend="auto"),
).run(
    "examples/motions/kimodo/soma_rp_v11/stand_walk_run_stop.npz",
    "runs/python-kimodo-demo",
    render_video=True,
    render_thumbnail=True,
)

print(kimodo_result.final_motion_path)
print(kimodo_result.video_path)
```

For GEM-X `.pt`, use the same API and supply the source FPS explicitly:

```python
from core_retarget import Retargeter, RunConfig

gemx_result = Retargeter(
    "g1",
    RunConfig(robot="g1", fps=30.0, backend="auto"),
).run(
    "examples/motions/gem-x/dance.pt",
    "runs/python-gemx-demo",
    render_video=True,
    render_thumbnail=True,
)

print(gemx_result.final_motion_path)
print(gemx_result.video_path)
```

## Input and output

SOMA-compatible source motions can be generated with
[Kimodo](https://github.com/nv-tlabs/kimodo) or
[GEM-X](https://github.com/NVlabs/GEM-X). CoRe dispatches by extension:

- `.npz` means a Kimodo SOMA77 motion with global joint positions and rotations.
- `.pt` means a GEM-X SOMA body-parameter result. Install the `gemx` extra for
  command-line or Python `.pt` input: `python -m pip install -e ".[gemx,video]"`.
  The `web` extra already includes this dependency.

See the [source-motion contract](docs/input-format.md) for the required keys,
FPS behavior, and safe-loading rules. The bundled examples contain eight
Kimodo `.npz` motions and eight GEM-X `.pt` motions.

Regardless of source format, the output remains a versioned robot-motion `.npz`
containing timestamps, MuJoCo `qpos`, named root/joint layouts, and contact
information. It has no object arrays or pickle payloads. Load it safely with:

```python
import numpy as np

motion = np.load("robot_motion.npz", allow_pickle=False)
qpos = motion["qpos"]
```

> [!NOTE]
> CoRe is research software. Inspect generated motions in simulation before
> using them on physical hardware.

## Documentation

- [Input motion format](docs/input-format.md)
- [Robot models](docs/robots.md)
- [Pipeline architecture](docs/architecture.md)
- [Licenses and provenance](docs/licenses.md)

## Contact

CoRe is created and maintained by
[Taemoon Jeong](https://taemoon.notion.site/taemoon-page).

- Email: [taemoon-jeong@korea.ac.kr](mailto:taemoon-jeong@korea.ac.kr)
- Profiles: [GitHub](https://github.com/tmjeong1103) · [LinkedIn](https://www.linkedin.com/in/taemoon-jeong-b84502306/) · [Google Scholar](https://scholar.google.co.kr/citations?user=RksrV_QAAAAJ&hl=ko)
- Bug reports and feature requests: [GitHub Issues](https://github.com/tmjeong1103/CoRe/issues)

## Acknowledgments

This project was developed at the
[Robot Intelligence Lab](https://sites.google.com/view/sungjoon-choi/home),
Korea University, under the guidance of
[Professor Sungjoon Choi](https://github.com/sjchoi86).

## Citation

If you use CoRe in your research, please cite both papers:

```bibtex
@inproceedings{jeong2025core,
  author    = {Jeong, Taemoon and Chai, Yoonbyung and Choi, Sol and
               Bak, Jaewan and Kim, Chanwoo and Yoon, Jihwan and
               Lee, Yisoo and Lee, Jongwon and Lee, Kyungjae and
               Kim, Joohyung and Choi, Sungjoon},
  title     = {CoRe: A Hybrid Approach of Contact-Aware Optimization
               and Learning for Humanoid Robot Motions},
  booktitle = {2025 IEEE-RAS 24th International Conference on
               Humanoid Robots (Humanoids)},
  year      = {2025},
  pages     = {293--300},
  doi       = {10.1109/Humanoids65713.2025.11203055}
}

@inproceedings{jeong2025robust,
  author    = {Jeong, Taemoon and Byun, Taehyun and Kim, Jihoon and
               Choi, Keunjun and Oh, Jaesung and Lee, Sungpyo and
               Darwish, Omar and Kim, Joohyung and Choi, Sungjoon},
  title     = {Robust and Expressive Humanoid Motion Retargeting via
               Optimization-Based Rig Unification},
  booktitle = {2025 IEEE/RSJ International Conference on
               Intelligent Robots and Systems (IROS)},
  year      = {2025},
  pages     = {21619--21626},
  doi       = {10.1109/IROS60139.2025.11246607}
}
```

## License

CoRe source code is released under the [Apache License 2.0](LICENSE). The
bundled example motions are licensed under
[CC BY 4.0](examples/LICENSE.md), Copyright 2026 Taemoon Jeong. Bundled robot
descriptions retain their respective licenses and provenance; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[docs/licenses.md](docs/licenses.md) for details.
