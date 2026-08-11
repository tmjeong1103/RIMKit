# CoRe

<p align="center"><b>Contact-Aware Motion Retargeting for Humanoid Robots</b></p>

<p align="center">
  <a href="https://github.com/tmjeong1103/CoRe/actions/workflows/ci.yml"><img src="https://github.com/tmjeong1103/CoRe/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://huggingface.co/spaces/robotaemoon/CoRe"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-FFD21E.svg" alt="Hugging Face demo"></a>
  <a href="https://doi.org/10.1109/Humanoids65713.2025.11203055"><img src="https://img.shields.io/badge/Paper-Humanoids%202025-b31b1b.svg" alt="Humanoids 2025 paper"></a>
  <a href="https://doi.org/10.1109/IROS60139.2025.11246607"><img src="https://img.shields.io/badge/Paper-IROS%202025-b31b1b.svg" alt="IROS 2025 paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="Apache-2.0 license"></a>
</p>

CoRe transforms SOMA human motion into contact-aware whole-body motion for
humanoid robots. It bundles eleven targets from Unitree, ROBOTIS, Apptronik,
LimX Dynamics, Fourier Intelligence, PNDbotics, Booster Robotics, and ENGINEAI
behind one Python and command-line interface, with robot-motion NPZ and video
export ready out of the box.

The pipeline builds on
[robust robot motion retargeting](https://doi.org/10.1109/IROS60139.2025.11246607)
and
[contact-aware motion refinement](https://doi.org/10.1109/Humanoids65713.2025.11203055).
CoRe is developed at the
[Robot Intelligence Lab (RILAB)](https://sites.google.com/view/sungjoon-choi/home).

<p align="center">
  <b>SOMA Motion (.npz)</b> &rarr; <b>DMR</b> &rarr; <b>CoRe</b> &rarr; <b>Robot Motion (.npz + MP4)</b>
</p>

## Highlights

- Contact-aware whole-body retargeting with collision refinement and grounding
- One pipeline for eleven bundled humanoid robot models
- Switch robots with a single `--robot` argument
- Compiled C++ MuJoCo kernels with a portable Python fallback
- Browser, command-line, and Python interfaces
- Eight ready-to-run example motions with 88 rendered robot results

## Result videos

The same eight human motions are retargeted to all eleven supported humanoid
robots using the complete CoRe pipeline. Every result below was rendered with
the same shared lighting setup. Players follow the public robot order:
**G1, H1, H2, R1, K1, Apollo, Oli, N1, ADAM Lite, T1, PM01**.

Each motion uses one wide player row. Scroll horizontally to compare all
eleven robots. Reproducible MP4/PNG files remain under
[`docs/media/final`](docs/media/final).

<details open>
<summary><b>Stand, walk, run, stop — all 11 robots</b></summary>

<br>

<div style="width: 100%; overflow-x: auto;">
<table style="display: block; overflow-x: auto; white-space: nowrap;">
  <tr>
    <td align="center"><b>G1</b><br><video src="https://github.com/user-attachments/assets/d92b8c0e-0eab-4907-a329-92e6f5fedbb8" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H1</b><br><video src="https://github.com/user-attachments/assets/7cfdfb9d-ee26-416e-aada-2e037a4a7aab" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H2</b><br><video src="https://github.com/user-attachments/assets/94381a0a-a82f-4cc7-9bb6-6993d95a60a1" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>R1</b><br><video src="https://github.com/user-attachments/assets/e9f48b5a-f251-4d69-9915-42120aac4ce2" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>K1</b><br><video src="https://github.com/user-attachments/assets/5bb624d6-4047-4e78-ba50-9347c9bc718d" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Apollo</b><br><video src="https://github.com/user-attachments/assets/dead06ec-12cc-4daa-9641-81f6cf20c3b6" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Oli</b><br><video src="https://github.com/user-attachments/assets/0de8e32d-a087-44bc-a078-8c49e84fad0a" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>N1</b><br><video src="https://github.com/user-attachments/assets/9969c2a5-353c-4bfd-9fd1-37d91bbd3b3b" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>ADAM Lite</b><br><video src="https://github.com/user-attachments/assets/9613fbb3-4a44-448f-8fe0-35a6ac22305d" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>T1</b><br><video src="https://github.com/user-attachments/assets/3c6616ee-b6e5-4133-8ce9-657aae9685dd" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>PM01</b><br><video src="https://github.com/user-attachments/assets/08af2a68-3692-4b0c-aafb-a0c003a308ef" width="240" controls preload="metadata"></video></td>
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
    <td align="center"><b>G1</b><br><video src="https://github.com/user-attachments/assets/67a5683a-04ad-4346-be84-44ad04b0b81f" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H1</b><br><video src="https://github.com/user-attachments/assets/97b20968-59a3-40c3-b765-bd01c5e0a891" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H2</b><br><video src="https://github.com/user-attachments/assets/a40f5946-dd77-4e29-a84a-0c36d2b9943f" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>R1</b><br><video src="https://github.com/user-attachments/assets/124f6587-a3c4-419f-b8d0-6a4e0dc3a00a" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>K1</b><br><video src="https://github.com/user-attachments/assets/54428ef3-3a5f-4f34-a8c1-bc5d13b7a913" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Apollo</b><br><video src="https://github.com/user-attachments/assets/cd293990-f821-474f-8007-d5f5f25fbb2f" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Oli</b><br><video src="https://github.com/user-attachments/assets/e9e7fc2c-5c0c-4071-aada-7218bbbab3c0" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>N1</b><br><video src="https://github.com/user-attachments/assets/812e6445-db1f-4292-a9d2-fd849adbee58" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>ADAM Lite</b><br><video src="https://github.com/user-attachments/assets/2758b5b8-6274-4b64-b9be-cf6e30e5d365" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>T1</b><br><video src="https://github.com/user-attachments/assets/102e7e51-b926-4541-ba00-bc1cb5d553a6" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>PM01</b><br><video src="https://github.com/user-attachments/assets/235e0141-9abb-4f9a-8c9b-41350d39f186" width="240" controls preload="metadata"></video></td>
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
    <td align="center"><b>G1</b><br><video src="https://github.com/user-attachments/assets/97aae43e-49d6-40e0-925a-e950f0f89432" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H1</b><br><video src="https://github.com/user-attachments/assets/11ebaf8c-2b81-45dd-9430-b4fbd6d5b92b" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H2</b><br><video src="https://github.com/user-attachments/assets/56b6ff3c-a964-461f-bbaa-d9ac43b5450b" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>R1</b><br><video src="https://github.com/user-attachments/assets/12397392-fb7c-4a35-a1f9-8dab361978b3" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>K1</b><br><video src="https://github.com/user-attachments/assets/3265a0d3-2189-4d22-8e6f-1a46b6bc9878" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Apollo</b><br><video src="https://github.com/user-attachments/assets/e1b66aa4-5ce9-49a0-bcab-c919f16114f1" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Oli</b><br><video src="https://github.com/user-attachments/assets/9e3ab1d6-7212-43c4-a905-c928389e53f9" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>N1</b><br><video src="https://github.com/user-attachments/assets/c78dd5cd-3c5d-4a5a-ad01-35dc0e586e5e" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>ADAM Lite</b><br><video src="https://github.com/user-attachments/assets/c509f498-40d5-486c-ae7f-c925375b6907" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>T1</b><br><video src="https://github.com/user-attachments/assets/3770958b-e75f-45d5-8a64-9cacaa319554" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>PM01</b><br><video src="https://github.com/user-attachments/assets/1589936f-01f0-4b85-b51c-e2e29ec372e1" width="240" controls preload="metadata"></video></td>
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
    <td align="center"><b>G1</b><br><video src="https://github.com/user-attachments/assets/6e67adb1-10f4-4bef-a914-dfe1600381a2" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H1</b><br><video src="https://github.com/user-attachments/assets/8b97ba50-9098-479f-8d13-6ff30009f563" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H2</b><br><video src="https://github.com/user-attachments/assets/82578220-d0a1-40ad-893e-29ec190468d0" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>R1</b><br><video src="https://github.com/user-attachments/assets/6a81fab2-90e1-4b33-83ab-cc2f03ec0442" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>K1</b><br><video src="https://github.com/user-attachments/assets/194a3c62-237b-42bb-b886-92f84d03e5b5" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Apollo</b><br><video src="https://github.com/user-attachments/assets/96e18f6b-86b7-4572-a0eb-4c670c7e1e72" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Oli</b><br><video src="https://github.com/user-attachments/assets/08900abc-e7f5-4fbb-8168-76d78d0bcdfc" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>N1</b><br><video src="https://github.com/user-attachments/assets/1848f407-1427-40e8-acbc-d962eb606e71" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>ADAM Lite</b><br><video src="https://github.com/user-attachments/assets/59cc7c8e-71f2-4874-a228-7ecee1907a58" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>T1</b><br><video src="https://github.com/user-attachments/assets/eba913eb-4199-4b2e-8b3b-6f7fffb0711f" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>PM01</b><br><video src="https://github.com/user-attachments/assets/ccf3f921-87c9-46af-9f22-f7208500d770" width="240" controls preload="metadata"></video></td>
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
    <td align="center"><b>G1</b><br><video src="https://github.com/user-attachments/assets/e4098f1f-4bcf-43f3-b9a5-601e4be23838" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H1</b><br><video src="https://github.com/user-attachments/assets/4e2db425-cfff-40f1-be0d-ee94f8abcd3e" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H2</b><br><video src="https://github.com/user-attachments/assets/45119e68-3f7d-4c9e-b52e-4aa4530675eb" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>R1</b><br><video src="https://github.com/user-attachments/assets/dd01e1be-0f4f-4d78-8180-502b4871b591" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>K1</b><br><video src="https://github.com/user-attachments/assets/2f107c25-5315-4f1d-b7c7-acd08a42f81f" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Apollo</b><br><video src="https://github.com/user-attachments/assets/0bdff099-ba1f-4761-b6a9-8bd747a08704" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Oli</b><br><video src="https://github.com/user-attachments/assets/639a590a-0b6a-4e6b-ab71-d863e2e6f047" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>N1</b><br><video src="https://github.com/user-attachments/assets/e03316d6-4d71-42d2-bdda-624fc70af80c" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>ADAM Lite</b><br><video src="https://github.com/user-attachments/assets/42ca5da0-1a5c-45cd-90d9-caf0113d3052" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>T1</b><br><video src="https://github.com/user-attachments/assets/c8681aa3-d791-40b8-96fa-d5b06070f9e7" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>PM01</b><br><video src="https://github.com/user-attachments/assets/42ed218a-928e-444c-8bdc-8e63a1a3935e" width="240" controls preload="metadata"></video></td>
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
    <td align="center"><b>G1</b><br><video src="https://github.com/user-attachments/assets/16a8cf1e-5e53-4993-8df1-b9eee6310fa3" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H1</b><br><video src="https://github.com/user-attachments/assets/ced429f8-cf46-4060-9840-8a2d7a63a9b8" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H2</b><br><video src="https://github.com/user-attachments/assets/1b1e04ef-0079-4ff1-bb56-c8a3b09cd52c" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>R1</b><br><video src="https://github.com/user-attachments/assets/c4ab355d-6d3f-4b2b-9c4f-0e111ed69f99" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>K1</b><br><video src="https://github.com/user-attachments/assets/a4a9cde4-1245-43ee-b4c3-4c427474519a" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Apollo</b><br><video src="https://github.com/user-attachments/assets/d2459393-7607-4ed2-bc55-1b7e40d29877" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Oli</b><br><video src="https://github.com/user-attachments/assets/5d8c0535-9282-470f-b4c4-7fd3c07527ad" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>N1</b><br><video src="https://github.com/user-attachments/assets/019d5106-6f7e-4f2a-83bf-5666ceff11f9" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>ADAM Lite</b><br><video src="https://github.com/user-attachments/assets/e036bd73-e518-44b9-a798-508181751368" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>T1</b><br><video src="https://github.com/user-attachments/assets/0743b955-5ea3-4980-a265-0d30ce793b5f" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>PM01</b><br><video src="https://github.com/user-attachments/assets/ead410d0-11d2-45b3-8835-bcade92bf393" width="240" controls preload="metadata"></video></td>
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
    <td align="center"><b>G1</b><br><video src="https://github.com/user-attachments/assets/32043ec4-bf70-4dba-ba40-f8bf6ab615a3" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H1</b><br><video src="https://github.com/user-attachments/assets/47b8c46b-51f1-403d-ab0f-0c6a4d982f6f" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H2</b><br><video src="https://github.com/user-attachments/assets/ee7d6340-6dcf-401e-ab91-eb0e5e83a263" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>R1</b><br><video src="https://github.com/user-attachments/assets/64049740-1db8-40e0-8d87-5e7617b4fcbf" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>K1</b><br><video src="https://github.com/user-attachments/assets/e8846289-c514-4488-9e1c-3856aad3effb" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Apollo</b><br><video src="https://github.com/user-attachments/assets/ba2717dc-668a-412f-a0dc-b57efbf21fff" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Oli</b><br><video src="https://github.com/user-attachments/assets/08abfbaf-7881-4545-8c7c-65ebef904033" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>N1</b><br><video src="https://github.com/user-attachments/assets/1798e24a-3a7b-47e4-997f-6887172b4008" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>ADAM Lite</b><br><video src="https://github.com/user-attachments/assets/98506975-6ab2-4647-8422-785a162cca35" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>T1</b><br><video src="https://github.com/user-attachments/assets/405641ed-cda8-42dc-a16e-23cf64b7cf92" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>PM01</b><br><video src="https://github.com/user-attachments/assets/2ff9b1c1-e616-4fe0-bdae-add33d9644a3" width="240" controls preload="metadata"></video></td>
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
    <td align="center"><b>G1</b><br><video src="https://github.com/user-attachments/assets/fb7927fb-6020-45ce-964d-53c3aa4419ec" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H1</b><br><video src="https://github.com/user-attachments/assets/187edb73-19db-457b-bc48-106d67c98dfb" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>H2</b><br><video src="https://github.com/user-attachments/assets/3a07b6f7-c56b-4d9d-83fd-349dc63006ba" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>R1</b><br><video src="https://github.com/user-attachments/assets/14500623-1384-4cef-b9e5-de7490d2d063" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>K1</b><br><video src="https://github.com/user-attachments/assets/80fb0a15-1297-4ac8-b6e2-a9fef7cd8bec" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Apollo</b><br><video src="https://github.com/user-attachments/assets/4cefe87c-b2ed-44cc-9dff-30875ae8affd" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>Oli</b><br><video src="https://github.com/user-attachments/assets/861e1492-3802-4428-9aa9-49504400d5dc" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>N1</b><br><video src="https://github.com/user-attachments/assets/5eef7e3b-dc48-4ada-a8cd-0c3ff0ff5c14" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>ADAM Lite</b><br><video src="https://github.com/user-attachments/assets/10d8b470-6f7f-4657-b383-15b4329eec0d" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>T1</b><br><video src="https://github.com/user-attachments/assets/a63da1ee-53fd-41d9-b9cd-b178e7884946" width="240" controls preload="metadata"></video></td>
    <td align="center"><b>PM01</b><br><video src="https://github.com/user-attachments/assets/df4ff9ac-4a9f-46fb-a207-cc5a41369720" width="240" controls preload="metadata"></video></td>
  </tr>
</table>
</div>
</details>

The gallery is reproducible from the bundled source motions with
`scripts/generate_example_outputs.py`; its checksummed manifest is
[`docs/media/final/index.json`](docs/media/final/index.json).

## Supported robots

| Manufacturer | Robot | Robot ID |
|---|---|---|
| Unitree Robotics | G1 29-DOF | `g1` |
| Unitree Robotics | H1 | `h1` |
| Unitree Robotics | H2 | `h2` |
| Unitree Robotics | R1 | `r1` |
| ROBOTIS | K1 | `k1` |
| Apptronik | Apollo | `apollo` |
| LimX Dynamics | Oli | `oli` |
| Fourier Intelligence | N1 | `n1` |
| PNDbotics | ADAM Lite | `adam` |
| Booster Robotics | T1 | `t1` |
| ENGINEAI | PM01 | `pm01` |

Switch the target humanoid by changing a single `--robot` argument.

## Installation

CoRe supports Python 3.10 through 3.13 on macOS and Linux. Installing from
source requires a C++17 compiler; the published package builds the native
MuJoCo kernels during installation.

```bash
git clone https://github.com/tmjeong1103/CoRe.git
cd CoRe

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[video]"

# Confirm that the compiled backend is available.
core-retarget backend --require-native
```

### Browser demo

Run CoRe through a local, private browser interface:

```bash
python -m pip install -e ".[web]"
core-retarget serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), upload a SOMA NPZ, and
choose any of the eleven bundled robots from the target selector. The page
validates the motion before it runs, streams pipeline progress, plays the final
MP4, and provides the robot motion NPZ and manifest for download. Uploaded
motions and results stay on the local machine under `runs/web`.

<details>
<summary><b>Web server options</b></summary>

```bash
core-retarget serve \
  --host 127.0.0.1 \
  --port 8000 \
  --runs-dir runs/web \
  --max-upload-mb 256
```

The local demo executes one retargeting job at a time to keep MuJoCo and CPU
usage predictable. Additional submissions wait in the local queue.

</details>

<details>
<summary><b>Deploy as a Hugging Face Space</b></summary>

CoRe includes a production-oriented Docker Space image with the native C++
backend, headless MuJoCo rendering, bounded public queue, and automatic result
expiration. [Open the live CoRe demo](https://huggingface.co/spaces/robotaemoon/CoRe)
or see the [Hugging Face deployment guide](docs/huggingface-space.md).

</details>

## Quick start

Run a bundled SOMA motion on Unitree G1:

```bash
core-retarget run \
  examples/motions/kimodo/soma_rp_v11/stand_walk_run_stop.npz \
  --robot g1 \
  --output runs/g1-demo \
  --video \
  --thumbnail
```

The generated result is written to:

```text
runs/g1-demo/stand_walk_run_stop/g1/
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
core-retarget run MOTION.npz --robot g1 --output runs --backend native
```

</details>

<details>
<summary><b>Generate all 8 motions for all 11 robots</b></summary>

```bash
python scripts/generate_example_outputs.py \
  --output runs/example-outputs \
  --gallery-dir docs/media/final
```

Add `--resume` to continue an interrupted batch.

</details>

## Python API

The Python API uses the same pipeline and output format as the CLI:

```python
from core_retarget import Retargeter, RunConfig

result = Retargeter("g1", RunConfig(robot="g1", backend="auto")).run(
    "examples/motions/kimodo/soma_rp_v11/stand_walk_run_stop.npz",
    "runs/python-demo",
    render_video=True,
    render_thumbnail=True,
)

print(result.final_motion_path)
print(result.video_path)
```

## Input and output

SOMA-compatible source motions can be generated with
[Kimodo](https://github.com/nv-tlabs/kimodo) or
[GEM-X](https://github.com/NVlabs/GEM-X).

CoRe accepts [SOMA human motion](docs/input-format.md) stored as NPZ. A motion
contains 3D joint positions and global joint rotations over time. The bundled
examples can be used immediately or replaced with another motion following the
same schema.

The output robot-motion NPZ contains timestamps, MuJoCo `qpos`, named
root/joint layouts, and contact information. Load it safely with:

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

## Citation

If you find CoRe useful for your research, please cite both papers:

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

CoRe source code is released under the [Apache License 2.0](LICENSE). Bundled
robot descriptions and example motions retain their respective licenses and
provenance; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[docs/licenses.md](docs/licenses.md) for details.
