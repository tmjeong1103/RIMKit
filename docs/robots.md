# Supported robots

The robot registry contains seventeen bundled humanoid models.

| Manufacturer | Robot | ID | Entry XML | `nq` | Actuated DOF |
|---|---|---|---|---:|---:|
| Unitree Robotics | G1 29-DOF | `g1` | `unitree/g1/g1_29dof.xml` | 36 | 29 |
| Unitree Robotics | H1 | `h1` | `unitree/h1/h1.xml` | 27 | 20 |
| Unitree Robotics | H2 | `h2` | `unitree/h2/h2_mujoco.xml` | 38 | 31 |
| Unitree Robotics | R1 | `r1` | `unitree/r1/R1_C++.xml` | 36 | 29 |
| ROBOTIS | K1 | `k1` | `robotis/k1/k1.xml` | 30 | 23 |
| Apptronik | Apollo | `apollo` | `apptronik/apollo/apptronik_apollo.xml` | 39 | 32 |
| LimX Dynamics | Oli | `oli` | `limx/oli/xml/HU_D04_01_retarget.xml` | 68 | 31 |
| Fourier Intelligence | N1 | `n1` | `fourier/n1/n1.xml` | 30 | 23 |
| Fourier Intelligence | GR3 | `gr3` | `fourier/gr3/mjcf/gr3.xml` | 38 | 31 |
| PNDbotics | ADAM Lite | `adam` | `pndbotics/adam/adam_lite.xml` | 32 | 25 |
| Booster Robotics | T1 | `t1` | `booster/t1/t1.xml` | 30 | 23 |
| Booster Robotics | T2 | `t2` | `booster/t2/t2.xml` | 38 | 31 |
| ENGINEAI | PM01 | `pm01` | `engineai/pm01/xml/serial_pm_v2.xml` | 31 | 24 |
| Menlo Research | Asimov-1 | `asimov1` | `asimov/asimov1/sim-model/xmls/asimov_1.xml` | 30 | 23 |
| AgiBot | X2-Ultra | `x2` | `agibot/x2/X2_URDF-v1.4.0/X2-Ultra.xml` | 38 | 31 |
| AgiBot | A3 T3.0 | `a3` | `agibot/a3/a3_t3d0/mjcf/a3.xml` | 38 | 31 |
| ROBROS | IGRIS-C | `igrisc` | `robros/igrisc/igris_c_retarget.xml` | 38 | 31 |

List the registry from an installed package:

```bash
rimkit robots list
rimkit robots verify
```

Every model can be selected by ID through the Python API, CLI, browser demo,
and example-output generator. The registry is the canonical source of package
paths, dimensions, scene wrappers, and output joint layouts.

Some vendor models contain passive or compatibility joints that remain in
MuJoCo `qpos` but are not actuated. Asimov-1 and GR3 distribute no MuJoCo
actuators, so their listed DOF is the number of non-floating articulated
dimensions. T1, T2, and IGRIS-C use pelvis-root retargeting models internally but export
their articulated qpos columns in the original vendor XML order. Consumers
should use the named layout stored in each
`core-robot-motion-v1` output instead of assuming that `qpos` columns equal the
actuator list. To replay or render an output on the packaged MuJoCo model,
reorder it with `rimkit.export.model_order_qpos(robot_id, qpos, joint_names)`.

Robot model licenses, upstream revisions, and RIMKit-local scene integration
changes are documented in [licenses.md](licenses.md) and in each vendor asset
directory's `SOURCE.yaml`.

IGRIS-C follows the validated Kimodo/GEM-X v3 research settings. Its fixed
base landmark is `Link_Waist_Pitch_aux`; it does not follow the articulated
hip midpoint. Aligned aux bodies provide limb position targets, physical
ankle-roll bodies provide rotation targets, and physical foot geometry
provides ground distances.
