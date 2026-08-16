# Supported robots

The robot registry contains eleven bundled humanoid models.

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
| PNDbotics | ADAM Lite | `adam` | `pndbotics/adam/adam_lite.xml` | 32 | 25 |
| Booster Robotics | T1 | `t1` | `booster/t1/t1.xml` | 30 | 23 |
| ENGINEAI | PM01 | `pm01` | `engineai/pm01/xml/serial_pm_v2.xml` | 31 | 24 |

List the registry from an installed package:

```bash
rimkit robots list
rimkit robots verify
```

Every model can be selected by ID through the Python API, CLI, browser demo,
and example-output generator. The registry is the canonical source of package
paths, dimensions, scene wrappers, and output joint layouts.

Some vendor models contain passive or compatibility joints that remain in
MuJoCo `qpos` but are not actuated. Consumers should use the named layout stored
in each `core-robot-motion-v1` output instead of assuming that `qpos` columns
equal the actuated joint list.

Robot model licenses, upstream revisions, and RIMKit-local scene integration
changes are documented in [licenses.md](licenses.md) and in each vendor asset
directory's `SOURCE.yaml`.
