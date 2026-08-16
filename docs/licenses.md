# Licensing and provenance

RIMKit code and RIMKit-authored scene wrappers are Apache-2.0.

The compiled native extension incorporates nanobind under BSD-3-Clause. Its
required notice is included in `licenses/nanobind-BSD-3-Clause.txt`.

The extension is compiled against and dynamically links the required MuJoCo
3.6.0 Python distribution under Apache-2.0; RIMKit does not bundle a separate
MuJoCo shared library. See `THIRD_PARTY_NOTICES.md` for dependency provenance.

Unitree G1, H1, H2, and R1 descriptions are derived from unitree_mujoco at
revision ae6a8403e272733e9996ef59990880330496177f and remain BSD-3-Clause.

ROBOTIS K1 is derived from ai_sapiens 0.1.1 at revision
c2880e89fb3451a07b6d2600e274224ffcf912e4 and remains Apache-2.0. The retained
research model is substantially modified. Its XML contains a modification
notice and its SOURCE.yaml and MODIFICATIONS.md describe known differences.

Apptronik Apollo, Fourier Intelligence N1, PNDbotics ADAM Lite, and Booster
Robotics T1 are derived from MuJoCo Menagerie at revision
71f066ad0be9cd271f7ed58c030243ef157af9f4. Apollo, N1, and T1 remain
Apache-2.0; ADAM Lite remains MIT.

LimX Dynamics Oli is derived from humanoid-description at revision
a90f734c153aa3ecffc8b674af1e0a323cb55d1a and remains Apache-2.0.

ENGINEAI PM01 is derived from GMR at revision
39c70d031287d899eade658cea3d88b41402356c and remains BSD-3-Clause.

The added robot models contain RIMKit-local retargeting landmarks or scene
integration changes. Their vendor-local SOURCE.yaml and FILES.sha256 files
record the exact source, modifications, and packaged file hashes.

The eight Kimodo example motions are generated outputs of
[nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo) and the
[Kimodo-SOMA-RP-v1.1](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1)
model. The eight GEM-X examples were estimated from original footage recorded
by the CoRe authors using [NVlabs/GEM-X](https://github.com/NVlabs/GEM-X) and
the [GEM-X model](https://huggingface.co/nvidia/GEM-X). The original GEM-X
footage is not distributed.

Both generator repositories are Apache-2.0 and their models use the NVIDIA Open
Model License. RIMKit bundles none of their code, model weights, or checkpoints.
The NVIDIA Open Model License states that output is not a Derivative Model,
that NVIDIA claims no ownership rights in output, and that the user remains
responsible for output and its subsequent use.

Copyright in the sixteen bundled example motion files is held by Taemoon Jeong.
They are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see
[`examples/LICENSE.md`](../examples/LICENSE.md). Their generator and model
provenance, frame counts, and exact hashes are recorded in the adjacent
`SOURCE.yaml` files.

Manufacturer names identify compatible assets only and do not imply
endorsement.
