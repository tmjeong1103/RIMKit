# Third-party notices

CoRe contains robot model files derived from third-party repositories. These
files retain their upstream licenses; the repository-level Apache-2.0 license
does not replace those terms.

## nanobind

The compiled `core_retarget._core_native` extension incorporates nanobind.

Source: https://github.com/wjakob/nanobind

License: BSD-3-Clause. A copy is provided at
licenses/nanobind-BSD-3-Clause.txt.

## MuJoCo

The native extension is compiled against and dynamically links to MuJoCo
3.6.0, which is installed as a required Python package dependency. CoRe wheels
do not bundle a separate copy of the MuJoCo shared library.

Source: https://github.com/google-deepmind/mujoco

License: Apache-2.0. The installed MuJoCo distribution carries its license
notice.

## Unitree Robotics robot descriptions

Models: G1, H1, H2, and R1

Source: https://github.com/unitreerobotics/unitree_mujoco

Pinned source revision: ae6a8403e272733e9996ef59990880330496177f

License: BSD-3-Clause. A copy is provided at
licenses/unitree-BSD-3-Clause.txt and with the packaged Unitree assets.

CoRe-local copies may contain retargeting landmarks and scene integration
changes. File-level provenance and hashes are recorded in the adjacent
SOURCE.yaml manifest.

## ROBOTIS K1 robot description

Model: K1

Source: https://github.com/ROBOTIS-GIT/ai_sapiens

Pinned source revision: c2880e89fb3451a07b6d2600e274224ffcf912e4

License: Apache-2.0. A copy is provided at
licenses/robotis-Apache-2.0.txt and with the packaged ROBOTIS assets.

The CoRe-local K1 model is substantially modified from upstream. The XML and
one torso mesh differ, and the upstream head model is not included. These
modifications are identified in SOURCE.yaml and MODIFICATIONS.md next to the
packaged model.

## MuJoCo Menagerie robot descriptions

Models: Apptronik Apollo, Fourier Intelligence N1, PNDbotics ADAM Lite, and
Booster Robotics T1

Source: https://github.com/google-deepmind/mujoco_menagerie

Pinned source revision: 71f066ad0be9cd271f7ed58c030243ef157af9f4

Licenses: Apache-2.0 for Apollo, N1, and T1; MIT for ADAM Lite. Copies are
provided under `licenses/` and beside the packaged assets.

The CoRe-local XML files contain retargeting landmarks and scene-integration
changes. Exact provenance, modifications, and hashes are recorded in each
vendor's adjacent SOURCE.yaml and FILES.sha256 manifests.

## LimX Dynamics Oli robot description

Model: Oli (HU_D04)

Source: https://github.com/limxdynamics/humanoid-description

Pinned source revision: a90f734c153aa3ecffc8b674af1e0a323cb55d1a

License: Apache-2.0. A copy is provided at
licenses/limx-Apache-2.0.txt and with the packaged LimX assets.

The CoRe-local XML scopes vendor defaults and uses CoRe's shared scene. Exact
provenance, modifications, and hashes are recorded beside the packaged model.

## ENGINEAI PM01 robot description

Model: PM01

Source: https://github.com/YanjieZe/GMR

Pinned source revision: 39c70d031287d899eade658cea3d88b41402356c

License: BSD-3-Clause. A copy is provided at
licenses/engineai-BSD-3-Clause.txt and with the packaged ENGINEAI assets.

The CoRe-local serial-links XML contains retargeting landmarks. Exact
provenance, modifications, and hashes are recorded beside the packaged model.

## Kimodo generation provenance

Example motion files: the eight NPZ examples under
examples/motions/kimodo/soma_rp_v11/.

Generator repository: https://github.com/nv-tlabs/kimodo

Generator code license: Apache-2.0

Generation model: https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1

Model license: NVIDIA Open Model License

CoRe includes generated NPZ output only. It does not include Kimodo code,
model weights, or a model checkpoint. The model license states that output is
not a Derivative Model and that NVIDIA claims no ownership in output. The NPZ
files are CoRe execution and regression examples, not a separately licensed
motion dataset. See examples/motions/kimodo/SOURCE.yaml for exact hashes and
the machine-readable provenance record.

## Trademarks

All manufacturer and product names are used only to identify compatible model
assets. No endorsement or affiliation is implied.
