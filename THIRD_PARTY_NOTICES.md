# Third-party notices

RIMKit contains robot model files derived from third-party repositories. These
files retain their upstream licenses; the repository-level Apache-2.0 license
does not replace those terms.

## nanobind

The compiled `rimkit._core_native` extension incorporates nanobind.

Source: https://github.com/wjakob/nanobind

License: BSD-3-Clause. A copy is provided at
licenses/nanobind-BSD-3-Clause.txt.

## MuJoCo

The native extension is compiled against and dynamically links to MuJoCo
3.6.0, which is installed as a required Python package dependency. RIMKit wheels
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

RIMKit-local copies may contain retargeting landmarks and scene integration
changes. File-level provenance and hashes are recorded in the adjacent
SOURCE.yaml manifest.

## ROBOTIS K1 robot description

Model: K1

Source: https://github.com/ROBOTIS-GIT/ai_sapiens

Pinned source revision: c2880e89fb3451a07b6d2600e274224ffcf912e4

License: Apache-2.0. A copy is provided at
licenses/robotis-Apache-2.0.txt and with the packaged ROBOTIS assets.

The RIMKit-local K1 model is substantially modified from upstream. The XML and
one torso mesh differ, and the upstream head model is not included. These
modifications are identified in SOURCE.yaml and MODIFICATIONS.md next to the
packaged model.

## MuJoCo Menagerie robot descriptions

Models: Apptronik Apollo, Fourier Intelligence N1, and PNDbotics ADAM Lite

Source: https://github.com/google-deepmind/mujoco_menagerie

Pinned source revision: 71f066ad0be9cd271f7ed58c030243ef157af9f4

Licenses: Apache-2.0 for Apollo and N1; MIT for ADAM Lite. Copies are
provided under `licenses/` and beside the packaged assets.

The RIMKit-local XML files contain retargeting landmarks and scene-integration
changes. Exact provenance, modifications, and hashes are recorded in each
vendor's adjacent SOURCE.yaml and FILES.sha256 manifests.

## Booster Robotics robot descriptions

Models: T1 and T2

Source: https://github.com/BoosterRobotics/booster_assets

Pinned source revision: 38a0ae84b17743a8aa21511f69ed38e7d22d1664

License: BSD-3-Clause. A copy is provided at
`licenses/booster-BSD-3-Clause.txt` and with the packaged Booster assets.

The RIMKit-local XML files use a pelvis-root kinematic tree for IK and add
retargeting landmarks. Exported motions retain the original vendor joint
order. Exact modifications and hashes are recorded beside the assets.

## Fourier Intelligence GR3 robot description

Model: GR3

Source: https://github.com/YanjieZe/GMR

Pinned source revision: bb1bbe40774794fceb2a7c579a3464a28e68c844

License: MIT. A copy is provided at `licenses/gmr-MIT.txt` and beside the
packaged GR3 assets.

The RIMKit-local XML removes the standalone scene, adds contact-aware
retargeting landmarks, and enables the physical foot and hand mesh collisions.

## AgiBot A3 robot description

Model: A3 T3.0

Source: https://github.com/AgibotTech/A3-A3U-robot-model

Pinned source revision: 589f508ff357447c610a3f3004419035ddc8f153

License: MulanPSL-2.0. A copy is provided at
`licenses/agibot-MulanPSL-2.0.txt` and with the packaged AgiBot assets.

The RIMKit-local XML removes the standalone terrain include and adds shoulder,
sole, toe, and hand-tip retargeting landmarks.

## LimX Dynamics Oli robot description

Model: Oli (HU_D04)

Source: https://github.com/limxdynamics/humanoid-description

Pinned source revision: a90f734c153aa3ecffc8b674af1e0a323cb55d1a

License: Apache-2.0. A copy is provided at
licenses/limx-Apache-2.0.txt and with the packaged LimX assets.

The RIMKit-local XML scopes vendor defaults and uses RIMKit's shared scene. Exact
provenance, modifications, and hashes are recorded beside the packaged model.

## ENGINEAI PM01 robot description

Model: PM01

Source: https://github.com/YanjieZe/GMR

Pinned source revision: 39c70d031287d899eade658cea3d88b41402356c

License: BSD-3-Clause. A copy is provided at
licenses/engineai-BSD-3-Clause.txt and with the packaged ENGINEAI assets.

The RIMKit-local serial-links XML contains retargeting landmarks. Exact
provenance, modifications, and hashes are recorded beside the packaged model.

## Asimov-1 robot description

Model: Asimov-1

Source: https://github.com/menloresearch/asimov-1

Pinned source revision: b8420ffe99159065152aa1321a03147c0962f251

Licenses: GPL-2.0-only for the simulation model and CERN-OHL-S-2.0 for the
hardware-derived mesh assets. Copies are provided as
`licenses/asimov-GPL-2.0-only.txt` and
`licenses/asimov-CERN-OHL-S-2.0.txt`, and beside the packaged assets.

The RIMKit-local MJCF removes the standalone scene, cameras, and physics
options, makes mesh paths include-safe, and adds hip, sole, and toe
retargeting landmarks. Exact provenance and hashes are recorded beside the
packaged model.

## AgiBot X2-Ultra robot description

Model: X2-Ultra (X2 Ultra new version)

Source: https://github.com/AgibotTech/agibot_x2_urdf

Pinned source revision: 77f43eb0904dae4c48ccd9154fee824f8ffd4d38

License: MulanPSL-2.0. A copy is provided at
`licenses/agibot-MulanPSL-2.0.txt` and with the packaged AgiBot assets.

The RIMKit-local MJCF removes the model-local timestep and tracking light and
adds sole, toe, hand, and hand-tip retargeting landmarks. Exact provenance and
hashes are recorded beside the packaged model.

## Kimodo generation provenance

Example motion files: the eight `.npz` examples under
examples/motions/kimodo/soma_rp_v11/.

Generator repository: https://github.com/nv-tlabs/kimodo

Generator code license: Apache-2.0

Generation model: https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1

Model license: NVIDIA Open Model License

RIMKit includes generated `.npz` output only. It does not include Kimodo code,
model weights, or a model checkpoint. The model license states that output is
not a Derivative Model and that NVIDIA claims no ownership in output. The
motion files are licensed by Taemoon Jeong under CC BY 4.0. See
examples/LICENSE.md and examples/motions/kimodo/SOURCE.yaml.

## GEM-X generation provenance

Example motion files: the eight `.pt` examples under examples/motions/gem-x/.

Generator repository: https://github.com/NVlabs/GEM-X

Generator code license: Apache-2.0

Generation model: https://huggingface.co/nvidia/GEM-X

Model license: NVIDIA Open Model License

The CoRe authors generated the bundled pose estimates from original footage
recorded by the authors. RIMKit includes neither that footage nor GEM-X code,
model weights, or checkpoints. The motion files are licensed by Taemoon Jeong
under CC BY 4.0. See examples/LICENSE.md and
examples/motions/gem-x/SOURCE.yaml.

## SOMA77 fixed bind rig

GEM-X `.pt` inputs contain SOMA body parameters rather than evaluated joints.
RIMKit therefore includes a compact bind-rig asset derived from Kimodo's
`somaskel77/skin_standard.npz`. Only joint names, parent connections, and bind
transforms are retained; mesh and skinning data are excluded.

Source: https://github.com/nv-tlabs/kimodo

License: Apache-2.0. Exact source/local hashes and the derivation are recorded
in `src/rimkit/assets/soma/SOURCE.yaml`.

## SOMA77 BVH joint-frame offsets

The BONES-SEED BVH converter includes the 77 rotation matrices from Kimodo's
`somaskel77/standard_t_pose_global_offsets_rots.p`, stored as a numeric NPZ with
joint names. They reproduce Kimodo's `to_standard_tpose` joint-frame conversion.

Source: https://github.com/nv-tlabs/kimodo

License: Apache-2.0. Source/local hashes and the derivation are recorded in
`src/rimkit/assets/soma/SOURCE.yaml`.

## BONES-SEED gallery media

`docs/media/final/dance_hiphop_shuffle/` contains robot renderings retargeted
from the BONES-SEED SOMA Uniform motion
`dance_hiphop_shuffle_square_R_001__A320`. No raw BONES-SEED motion data or
converted source motion is included in this repository.

Motion Data by Bones Studio: https://bones.studio/

Use of the underlying dataset is subject to the BONES-SEED license:
https://bones.studio/info/seed-license

## Trademarks

All manufacturer and product names are used only to identify compatible model
assets. No endorsement or affiliation is implied.

## ROBROS IGRIS-C robot description

ROBROS IGRIS-C is retained from the validated serial research model associated
with `robrosinc/igris_c_description_public`. The source checkout revision is
`784616e20a273a7a460689c11b83fd728e53100c`; the research snapshot differs from
that revision's newer v2 xacro. Exact packaged snapshot hashes and modifications
are recorded under `assets/robots/robros`. The upstream `package.xml` declares
BSD 3-clause and is retained verbatim as the license declaration; no standalone
license text exists in the inspected revision.
