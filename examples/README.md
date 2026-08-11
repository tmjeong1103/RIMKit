# Examples

The examples directory contains eight KiMoDo SOMA77 motions in `.npz` format
and eight GEM-X SOMA motions in `.pt` format. They are used for quick starts
and regression comparisons. The directory is intentionally outside the
installed Python package so that large motion collections do not silently
inflate every wheel. The KiMoDo gallery batch entry point lives at
[`scripts/generate_example_outputs.py`](../scripts/generate_example_outputs.py).

The default KiMoDo quick-start example is:

    motions/kimodo/soma_rp_v11/stand_walk_run_stop.npz

A GEM-X quick-start example is:

    motions/gem-x/dance.pt

The stationary regression used to reveal lateral leg drift is:

    motions/kimodo/soma_rp_v11/march_in_place_contacts.npz

The files are preserved byte-for-byte from the supplied source artifacts.
The NPZ payload does not embed the original prompts, random seeds, constraints,
generator commit, or generation time, so those fields are recorded as unknown
rather than reconstructed from filenames.

They were generated with
[Kimodo](https://github.com/nv-tlabs/kimodo), using the
[Kimodo-SOMA-RP-v1.1](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1)
model. CoRe does not include Kimodo source code, model weights, or a model
checkpoint. The generator repository is Apache-2.0, while the model is covered
by the NVIDIA Open Model License. That model license classifies generated
output as something other than a Derivative Model and says NVIDIA claims no
ownership in the output. Those provisions do not assign a data license to the
two NPZ files themselves.

The NPZ files are included only to run CoRe examples and regression checks;
they are not offered as a standalone motion dataset. `NOASSERTION` records
that no separate data license is assigned. This policy is intentional and is
not a release gate for CoRe.

Reference identity:

- `stand_walk_run_stop.npz`: 150 frames at 30 Hz; SHA-256
  `16112abc72c0dbb85eb6b32d2ae284d40ebcc496214f9d5ac1fa8a29e12b9a07`
- `march_in_place_contacts.npz`: 240 frames at 30 Hz; SHA-256
  `458c9a9fb2f0153d84023489b674e85f69688e8cc6bf90a3f08ee82bc68a11c4`

See `motions/kimodo/SOURCE.yaml` for the machine-readable provenance record.

From the repository root, install rendering and both source adapters:

```bash
python -m pip install -e ".[gemx,video]"
```

Generate a complete unreviewed KiMoDo candidate for one robot:

```bash
core-retarget run \
  examples/motions/kimodo/soma_rp_v11/stand_walk_run_stop.npz \
  --robot g1 --output runs --video --thumbnail
```

Select GEM-X by passing a `.pt` path. GEM-X PT has no embedded sampling rate,
so provide the source rate explicitly; the bundled examples are 30 Hz:

```bash
core-retarget run \
  examples/motions/gem-x/dance.pt \
  --robot g1 --fps 30 --output runs --video --thumbnail
```

Both commands run contact extraction, DMR, initial collision handling, target
trajectory extraction, ARA, FPA target generation, FPA IK and grounding, final
collision handling, Stage 9 diagnostics, and `core-robot-motion-v1` export.
The results are written below `runs/stand_walk_run_stop/g1` and
`runs/dance/g1`, respectively. Final previews use the provider-aware 1280×720
CoRe camera and top-left LF/RF contact panel.

Generate all eight bundled KiMoDo motions for all eleven supported robots and
optionally publish the portable final MP4/PNG gallery directory:

```bash
python scripts/generate_example_outputs.py \
  --output runs/example-outputs \
  --gallery-dir docs/media/final
```

Re-running with `--resume` skips only complete outputs whose manifest and
artifact hashes still validate. Select a smaller matrix by repeating
`--motion stand_walk_run_stop` and/or `--robot g1` with the desired IDs.

Complete means every Stage 1 through Stage 9 function and final export ran;
it does not mean the motion passed visual or physical review. Example manifests
and gallery metadata therefore use `review_status=unreviewed`. Some results
can retain visible support-foot clearance, so
inspect each final MP4 before using its qpos in simulation or on hardware.

For an intentionally incomplete view of only DMR and the first arm-collision
pass, use `core-retarget review`. That diagnostic output records
`pipeline_complete=false` and must not be presented as final robot motion.
