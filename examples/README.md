# Examples

The examples directory contains eight Kimodo SOMA77 motions in `.npz` format
and eight GEM-X SOMA motions in `.pt` format. They are used for quick starts
and regression comparisons. The directory is intentionally outside the
installed Python package so that large motion collections do not silently
inflate every wheel. The bundled gallery batch entry point lives at
[`scripts/generate_example_outputs.py`](../scripts/generate_example_outputs.py).

The default Kimodo quick-start example is:

    motions/kimodo/soma_rp_v11/stand_walk_run_stop.npz

A GEM-X quick-start example is:

    motions/gem-x/rapid_stepping.pt

The stationary regression used to reveal lateral leg drift is:

    motions/kimodo/soma_rp_v11/march_in_place_contacts.npz

The files are preserved byte-for-byte from their release inputs. The Kimodo
`.npz` payloads do not embed the original prompts, random seeds, constraints,
generator commit, or generation time, so those fields are recorded as unknown
rather than reconstructed from filenames.

They were generated with
[Kimodo](https://github.com/nv-tlabs/kimodo), using the
[Kimodo-SOMA-RP-v1.1](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1)
model. RIMKit does not include Kimodo source code, model weights, or a model
checkpoint. The generator repository is Apache-2.0, while the model is covered
by the NVIDIA Open Model License. That model license classifies generated
output as something other than a Derivative Model and says NVIDIA claims no
ownership in the output.

The GEM-X `.pt` examples were produced by the CoRe authors from original
footage recorded by the authors. The source footage, GEM-X code, and model
weights are not included.

All sixteen bundled example motions are licensed under CC BY 4.0, with
attribution to Taemoon Jeong. See [`LICENSE.md`](LICENSE.md) for the grant and
suggested attribution. Machine-readable provenance, frame counts, and exact
file hashes are recorded in:

- [`motions/kimodo/SOURCE.yaml`](motions/kimodo/SOURCE.yaml)
- [`motions/gem-x/SOURCE.yaml`](motions/gem-x/SOURCE.yaml)

From the repository root, install rendering and both source adapters:

```bash
python -m pip install -e ".[gemx,video]"
```

Generate a complete unreviewed Kimodo candidate for one robot:

```bash
rimkit run \
  examples/motions/kimodo/soma_rp_v11/stand_walk_run_stop.npz \
  --robot g1 --output runs --video --thumbnail
```

Select GEM-X by passing a `.pt` path. GEM-X `.pt` has no embedded sampling rate,
so provide the source rate explicitly; the bundled examples are 30 Hz:

```bash
rimkit run \
  examples/motions/gem-x/rapid_stepping.pt \
  --robot g1 --fps 30 --output runs --video --thumbnail
```

Both commands run contact extraction, DMR, initial collision handling, target
trajectory extraction, ARA, FPA target generation, FPA IK and grounding, final
collision handling, Stage 9 diagnostics, and `core-robot-motion-v1` export.
The results are written below `runs/stand_walk_run_stop/g1` and
`runs/rapid_stepping/g1`, respectively. Final previews use the provider-aware 1280×720
CoRe camera and top-left LF/RF contact panel.

Generate all eight bundled Kimodo motions for all eleven supported robots and
optionally publish the portable final MP4/PNG gallery directory:

```bash
python scripts/generate_example_outputs.py \
  --source-set kimodo \
  --output runs/example-outputs \
  --gallery-dir docs/media/final
```

Run the equivalent 8 × 11 GEM-X matrix at the bundled motions' 30 Hz source
rate with:

```bash
python scripts/generate_example_outputs.py \
  --source-set gem-x \
  --output runs/gem-x-example-outputs \
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
pass, use `rimkit review`. That diagnostic output records
`pipeline_complete=false` and must not be presented as final robot motion.
