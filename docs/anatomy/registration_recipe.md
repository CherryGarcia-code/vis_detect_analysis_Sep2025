# Probe-Track Registration Recipe (Operator SOP)

This is the human, off-repo workflow that produces the one **track artifact** per
subject (`data/anatomy/<subject>_shank_tracks.json`) the in-repo localizer consumes.
The artifact contract is defined in `src/visdetect/anatomy/tracks.py`; the design
rationale is in `docs/superpowers/specs/2026-06-17-channel-anatomical-localization-design.md`.

The probe is a chronic/fixed NP2.0 four-shank, so **one track-set per subject** is
reused across all that subject's sessions (sessions differ only by the active
y-window of the bank).

## Coordinate frame

All CCF coordinates are **microns** in the `allen_mouse_25um` BrainGlobe atlas
space, axis order **(AP, ML, DV)**. Each shank polyline is ordered **deepest point
first** (index 0 = closest to the tip). Record the atlas name in the meta JSON so it
travels with the artifact.

## Steps

1. **Reconstruct the serial stack → 3D volume.** Produce a single TIFF volume from
   the imaged sections. Note the voxel size (µm, one value per axis) and the
   BrainGlobe **orientation code** of your volume (e.g. `asr`).

2. **Register to the Allen CCF with brainreg.** Run headless (locally, or on the
   SLURM `cpu` partition via an sbatch wrapper modelled on
   `slurm/run_unitmatch_subject.sbatch`):

   ```
   py scripts/anatomy/run_brainreg.py --image <vol.tif> --out <dir> \
       --voxel <vx> <vy> <vz> --orientation <code> --atlas allen_mouse_25um
   ```

3. **Trace each shank's dye track** in napari + brainglobe-segmentation, in the
   registered (atlas) space:
   - Full track visible → trace it; `method = "brainreg_traced"`.
   - Partial track → trace the visible segment, record `method = "extended_from_tip"`
     and a `planned_vector` (tip→entry direction) so upper channels extrapolate along
     it (with growing σ).
   - No dye on a shank → plan it in Pinpoint; `method = "pinpoint_planned"` and supply
     `planned_entry` + `planned_vector`.

### Recommended path: import the brainglobe-segmentation output directly

brainglobe-segmentation writes, per shank, an `atlas_space/tracks/<name>.npy` spline
(shape `(N,3)`, **microns**, axis order `(AP, DV, ML)`) plus a `<name>.csv` of the
Allen region at each point. `scripts/anatomy/import_brainglobe_tracks.py` consumes
those `.npy` files directly — no manual CSV/JSON authoring needed:

```
py scripts/anatomy/import_brainglobe_tracks.py --subject <S> \
    --tracks-dir data/anatomy/<S>/segmentation/atlas_space/tracks \
    --hemisphere left \
    --shank-order shank1_med shank2_fit shank3 shank4
```

- It applies the verified transform `(AP,DV,ML)->(AP,ML,DV)` and reorders each polyline
  **deepest-first**, then writes/validates `data/anatomy/<S>_shank_tracks.json`.
- `--shank-order` lists the `.npy` stems in **probe electrode-shank order** (index 0 =
  smallest-x column in the Kilosort channel map). Index 0 first. If omitted it auto-orders
  medial→lateral by tip ML (use `--lateral-first` to flip) — but **prefer an explicit
  order** and confirm which physical shank is electrode-0 from your implant geometry.
- **Determine `--hemisphere` from the data, not the napari view.** napari's display can
  flip L/R depending on viewing direction; the saved atlas-space coordinates are what
  matter. Query brainglobe's hemisphere volume at a tip coordinate, e.g.
  `BrainGlobeAtlas("allen_mouse_25um").hemispheres[ap//25, dv//25, ml//25]` (1=left,
  2=right). In Allen `asr`, ML < 5700 µm = right, ML > 5700 µm = left.
- Cross-check: our `AllenAtlas.region_at` on a transformed point must match the region in
  the shank's `.csv` (this was verified at 100% for BG_046).

Then skip to step 6. The manual two-file contract below (steps 4–5) remains available for
non-brainglobe tracing (e.g. Pinpoint-only planning).

4. **Export the tracing to our contract** (two files, both under our control):
   - `<subject>_track_points.csv` — columns
     `probe_shank_index,point_order,ap_um,ml_um,dv_um` (deepest point = smallest
     `point_order`). You may leave `probe_shank_index` **blank** and instead provide a
     `shank_group` column; `import_track.py` will then assign probe shank indices from
     the documented orientation + hemisphere.
   - `<subject>_track_meta.json`:
     ```json
     {
       "subject": "BG_046",
       "hemisphere": "right",
       "barcode_orientation": "forward",
       "atlas": "allen_mouse_25um",
       "source_tool": "brainglobe-segmentation",
       "created": "2026-06-17",
       "shanks": {
         "0": {"tip_y_um": 0.0, "method": "brainreg_traced",
               "sigma_along_um": 25.0, "sigma_across_um": 25.0, "sigma_growth_k": 0.0,
               "planned_entry": null, "planned_vector": [0, 0, -1]}
       }
     }
     ```
     `barcode_orientation` and `hemisphere` are the **single documented place** the
     vendor/hemisphere shank-ordering convention is encoded; the monotonicity guard
     (`validate_shank_order`) catches a wrong entry.

5. **Build the validated artifact:**

   ```
   py scripts/anatomy/import_track.py \
       --points <subject>_track_points.csv \
       --meta   <subject>_track_meta.json \
       --out    data/anatomy/<subject>_shank_tracks.json
   ```

6. **Run the in-repo pipeline and eyeball the QC figure:**

   ```
   py scripts/anatomy/build_channel_atlas.py --subject <S>
   py scripts/anatomy/localize_units.py      --subject <S>
   py scripts/anatomy/plot_shank_anatomy.py  --subject <S>
   ```

   Sanity checks on `figures/anatomy/<S>_shank_anatomy.png`:
   - **BG_046** must read cortex → white matter → striatum down each shank, with
     larger σ on the extended (extrapolated) upper channels.
   - **BG_038** must localize to **GPe** (positive control).
