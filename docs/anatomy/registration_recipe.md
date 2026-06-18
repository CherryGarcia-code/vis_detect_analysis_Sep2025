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
