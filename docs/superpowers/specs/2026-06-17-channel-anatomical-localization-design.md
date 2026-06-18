# Design spec — Channel & unit anatomical localization (CCF coordinates + Allen region per channel/unit)

| | |
|---|---|
| **Topic** | Estimate the Allen-CCF location (AP/ML/DV) + brain region + uncertainty of every recording channel, per chanmap, per subject; propagate to units via peak channel |
| **Date** | 2026-06-17 |
| **Status** | SPEC — awaiting user review, then writing-plans |
| **Subjects** | ~7 striatal-targeted (of 11 total). Subject-parametrized; **validate end-to-end on BG_046 first** |
| **Method anchor** | BrainGlobe (`brainreg` volumetric deformable registration + `brainglobe-segmentation` track tracing + `brainglobe-atlasapi`); Allen Mouse CCF. Lineage of the lab's original allenCCF/SHARP-Track per-cluster `brain_region_comb` (`scripts/conversion/matlab_scripts/.../Brain_regions/`), rebuilt in Python |
| **Scientific stakes** | Catches mis-targeting (e.g. BG_038 = GPe, not striatum — see `memory/qc_celltype_yield_investigation_jun2026.md`); answers "are these channels in striatum or the cortex above?"; enables region-filtered and depth/topography analyses tied to the PPC/aMOs/pMOs→D1/D2 proposal scope (`memory/proposal_aims.md`) |

---

## 1. Goal & motivation

For each subject, estimate the anatomical location of every Neuropixels 2.0 recording channel — metric Allen-CCF coordinates `(AP, ML, DV)` **and** an Allen region label, each with an explicit **uncertainty** — for every chanmap used across that subject's sessions, and propagate the location to every analyzable unit via its peak channel. Persist the result so the rest of the pipeline can join on it (filter/group units by region, mark the cortex↔striatum transition, relate cell types to subregion).

This information existed in the original MATLAB lab pipeline (per-cluster `good_cl_coord.brain_region_comb`) but was **never carried into the Python PKLs**; the current `Cluster` dataclass (`src/visdetect/core/session.py`) stores only `cluster_id`, `spike_times`, `quality`. This is a clean rebuild in Python.

## 2. What this spec does and does NOT cover

**In scope (Phase A):**
- A **tool-agnostic track artifact** per subject (the contract): the 4 shank trajectories in CCF + depth calibration + per-shank method + uncertainty.
- In-repo, automated consumer: chanmap → per-channel CCF + region + confidence; unit → peak channel → location.
- Sidecar persistence (`data/anatomy/`) + new columns on `build_unit_table`. **No PKL re-ingest.**
- Per-subject **probe-orientation** handling (barcode forward/backward → medial/lateral correctness).
- A per-subject QC figure for human validation.
- Thin in-repo wrappers for the BrainGlobe steps + an adapter from the tracer's export to the track artifact.

**Explicitly OUT of scope (recorded so a fresh chat doesn't overreach):**
- **Phase B** — promoting the location fields into the `Cluster` dataclass and baking them into PKLs on the next re-ingest. Deferred until the Phase-A localization is trusted. (Designed so it is an additive follow-up, not a rewrite.)
- **Building a registration/tracing GUI.** We use `brainreg` + `brainglobe-segmentation` + Pinpoint. The only manual step is identifying/tracing the dye track in napari (irreducibly interactive).
- **Cross-subject spatial warping / averaging.** Coordinates are stored in a common CCF frame so this is *possible* later, but no group-level spatial alignment is built now.
- **Automatic dye-track detection.** Tracing stays human-in-the-loop.
- **Re-deriving the manifest / session selection.** Localization is keyed by subject + chanmap, independent of the staging filter.

## 3. Key facts the design is built on (verified)

- **Probe = Neuropixels 2.0, four-shank.** Confirmed from `channel_positions.npy`: 8 unique x-columns clustering into **4 shanks ~250 µm apart**, 2 columns/shank ~32 µm apart, ~15 µm row pitch; 383–384 active channels.
- **Chronic, physically-fixed probe.** Therefore **one set of 4 shank trajectories per subject** across all sessions. Each session's chanmap only selects a **depth window (bank)** along the fixed shanks (verified: BG_031 active y starts ~765 µm, BG_046 ~1515 µm — the offset shifts per session, the geometry does not). The pipeline **asserts** that a subject's per-session geometries are identical up to the y-offset.
- **`channel_positions.npy` knows only *probe* shank index** (ordered by x-column); it carries no anatomical medial/lateral information — hence §7 (orientation).
- **Histology:** complete, ordered whole-brain serial stacks for all subjects → `brainreg` volumetric registration is viable. Track visibility **varies by shank**: some shanks have a fully traceable dye track; others (incl. **BG_046**, which has only the lower track in dye) need **upward extension from the tip along the planned angle**; a shank with no dye falls back to a Pinpoint planned trajectory.
- **Peak-channel source:** the UnitMatch-extracted **RawWaveforms** are the primary source (raw, per-unit average across channels), with KS `templates.npy` only as a fallback for any subject/session not yet extracted. RawWaveform coverage is expected to reach all subjects.

## 4. Architecture & data flow

```
[HUMAN, per subject — documented SOP, run from our .venv]
  reconstruct serial stack ─► run_brainreg.py (wraps brainreg CLI, unattended)
       ─► CCF-registered volume
  brainglobe-segmentation (napari) ─► trace each shank's visible track   ◄── interactive
  (no-dye shank) ─► Pinpoint planned trajectory export
       │
       ▼  import_track.py (adapter)
  data/anatomy/<subject>_shank_tracks.json   ◄────────────── THE CONTRACT
  data/anatomy/<subject>_probe_config.json   (orientation, hemisphere)

[IN-REPO, automated]
  channel_map.py ─► per channel: (probe_shank_idx, x, y) + chanmap_signature
  (orientation)  ─► probe_shank_idx ─► anatomical shank / correct track
  localize.py    ─► place channel at depth y on its shank polyline (arc-length)
                    ─► CCF (AP,ML,DV)
  atlas.py       ─► region_at(CCF) ─► acronym/name + coarse region
                    ─► region_confidence (track σ × border distance)
       │
       ├─► data/anatomy/<subject>_channel_atlas.csv   (one row-set per chanmap_signature)
       │
  peak_channel.py ─► unit ─► peak channel (RawWaveforms primary; KS templates fallback)
       ▼
  localize_units.py ─► join ─► per-unit location ─► merged into build_unit_table:
     peak_channel, shank, depth_um, ccf_ap, ccf_ml, ccf_dv,
     region_acronym, region_name, region_coarse, region_confidence, loc_method
       ▼
  plot_shank_anatomy.py ─► per-subject QC figure
```

## 5. The track artifact (the contract)

`data/anatomy/<subject>_shank_tracks.json`, **schema-validated fail-loud** on load (same discipline as the frozen unit-label-table contract, `memory/p0_spine_audit_done_june2026.md`). Tool-agnostic: `brainglobe-segmentation`, SHARP-Track, or Pinpoint can all populate it.

Per artifact: `subject`, `atlas` (e.g. `allen_mouse_25um`), `hemisphere` (`left`|`right`), `barcode_orientation` (`forward`|`backward`), `created`, `source_tool`, and a `shanks` list of 4 entries. Each shank entry:

| field | meaning |
|---|---|
| `probe_shank_index` | 0–3, matching the x-column ordering in `channel_positions.npy` |
| `ccf_polyline` | ordered list of `[AP, ML, DV]` µm points tracing the shank in CCF (deepest = tip) |
| `depth_calibration` | the physical `y_um` (channel-coordinate depth-along-shank) at the **tip** point, plus the convention to map any channel `y_um` to a position on the polyline by **arc length** |
| `planned` | planned entry point + insertion vector (used for upward extension / Pinpoint shanks) |
| `method` | `brainreg_traced` \| `extended_from_tip` \| `pinpoint_planned` |
| `sigma_um` | uncertainty: `{along_track, across_track}`, and for `extended_from_tip` the growth rate `k` of σ with distance above the deepest dye point |

The artifact is the **only** thing the human-in-the-loop produces; everything downstream is automated and re-runnable.

## 6. In-repo modules (`src/visdetect/anatomy/` — new subpackage)

- **`tracks.py`** — `ShankTrack` / `TrackArtifact` dataclasses; JSON load/save; fail-loud schema validation; the orientation/hemisphere metadata.
- **`atlas.py`** — thin wrapper over `brainglobe-atlasapi`: load Allen Mouse CCF annotation + structure tree; `region_at(ccf_xyz) → (acronym, name, id)`; `coarse_region(acronym) → {CP, GPe, CTX, WM/fiber, VS/ventricle, other, out}`; `border_distance(ccf_xyz)` (distance to nearest differing-region voxel) for confidence.
- **`channel_map.py`** — parse `channel_positions.npy` / KS `chanMap` → `(channel, probe_shank_index, x, y)`; `chanmap_signature()` = stable hash of the sorted active-site geometry (so sessions sharing a bank share one atlas row-set); assert per-subject geometry constant up to y-offset.
- **`orientation.py`** (or folded into `channel_map.py`) — map `probe_shank_index` → anatomical shank / the correct track, using `barcode_orientation` + `hemisphere`; the validation guard of §7.
- **`localize.py`** — core: place each channel at its `y_um` on the chosen shank polyline (arc-length interpolation from the calibrated tip) → CCF; assemble the per-channel atlas; the uncertainty model of §8.
- **`peak_channel.py`** — per-unit peak channel: **RawWaveforms primary** (channel of max peak-to-peak on the per-unit average; RawWaveforms live alongside `channel_positions.npy` in the UnitMatch input dirs), **KS `templates.npy` fallback**.

**Scripts (`scripts/anatomy/`):**
- `run_brainreg.py` — wrap the `brainreg` CLI (volume + orientation + atlas → registered volume).
- `import_track.py` — adapt `brainglobe-segmentation` / Pinpoint export → `<subject>_shank_tracks.json`.
- `build_channel_atlas.py --subject BG_046` — track artifact + that subject's session chanmaps → `<subject>_channel_atlas.csv`.
- `localize_units.py --subject BG_046` — peak channels + channel atlas → per-unit location; merge columns into `build_unit_table`.
- `plot_shank_anatomy.py --subject BG_046` — QC figure (§9).

**SOP doc:** `docs/anatomy/registration_recipe.md` — the human brainreg → trace → (Pinpoint) → `import_track.py` workflow, including how `barcode_orientation`/`hemisphere` are read from the implant documentation.

## 7. Probe orientation (barcode forward/backward) — correctness, not cosmetics

The probe is implanted with its 4 shanks spread along ~ML (one end medial, the other lateral), inserted ~vertically (DV); the electrode/barcode face normal points along ~AP. **In some mice the barcode faces anterior, in others posterior** (documented per mouse). A barcode flip is a **180° rotation about the DV insertion axis** → the **medial↔lateral order of the shank row reverses**. (The two within-shank columns lie in the ML–DV plane, so the flip produces no meaningful AP offset — it is a clean medial/lateral swap.)

Because `channel_positions.npy` knows only the *probe* shank index, a naive `probe_shank_index → ML` mapping would **silently swap medial and lateral in half the mice**.

**Turning "barcode forward" into an absolute "shank 0 = medial" requires two facts that are recorded as data, not assumed in code:** (1) the IMEC/SpikeGLX shank-numbering-vs-barcode convention, and (2) the implant **hemisphere** (medial direction depends on it). The vendor convention is recorded once in the SOP/config and validated, never hard-coded from assumption.

**Design (robust by construction):**
- For `brainreg_traced` shanks, medial/lateral order is **empirical** — the traced shank-tip ML coordinates are ground truth. `barcode_orientation` + `hemisphere` are used to **associate which traced track belongs to which `probe_shank_index`**.
- For `extended_from_tip` / `pinpoint_planned` shanks (no full tracing to anchor on), the orientation flag + planned geometry **determine** the shank's ML placement.
- **Validation guard (fail-loud):** the (orientation-corrected) shank-tip ML coordinates must be **monotonic** in shank index and spaced **~250 µm**. A convention entered backward then surfaces as a failed check, not a silent medial/lateral swap.

## 8. Uncertainty model (a first-class output)

Given low-to-moderate registration precision and upward extension, per-channel uncertainty is required — it is what lets the deliverable say *"shank 2 crosses cortex→striatum near channel 40, ±X µm."*

- **Along-track σ:** from `brainreg`-trace residuals for `brainreg_traced` segments; for `extended_from_tip` segments σ **grows with distance above the deepest dye point** (`σ(d) = σ0 + k·d`); `pinpoint_planned` shanks carry a fixed insertion-error σ (~100–150 µm, recorded as a configurable default).
- **Per-channel output:** a **hard region label** (argmax) **plus** `region_confidence` = P(true location ∈ assigned region) from the track σ convolved against `border_distance` in the annotation volume. Border channels are flagged; optionally top-2 regions are stored.
- The QC figure renders each shank's cortex→striatum (and striatum→GPe) crossing with its ±σ band.

## 9. Validation

- **BG_038 positive control:** must come back **GPe** (per `memory/qc_celltype_yield_investigation_jun2026.md`). If it does not, the pipeline or the tracing is wrong.
- **BG_046 sanity:** the QC figure must show a plausible **cortex → white-matter (corpus callosum) → striatum** descent along depth, with the **upper (extended) channels carrying visibly larger σ** than the dye-traced lower channels.
- **Orientation guard:** §7 monotonicity/spacing check passes for every subject.
- **Chanmap invariance:** assert per-subject geometry constant up to y-offset; distinct chanmap signatures correspond to distinct depth windows only.

## 10. Testing

- **Synthetic track artifact + synthetic chanmap → known region mapping** (a hand-built shank polyline through a toy annotated volume; channels at known depths must return the expected regions and monotonic depths).
- **`chanmap_signature` stability** (same geometry → same hash; shifted y-offset → different signature; reordered channels → same signature).
- **Peak-channel** on a synthetic RawWaveform stack with a known peak channel; KS-template fallback path.
- **Schema fail-loud:** malformed / incomplete track artifact raises with a clear message.
- **Orientation guard:** a deliberately back-to-front orientation triggers the monotonicity failure.

## 11. Persistence & integration (Phase A)

- **Sidecars** under `data/anatomy/`: `<subject>_shank_tracks.json` (input), `<subject>_probe_config.json` (input), `<subject>_channel_atlas.csv` (derived, keyed by `chanmap_signature`), and a session→signature map.
- **`build_unit_table` columns added:** `peak_channel, shank, depth_um, ccf_ap, ccf_ml, ccf_dv, region_acronym, region_name, region_coarse, region_confidence, loc_method`.
- **PKLs / `Cluster` untouched** in Phase A. **Phase B** (deferred) adds the same fields to `Cluster` and bakes them in on the next re-ingest — additive, not a rewrite.

## 12. Inputs the user must supply (not now — at run time)

- The reconstructed serial-section **volume** per subject (for `brainreg`), with orientation/voxel-spacing metadata.
- The traced shank tracks (the napari step) → fed through `import_track.py`.
- The documented **`barcode_orientation` (forward/backward) and implant `hemisphere`** per subject.
- Planned stereotaxic **insertion coordinates + angle** per subject (for extension / Pinpoint shanks).

## 13. Open items / risks

- **Vendor shank-numbering convention** must be pinned once (SOP) and is guarded by §7 — until pinned, treat medial/lateral as empirically derived from tracing.
- **Volume reconstruction quality** from hand-cut sections is the main upstream risk to `brainreg`; the QC figure + BG_038 control are the backstops.
- **Extension length** for very-partial tracks (e.g. BG_046) drives σ; if the dye segment is short, the planned-angle assumption dominates and confidence should reflect that.
- **`brainglobe-atlasapi` install** on this Windows machine to be verified at planning time (pure-Python; atlas downloads cached).

## 14. Compute & cluster execution

Steps split cleanly into interactive (must stay local, needs a display) vs heavy-but-headless (can go to SLURM):

- **`brainreg` volumetric deformable registration** is the one compute-heavy, **non-interactive** step (CPU/RAM-bound; worse at 10 µm than 25 µm atlas). It can run on the **ceph SLURM `cpu` partition**, which per `memory/unitmatch_multisubject_jun2026.md` is **uncapped** (`MaxMemPerNode=UNLIMITED`, 512 GB nodes, `MaxTime=10d`). `run_brainreg.py` is written CLI/unattended so it drops straight into an sbatch wrapper analogous to `slurm/run_unitmatch_subject.sbatch`. Atlas downloads via `brainglobe-atlasapi` are cached (do once where there is network, e.g. local, then ship the cache).
- **napari track tracing** (`brainglobe-segmentation`) and **Pinpoint** are **interactive** → run on a local/workstation display, **never** on the cluster.
- The **in-repo consumer** (channel mapping, peak channel, atlas lookups, unit-table merge, QC plots) is light → runs locally. Only escalate to SLURM if a future whole-volume or 10 µm operation proves heavy.

Cluster path mirrors the existing setup: conda env on ceph, `cpu` partition, `--mem` auto-routes to the 512 GB nodes. See `memory/unitmatch_multisubject_jun2026.md` for the partition/resource notes and `memory/worktree_*` for the editable-install / data-junction gotchas if run from a worktree.

## 15. References

- Lab heritage: `scripts/conversion/matlab_scripts/NPX-analysis-master/analysis/Brain_regions/` (`get_units_per_regions_probe.m`, `addSimpleNamesBrainReg.m`, Allen structure-tree CSVs).
- BrainGlobe: `brainreg`, `brainglobe-segmentation`, `brainglobe-atlasapi` (Allen Mouse CCF). Pinpoint / Neuropixels Trajectory Explorer for planned trajectories.
- Project: `memory/qc_celltype_yield_investigation_jun2026.md` (BG_038 = GPe), `memory/proposal_aims.md` (PPC/aMOs/pMOs→D1/D2 scope), `memory/unitmatch_multisubject_jun2026.md` (RawWaveforms extraction + `channel_positions.npy` layout), `src/visdetect/core/session.py` (current `Cluster`).
