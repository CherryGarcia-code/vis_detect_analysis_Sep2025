"""CCF <-> stereotaxic (Bregma-referenced) coordinate conversion + plot mapping.

Allen ``allen_mouse_25um`` CCF coordinates are absolute microns measured from a
corner of the reference volume: AP increases posterior, ML increases toward the
LEFT hemisphere (CCF ML > 5700 um = left), DV increases ventral. Paxinos /
stereotaxic coordinates instead reference Bregma (AP, DV) and the midline (ML),
which is why the raw CCF axis numbers look unfamiliar.

``BREGMA_AP_UM`` is the community-standard CCF location of Bregma (IBL /
Pinpoint). Cross-checked in this project and self-consistent at 5400 um:
  - BG_046 slice CCF-AP 5150 um -> Bregma +0.25 mm  (Paxinos Fig 29)
  - BG_039 slice CCF-AP 4921 um -> Bregma +0.48 mm  (Paxinos Fig 27, +0.50 mm)
"""
from __future__ import annotations

import numpy as np

BREGMA_AP_UM = 5400.0     # CCF AP of Bregma
MIDLINE_ML_UM = 5700.0    # CCF ML of the midline (half the 11400 um ML extent)


def ap_to_bregma_mm(ccf_ap_um):
    """CCF AP (um) -> mm relative to Bregma (+ = anterior to Bregma)."""
    return (BREGMA_AP_UM - np.asarray(ccf_ap_um, float)) / 1000.0


def ml_to_lateral_mm(ccf_ml_um):
    """CCF ML (um) -> signed mm from the midline. |value| = lateral distance;
    sign follows CCF ML (which increases toward the LEFT hemisphere)."""
    return (np.asarray(ccf_ml_um, float) - MIDLINE_ML_UM) / 1000.0


def dv_to_depth_mm(ccf_dv_um, pia_dv_um):
    """CCF DV (um) -> mm below the brain surface (pia) at the penetration site."""
    return (np.asarray(ccf_dv_um, float) - float(pia_dv_um)) / 1000.0


def pia_dv_um(track_artifact):
    """Brain-surface (pia) DV in um = mean of the shallowest traced point across
    shanks. Polylines are deepest-first, so row -1 is the cortical surface."""
    surf = [float(np.asarray(s.ccf_polyline, float)[-1, 2]) for s in track_artifact.shanks]
    return float(np.mean(surf)) if surf else 0.0


class CoordMap:
    """Map CCF (ML, DV) microns -> plot coordinates.

    mode='ccf' (default): identity (microns); image untouched; CCF axis labels.
    mode='stereotaxic': millimetres referenced to the midline (ML) and the brain
    surface (DV). ML is FLIPPED so the anatomical LEFT hemisphere sits on the LEFT
    of the plot (neurological convention) and signed left<0 / right>0; DV becomes
    depth below pia (dorsal still up). Lengths (scale bars) convert um -> mm.
    """

    def __init__(self, mode: str = "ccf", pia_dv_um: float = 0.0):
        if mode not in ("ccf", "stereotaxic"):
            raise ValueError(f"coords mode {mode!r} not in ('ccf', 'stereotaxic')")
        self.mode = mode
        self.stereo = mode == "stereotaxic"
        self.pia = float(pia_dv_um)

    def x(self, ml_um):
        """CCF ML (um) -> plot x. Stereotaxic: (midline - ml)/1000 so the left
        hemisphere (ml > midline) maps to negative x and is drawn on the left."""
        ml = np.asarray(ml_um, float)
        return (MIDLINE_ML_UM - ml) / 1000.0 if self.stereo else ml

    def y(self, dv_um):
        """CCF DV (um) -> plot y (depth below pia in mm when stereotaxic)."""
        dv = np.asarray(dv_um, float)
        return (dv - self.pia) / 1000.0 if self.stereo else dv

    def length(self, um):
        """A length (e.g. a 500 um scale bar) in plot units."""
        return um / 1000.0 if self.stereo else um

    def image(self, img, extent_um):
        """Transform a coronal image and its [ml0, ml1, dv1, dv0] (um) extent.
        Stereotaxic mode mirrors the image L-R (np.fliplr over the ML axis) so it
        matches the flipped ML axis."""
        if not self.stereo:
            return img, extent_um
        ml0, ml1, dv1, dv0 = extent_um
        return np.fliplr(img), [self.x(ml1), self.x(ml0), self.y(dv1), self.y(dv0)]

    def ap_title(self, ap_um) -> str:
        if self.stereo:
            return f"Bregma {float(ap_to_bregma_mm(ap_um)):+.2f} mm"
        return f"AP ≈ {ap_um:.0f} µm"

    @property
    def xlabel(self) -> str:
        return "ML from midline (mm)  (− left · + right)" if self.stereo else "ML (µm)"

    @property
    def ylabel(self) -> str:
        return "Depth below brain surface (mm)" if self.stereo else "DV (µm)"
