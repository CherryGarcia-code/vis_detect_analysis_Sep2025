"""Generate shank_assignment.png using helpers from run_unitmatch_pair.py

This script imports derive_shanks_from_channel_positions and
_save_shank_assignment_plot from the runner module and writes the
PNG into the report_dir from config/unitmatch_sessions.yml.
"""

from pathlib import Path
import yaml
import numpy as np

# Import helpers from the runner
from run_unitmatch_pair import (
    derive_shanks_from_channel_positions,
    _save_shank_assignment_plot,
)

# Load config
cfg = yaml.safe_load(Path("config/unitmatch_sessions.yml").read_text())
ks_list = [s["path"] for s in cfg.get("sessions", [])]
if len(ks_list) == 0:
    raise SystemExit("No sessions found in config")

first_ks = Path(ks_list[0])
ch_file = first_ks / "channel_positions.npy"
if not ch_file.exists():
    raise SystemExit(f"channel_positions.npy not found at {ch_file}")

cp = np.load(ch_file)
# Derive shanks
no_shanks, shank_dist, channel_shanks = derive_shanks_from_channel_positions(
    cp, expected_shanks=4
)
print("Derived:", no_shanks, shank_dist)

# Ensure output dir
out_dir = Path(cfg.get("report_dir", "table_output/unitmatch"))
out_dir.mkdir(parents=True, exist_ok=True)
png_path = out_dir / "shank_assignment.png"

# Save plot (if derivation failed, still attempt to make a simple scatter)
if channel_shanks is not None:
    _save_shank_assignment_plot(cp, channel_shanks, str(png_path))
    print("Wrote", png_path)
else:
    # fallback: save raw channel positions
    import matplotlib.pyplot as plt

    x = cp[:, 0]
    y = cp[:, 1] if cp.shape[1] > 1 else np.zeros_like(x)
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y)
    for i, (xx, yy) in enumerate(zip(x, y)):
        plt.text(xx, yy, str(i), fontsize=6, alpha=0.6)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Channel positions (no shanks derived)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print("Wrote fallback", png_path)
