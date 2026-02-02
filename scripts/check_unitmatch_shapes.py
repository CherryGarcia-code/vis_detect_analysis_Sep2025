import numpy as np
from pathlib import Path

# Define the base directory
base_dir = Path(r"e:\python_analysis\git_repos\vis_detect_analysis_Sep2025\data\unit_match\input\BG_046\01072025")

# Files to check
files_to_check = {
    "RawWaveform (0.npy)": base_dir / "RawWaveforms" / "0.npy",
    "channel_positions": base_dir / "channel_positions.npy",
    "channel_map": base_dir / "channel_map.npy"
}

print(f"Checking shapes in: {base_dir}\n")

for name, file_path in files_to_check.items():
    if file_path.exists():
        try:
            data = np.load(file_path)
            print(f"{name}: {data.shape}")
        except Exception as e:
            print(f"{name}: Error loading file - {e}")
    else:
        print(f"{name}: File not found at {file_path}")
