# visdetect package

This package contains the core logic for the visual change detection analysis.

## Structure

- **core/**: Core data structures and I/O.
  - `session.py`: `Session`, `Trial`, `Cluster` dataclasses.
  - `io.py`: MAT file loading.
  - `legacy_io.py`: Pickle loading (legacy).
  - `qc.py`: Quality control functions.
  - `kilosort.py`: Kilosort data integration.

- **analysis/**: Analysis modules.
  - `align.py`: Event alignment and PETHs.
  - `decoding.py`: Population decoding.
  - `tuning.py`: Tuning curve analysis.
  - `responsiveness.py`: Unit responsiveness metrics.
  - `population.py`: Population dynamics.
  - `tracking.py`: Unit tracking (UnitMatch).
  - `tf_pulse.py`: TF pulse analysis.
  - `lick_decoding.py`: Lick-aligned decoding.
  - `lick_responsiveness.py`: Lick responsiveness.
  - `optotagging.py`: Optotagging analysis.
  - `su_analysis.py`: Single unit analysis helpers.
  - `coding_direction.py`: Coding direction analysis.
  - `responsive_analysis.py`: Responsive unit analysis.

- **plotting/**: Visualization helpers (currently empty, logic in analysis modules).

- **utils/**: Shared utilities.
  - `synthetic.py`: Synthetic data generation.
  - `progress.py`: Progress bar helper.
  - `matlab_ports/`: Python ports of MATLAB logic.
  - `integrations/`: External tool integrations (Bombcell).

## Usage

```python
from visdetect import Session
from visdetect.core.io import load_mat_file_to_session
from visdetect.analysis import align

session = load_mat_file_to_session("path/to/session.mat")
peth = align.compute_peth_for_session(session, "Change", [-1, 1], 0.01)
```
