# visdetect package

This package contains the core logic for the visual change detection analysis.

## Structure

- **core/**: Core data structures and I/O.
  - `session.py`: `Session`, `Trial`, `Cluster` dataclasses + unified pickle/MAT loader with full legacy format support.
  - `io.py`: MAT file loading.
  - `qc.py`: Quality control functions.
  - `kilosort.py`: Kilosort data integration.

- **analysis/**: Analysis modules.
  - `behavior.py`: **Canonical** SDT metrics, d', psychometrics (single source of truth).
  - `align.py`: Event alignment and PETHs.
  - `decoding.py`: Population decoding.
  - `tuning.py`: Tuning curve analysis.
  - `responsiveness.py`: Unit responsiveness metrics.
  - `population.py`: Population dynamics.
  - `tf_pulse.py`: TF pulse analysis.
  - `lick_decoding.py`: Lick-aligned decoding.
  - `lick_responsiveness.py`: Lick responsiveness.
  - `optotagging.py`: Optotagging analysis.
  - `su_analysis.py`: Single unit analysis helpers.
  - `coding_direction.py`: Coding direction analysis.

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
