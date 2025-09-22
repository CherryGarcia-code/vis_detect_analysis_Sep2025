# Instructions for Copilot

You are assisting with analysis and visualization of cortico-basal ganglia recordings from a visual change-detection task in mice.

## Tasks to Prioritize

- **Data Wrangling:**  
  - Create Python data classes to represent sessions, trials, neural data, and behavioral events using the provided JSON schema.
  - Ensure classes are flexible for batch processing and easy access to trial-level and unit-level data.

- **Analysis:**  
  - Align neural data (spike times, clusters) with behavioral events (trial outcomes, stimulus changes).
  - Compute and visualize peri-event time histograms (PETHs) for different trial outcomes.
  - Group neurons by response patterns (e.g., clustering, PCA).
  - Track units across sessions if possible.

- **Dimensionality Reduction:**
  - Apply dimensionality reduction techniques (e.g., PCA, t-SNE, UMAP, and coding direction analyses) to neural population activity.
  - Use these methods to visualize and interpret high-dimensional neural data, identify patterns, and relate neural trajectories to behavioral events.


- **Visualization:**  
  - Plot trial-by-trial neural and behavioral data (heatmaps, raster plots, summary stats).
  - Visualize learning curves and changes in neural activity over training.

## Coding Preferences

- Use modern Python (3.9+), pandas, numpy, matplotlib, seaborn, and scikit-learn.
- Avoid deprecated functions.
- Write clear, well-commented code.
- Where possible, use type hints and docstrings.

## Data Reference

- Refer to `RAW_SESSION_SCHEMA_BG_031_260325.json` for field names and data structure.
- If unsure about a field, ask for clarification or suggest a reasonable default.

## Research Focus

- Prioritize analyses that address the research questions listed in the `README.md`.
- Suggest additional analyses or visualizations if they could provide new insights.
