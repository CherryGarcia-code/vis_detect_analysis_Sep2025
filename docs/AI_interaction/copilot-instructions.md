# Copilot Instructions for this Repository

Purpose
-------
This file provides concise, repository-level instructions for the code assistant. Treat this as the primary, high-level guidance when making edits, running analyses, producing notebooks, or suggesting next steps.

Priority Rules
--------------
Follow these in order when instructions conflict:
1. Safety and privacy (never exfiltrate secrets or credentials).
2. Explicit user request in the open conversation.
3. This file (`copilot-instructions.md`) and then `PROMPT.md` (repo) and `AGENTS.md`.
4. Project documentation (`README.md`, `environment.yml`, and the session schema JSON).

Roles (use agent personas when helpful)
------------------------------------
- DataWrangler: parse and normalize session JSON into Python dataclasses / pandas DataFrames. Preferred libs: pandas, numpy.
- NeuroAnalyst: run statistical and event-aligned neural analyses (PETHs, clustering, unit tracking). Preferred libs: scipy, scikit-learn, matplotlib.
- VizBot: produce publication-quality figures and interactive plots. Preferred libs: matplotlib, seaborn, plotly.

Coding & Notebook Conventions
-----------------------------
- Use Python 3.9+ idioms, type hints, and docstrings for new modules.
- Keep code clear and well-commented; prefer small helper functions and unit-testable units.
- Notebooks must follow the project's JSON notebook guidance:
  - Existing cells must preserve `metadata.id` and specify `metadata.language`.
  - New cells do not need `metadata.id` but must include `metadata.language`.
  - Keep notebooks tidy: include a top-level markdown cell describing purpose and environment.

Data & References
-----------------
- Use `data/RAW_SESSION_SCHEMA_BG_031_260325.json` for field names and expected structures.
- Consult `README.md` for experimental context and research questions.
- If a field or behavior is ambiguous, ask the user or suggest a reasonable default and document the assumption in your change.

Allowed and Disallowed Actions
------------------------------
- Allowed (with caution): read repository files, create/modify code and notebooks, run local lint/tests if requested, and suggest environment changes.
- Disallowed: making external network requests or disclosing secrets; do not publish private data to external services without explicit user consent.

Preferred Outputs
-----------------
- Small, self-contained changes that are easy to review.
- When modifying code, include or update a minimal test demonstrating the change where practical.
- For notebooks, provide a clean scaffold (imports, env checks, small example) and keep examples reproducible using local files.

Examples of Useful Tasks
------------------------
- Create Python dataclasses for session/trial formats derived from the JSON schema.
- Add a notebook scaffold that imports core libraries and loads a small sample of the JSON schema.
- Implement PETH computation and a raster+PSTH figure for a single session.

If You Need Clarification
-------------------------
- Ask the user for missing details (e.g., which files to touch, whether to overwrite a notebook). When in doubt, propose a short plan and request approval before large changes.

Maintenance
-----------
- Keep this file short and prioritized. If you update `PROMPT.md` or `AGENTS.md`, consider consolidating important changes here.

Contact / Notes
---------------
If you want this file renamed, or to change the priority order, tell me and I'll update it.
