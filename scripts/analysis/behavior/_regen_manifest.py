"""Quick script to regenerate the v2 manifest and compare."""
import sys, os

from scripts.analysis.stage_sessions import stage_sessions

stage_sessions(
    subject_dir="data/pkls/BG_046",
    subject_name="BG_046",
    output_csv="data/BG_046_staging_manifest_v2_new.csv",
)
