"""
Filter sessions based on behavioral quality control criteria defined in a YAML config.

.. deprecated::
    This script is NOT part of the active analysis pipeline.  Session inclusion
    is now determined by ``scripts/analysis/stage_sessions.py`` which writes the
    staging manifest (``data/BG_046_staging_manifest.csv``).  The old
    ``config/session_qc.yml`` config has been moved to ``archive/deprecated_modules/``.

Usage (legacy):
    python scripts/data_management/filter_sessions.py --manifest data/BG_046_sessions_manifest.csv --out-dir data/filtered
"""
import argparse
import pandas as pd
import yaml
from pathlib import Path
import sys

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def apply_filters(df, rules):
    """
    Apply a list of rules to the dataframe.
    Returns:
        df_kept: DataFrame of sessions that passed all rules.
        df_excluded: DataFrame of sessions that failed, with 'exclusion_reason' column.
    """
    # Start with all included
    df['is_excluded'] = False
    df['exclusion_reason'] = ""
    
    total_excluded = 0
    
    print(f"Applying {len(rules)} QC rules...")
    
    for rule in rules:
        name = rule.get('name', 'Unnamed Rule')
        condition = rule['condition']
        reason = rule.get('reason', name)
        
        try:
            # Find rows matching the EXCLUSION condition
            # The condition in YAML defines what to EXCLUDE (e.g. "fraction_miss > 0.38")
            mask = df.eval(condition)
            
            # Only update rows that aren't already excluded (to keep the primary reason)
            # Or maybe we want to append reasons? Let's append for completeness.
            
            new_excludes = mask & ~df['is_excluded']
            count = new_excludes.sum()
            
            if count > 0:
                print(f"  - Rule '{name}' ({condition}): excluded {count} sessions.")
                
                # Mark as excluded
                df.loc[mask, 'is_excluded'] = True
                
                # Append reason
                # For rows that were already excluded, add comma
                existing_mask = mask & (df['exclusion_reason'] != "")
                df.loc[existing_mask, 'exclusion_reason'] += f"; {reason}"
                
                # For newly excluded
                new_mask = mask & (df['exclusion_reason'] == "")
                df.loc[new_mask, 'exclusion_reason'] = reason
                
        except Exception as e:
            print(f"Error evaluating rule '{name}': {e}")
            sys.exit(1)

    df_kept = df[~df['is_excluded']].copy().drop(columns=['is_excluded', 'exclusion_reason'])
    df_excluded = df[df['is_excluded']].copy().drop(columns=['is_excluded'])
    
    return df_kept, df_excluded

def main():
    parser = argparse.ArgumentParser(description="Filter sessions based on QC criteria.")
    parser.add_argument('--manifest', required=True, help='Path to input session manifest CSV')
    parser.add_argument('--config', default='config/session_qc.yml', help='Path to QC config YAML')
    parser.add_argument('--profile', default='default', help='QC profile to use from config')
    parser.add_argument('--out-dir', required=True, help='Directory to save filtered manifests')
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return
    
    if not config_path.exists():
        repo_root = Path(__file__).resolve().parents[2]
        config_path = repo_root / args.config
        if not config_path.exists():
            print(f"Config not found: {config_path}")
            return

    # Load Data
    # Force session_name to be string to preserve leading zeros
    df = pd.read_csv(manifest_path, dtype={'session_name': str})
    # Ensure padding just in case
    df['session_name'] = df['session_name'].apply(lambda x: x.zfill(8) if x.isdigit() and len(x) == 7 else x)
    
    print(f"Loaded {len(df)} sessions from {manifest_path.name}")

    # Load Config
    config = load_config(config_path)
    if args.profile not in config:
        print(f"Profile '{args.profile}' not found in config. Available: {list(config.keys())}")
        return
    
    profile = config[args.profile]
    print(f"Using profile '{args.profile}': {profile.get('description', '')}")
    
    # Apply Filters
    df_kept, df_excluded = apply_filters(df, profile['rules'])
    
    # Save Outputs
    kept_path = out_dir / f"{manifest_path.stem}_clean.csv"
    excluded_path = out_dir / f"{manifest_path.stem}_excluded.csv"
    
    df_kept.to_csv(kept_path, index=False)
    df_excluded.to_csv(excluded_path, index=False)
    
    print("\nFiltering Complete:")
    print(f"  Total Sessions: {len(df)}")
    print(f"  Kept:           {len(df_kept)} -> {kept_path}")
    print(f"  Excluded:       {len(df_excluded)} -> {excluded_path}")

if __name__ == "__main__":
    main()
