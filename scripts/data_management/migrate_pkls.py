"""
Migrate legacy session .pkl files to the current package structure.

This script loads sessions (using the backward-compatible loader) and re-saves them.
This updates the internal module paths (e.g. 'src.visdetect' -> 'visdetect') so they
can be opened natively without custom unpicklers in the future.

Usage:
    python scripts/misc_utils/migrate_pkls.py --pkl-dir pkls/BG_046 --overwrite
"""
import argparse
import sys
from pathlib import Path
from tqdm import tqdm


from visdetect.core.session import load_session, save_session

def main():
    parser = argparse.ArgumentParser(description="Migrate legacy pkl files to current format.")
    parser.add_argument('--pkl-dir', required=True, help='Directory containing .pkl files')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files. If not set, saves as .new.pkl')
    args = parser.parse_args()

    pkl_dir = Path(args.pkl_dir)
    pkl_files = list(pkl_dir.glob("*.pkl"))
    
    if not pkl_files:
        print(f"No .pkl files found in {pkl_dir}")
        return

    print(f"Found {len(pkl_files)} files. Starting migration...")
    
    success_count = 0
    
    for pkl_path in tqdm(pkl_files):
        try:
            # 1. Load (uses the compatibility shim)
            session = load_session(str(pkl_path))
            
            # 2. Determine output path
            if args.overwrite:
                out_path = pkl_path
            else:
                out_path = pkl_path.with_suffix('.new.pkl')
            
            # 3. Save (uses current class definitions)
            save_session(session, str(out_path))
            success_count += 1
            
        except Exception as e:
            print(f"Failed to migrate {pkl_path.name}: {e}")

    print(f"\nMigration complete. Successfully updated {success_count}/{len(pkl_files)} files.")
    if not args.overwrite:
        print("Note: Files were saved with .new.pkl extension. Verify them before renaming.")

if __name__ == "__main__":
    main()
