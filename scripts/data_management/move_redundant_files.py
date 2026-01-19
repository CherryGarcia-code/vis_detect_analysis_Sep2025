"""
Script to move files identified as 'FOUND' in the redundancy check to a 'to_delete' folder.
Mirrors the directory structure in the destination.
Removes empty directories in the source after moving.

Usage:
    python scripts/data_management/move_redundant_files.py --csv table_output/backup_redundancy_check.csv --backup-root X:/.../raw_backup --to-delete-root X:/.../raw_backup/to_delete
"""

import argparse
import os
import sys
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def move_files(df, backup_root, to_delete_root):
    print(f"Loading files to move from backup: {backup_root}")
    print(f"Destination for redundant files: {to_delete_root}")
    
    backup_root = Path(backup_root).resolve()
    to_delete_root = Path(to_delete_root).resolve()
    
    if not args.dry_run:
        to_delete_root.mkdir(parents=True, exist_ok=True)
    
    # Filter for FOUND files
    found_files = df[df['status'] == 'FOUND'].copy()
    
    print(f"Found {len(found_files)} files marked as FOUND to move.")
    
    moved_count = 0
    errors = []
    
    for idx, row in tqdm(found_files.iterrows(), total=len(found_files), unit="file"):
        src = Path(row['backup_file']).resolve()
        
        # Safety check: ensure file is inside backup_root
        if not src.is_relative_to(backup_root):
            errors.append(f"File outside backup root: {src}")
            continue
            
        try:
            rel_path = src.relative_to(backup_root)
        except ValueError:
            errors.append(f"Could not determine relative path: {src}")
            continue
            
        dst = to_delete_root / rel_path
        
        if not src.exists():
            errors.append(f"Source file not found (already moved?): {src}")
            continue
            
        if args.dry_run:
            # mimic move
            pass
        else:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                moved_count += 1
            except Exception as e:
                errors.append(f"Error moving {src}: {e}")

    print(f"\nMoved {moved_count} files.")
    if errors:
        print(f"Encountered {len(errors)} errors:")
        for e in errors[:10]:
            print("  " + e)
        if len(errors) > 10:
            print("  ...")
            
    return moved_count > 0

def cleanup_empty_dirs(root_dir, exclude_dir=None):
    """
    Walks bottom-up and removes empty directories.
    """
    print("\nCleaning up empty directories in source...")
    root_dir = Path(root_dir).resolve()
    if exclude_dir:
        exclude_dir = Path(exclude_dir).resolve()

    removed_count = 0
    
    # Bottom-up traversal
    for root, dirs, files in os.walk(root_dir, topdown=False):
        current_dir = Path(root)
        
        # Don't delete the root itself
        if current_dir == root_dir:
            continue
            
        # Don't delete the destination directory if it's inside the source
        if exclude_dir and (current_dir == exclude_dir or exclude_dir.is_relative_to(current_dir)):
            # If current_dir is a parent of exclude_dir, we definitely can't delete it
            # If current_dir IS exclude_dir, we can't delete it
            continue

        try:
            # Check if empty (os.rmdir fails if not empty, but it's cleaner to check or catch)
            # We trust os.rmdir to fail if not empty, so simply try it.
            # However, os.walk lists dirs/files from the *start* of the iteration.
            # We need to check if it's actually empty now.
            if not any(current_dir.iterdir()):
                if not args.dry_run:
                    current_dir.rmdir()
                removed_count += 1
        except OSError:
            # Directory not empty or other error
            pass
            
    print(f"Removed {removed_count} empty directories.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move redundant files to a to_delete folder.")
    parser.add_argument('--csv', required=True, help="Path to the redundancy check CSV.")
    parser.add_argument('--backup-root', required=True, help="Root folder of the backup (source).")
    parser.add_argument('--to-delete-root', required=True, help="Destination folder for redundant files.")
    parser.add_argument('--dry-run', action='store_true', help="Print actions without executing moves.")
    
    args = parser.parse_args()
    
    if not Path(args.csv).exists():
        print(f"CSV not found: {args.csv}")
        sys.exit(1)
        
    df = pd.read_csv(args.csv)
    
    did_move = move_files(df, args.backup_root, args.to_delete_root)
    
    if did_move or args.dry_run:
        cleanup_empty_dirs(args.backup_root, exclude_dir=args.to_delete_root)
