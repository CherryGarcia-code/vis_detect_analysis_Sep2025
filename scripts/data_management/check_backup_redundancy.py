"""
Script to check for redundancy between a backup folder and a parent data folder.
Purpose: To identify which files in 'raw_backup' already exist in the parent folder (moved/organized),
so that the backup can be safely cleaned up.

Usage:
    python scripts/data_management/check_backup_redundancy.py --backup X:/.../raw_backup --parent X:/.../wEPhys --out table_output/backup_check.csv
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

def get_file_info(filepath):
    """Returns (filename, size_in_bytes) tuple."""
    try:
        stat = os.stat(filepath)
        return (filepath.name, stat.st_size)
    except OSError:
        return None

def build_parent_index(search_root, exclude_dir):
    """
    Builds an index of files in search_root, excluding the exclude_dir tree.
    Returns:
        dict: {(filename, size): [list_of_absolute_paths]}
    """
    print(f"\n[1/3] Indexing parent folder: {search_root}")
    print(f"      Excluding backup folder: {exclude_dir}")
    
    index = defaultdict(list)
    search_root = Path(search_root).resolve()
    exclude_dir = Path(exclude_dir).resolve()
    
    file_count = 0
    
    for root, dirs, files in os.walk(search_root):
        root_path = Path(root).resolve()

        # Logic to skip the exclude_dir completely
        # Modify 'dirs' in-place to prevent os.walk from entering exclude_dir
        # We look for the folder name in 'dirs' that allows us to reach exclude_dir
        
        # Check if exclude_dir is a direct subdirectory of current root
        indices_to_remove = []
        for i, d_name in enumerate(dirs):
            d_full = root_path / d_name
            if d_full == exclude_dir:
                indices_to_remove.append(i)
        
        if indices_to_remove:
            for i in sorted(indices_to_remove, reverse=True):
                print(f"      Skipping excluded folder: {root_path / dirs[i]}")
                del dirs[i]
        
        # Safety: Double check we aren't inside the excluded dir (e.g. through symlinks or path logic overlap)
        if exclude_dir == root_path or exclude_dir in root_path.parents:
            continue

        for f in files:
            full_path = root_path / f
            info = get_file_info(full_path)
            if info:
                index[info].append(str(full_path))
                file_count += 1
                
                if file_count % 1000 == 0:
                    print(f"      Indexed {file_count} files...", end='\r')
    
    print(f"      Finished indexing. Total files in parent (excluding backup): {file_count}")
    return index

def check_backup(backup_root, parent_index):
    """
    Walks backup_root and checks if files exist in parent_index.
    """
    print(f"\n[2/3] Checking backup folder: {backup_root}")
    backup_root = Path(backup_root).resolve()
    
    results = []
    
    files_to_check = []
    # Pre-collect files to allow for a progress bar
    print("      Collecting file list from backup...")
    for root, _, files in os.walk(backup_root):
        root_path = Path(root)
        for f in files:
            files_to_check.append(root_path / f)
            
    print(f"      Found {len(files_to_check)} files in backup. processing...")
    
    found_count = 0
    missing_count = 0
    
    for backup_file_path in tqdm(files_to_check, unit="file"):
        info = get_file_info(backup_file_path)
        
        if not info:
            # Could not access file
            results.append({
                "backup_file": str(backup_file_path),
                "filename": backup_file_path.name,
                "size_bytes": -1,
                "status": "ERROR_ACCESS",
                "matches_in_parent_count": 0,
                "matches": ""
            })
            continue
            
        matches = parent_index.get(info)
        
        if matches:
            status = "FOUND"
            found_count += 1
            csv_matches = ";".join(matches)
            n_matches = len(matches)
        else:
            status = "MISSING"
            missing_count += 1
            csv_matches = ""
            n_matches = 0
        
        results.append({
            "backup_file": str(backup_file_path),
            "filename": info[0],
            "size_bytes": info[1],
            "status": status,
            "matches_in_parent_count": n_matches,
            "matches": csv_matches
        })
            
    print(f"      Done. Found: {found_count}, Missing: {missing_count}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Check backup redundancy.")
    parser.add_argument('--backup', required=True, help="Path to the backup folder (source of truth to check).")
    parser.add_argument('--parent', required=True, help="Path to the parent folder (where files should exist).")
    parser.add_argument('--out', required=True, help="Path to save the CSV report.")
    
    args = parser.parse_args()
    
    backup_path = Path(args.backup)
    parent_path = Path(args.parent)
    
    if not backup_path.exists():
        print(f"Error: Backup path does not exist: {backup_path}")
        sys.exit(1)
        
    if not parent_path.exists():
        print(f"Error: Parent path does not exist: {parent_path}")
        sys.exit(1)
        
    # Ensure output dir exists
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Index Parent
    index = build_parent_index(parent_path, exclude_dir=backup_path)
    
    # 2. Check Backup against Index
    results = check_backup(backup_path, index)
    
    # 3. Save Report
    print(f"\n[3/3] Saving report to {out_path}")
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    
    # Summary
    print("\nSummary:")
    print(f"Total backup files checked: {len(df)}")
    if not df.empty and 'status' in df.columns:
        print(df['status'].value_counts().to_string())
    
    print("\nNext steps:")
    print("1. Review the generated CSV.")
    print("2. Filter for status='MISSING' to see what was not found in the parent structure.")
    print("3. Filter for status='FOUND' to confirm what is safe to delete from backup.")

if __name__ == "__main__":
    main()
